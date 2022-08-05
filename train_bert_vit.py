# -*- coding: utf-8 -*-
import torch 
import argparse
from models.bert_vit import BertViTQPicModel
from transformers import AutoTokenizer
from dataset.bert_vit_dataset import (
    read_qpic_line, collect_qpic_data
)
from models.module_utils import get_lr_schedule_fn
import numpy as np 
import torch.optim as optim 
import time 
from utils import (
    time_since, union_metrics, average_metrics, 
    load_file, dump_file 
)
import os 
import torch.distributed as dist 
from queue import Queue
import threading 
import random 
import deepspeed 


def gen_train_data(thread_id, qu, args, lock):
    tokenizer_thread = AutoTokenizer.from_pretrained("bert-base-chinese")
    train_dir = os.path.abspath(args.train_dir_path)
    train_files = [file_ for d, file_ in enumerate(os.listdir(train_dir)) if d % dist.get_world_size() == dist.get_rank()]
    if thread_id == 0:
        print('rank: {}, train_files: {}'.format(dist.get_rank(), train_files))
    data_list = []
    while True:
        random.shuffle(train_files)
        for file_ in train_files:
            filepath = os.path.join(train_dir, file_)
            i = -1
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    i += 1
                    if i % args.num_thread != thread_id:
                        continue 
                    try:
                        data = read_qpic_line(line, tokenizer_thread)
                        if data is None:
                            continue
                    except Exception as e:
                        print('read data exception: ', e)
                        continue 
                    data_list.append(data)
                    if len(data_list) >= args.batch_size:
                        batch = data_list[:args.batch_size]
                        batch = collect_qpic_data(batch,)
                        data_list = data_list[args.batch_size:]
                        lock.acquire()
                        qu.put(batch)
                        lock.release()

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt 


def train(model, inputs):

    loss = model(**inputs)
    model.backward(loss)
    model.step() 

    outputs = {
        'loss': loss,
    }

    return outputs 


def run():

    global args, global_step

    args.distributed = True 
    args.gpu = args.local_rank 
    args.world_size = dist.get_world_size()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    args.vocab_size = tokenizer.vocab_size

    model = BertViTQPicModel(args)

    load_model_path = os.path.abspath(args.load_model_path)
    if os.path.exists(load_model_path):
        model = load_file(load_model_path, model, args)
        print('{} model load succeed from {}'.format(model.__class__.__name__, load_model_path))
    else:
        print('{} model not exist in {}'.format(model.__class__.__name__, load_model_path))
    
    if args.local_rank == 0:
        print(model)
        print('num. model params: {}, num. training params: {}'.format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ))

    lr_schedule_fn = get_lr_schedule_fn(args.lr_scheduler_name)
    global_step = args.global_step
    train_loader_size = args.train_loader_size
    save_steps = args.save_steps
    global_epoch = global_step // train_loader_size
    args.num_train_steps = train_loader_size * args.epochs 
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model, optimizer, _, _ = deepspeed.initialize(
        args=args, model=model, optimizer=optimizer
    )
    args.batch_size = model.train_micro_batch_size_per_gpu()
    if args.local_rank == 0:
        print(args)

    lock = threading.Lock()
    qu = Queue(args.queue_size)
    for t in range(args.num_thread):
        p = threading.Thread(target=gen_train_data, args=(t, qu, args, lock))
        p.start()
    time.sleep(10)
    
    for i in range(global_epoch, args.epochs):
        model.train()
        st = time.time()
        train_metrics, train_steps = None, 0
        j = -1
        if global_step > i * train_loader_size:
            j = global_step % train_loader_size
        while j < train_loader_size:
            try:
                inputs = qu.get(block=True, timeout=60)
                qu.task_done()
            except Exception as e:
                print('rank: {}, exception: {}'.format(args.local_rank, e))
                break 
            j += 1
            if j < 1 and args.local_rank == 0:
                print(inputs)
        
            inputs = { k:inputs[k].to(args.gpu) for k in inputs }
            train_m = train(model, inputs)
            train_metrics = union_metrics(train_metrics, train_m) 
            train_steps += 1
            global_step += 1
            torch.cuda.synchronize()
            new_lr = args.lr 
            if lr_schedule_fn is not None:
                new_lr = args.lr * lr_schedule_fn(global_step / args.num_train_steps, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

            if (j+1) % args.print_every == 0:
                train_metrics = average_metrics(train_metrics, train_steps)
                train_metrics = { k: train_metrics[k].item() for k in train_metrics }
                if args.local_rank == 0:
                    print('Epoch: {}, step: {} / {}, {}, train loss: {:.4f}, lr: {:.10f}'.format(
                        i+1, j+1, train_loader_size, time_since(st, (j+1) / train_loader_size), train_metrics['loss'],
                        optimizer.param_groups[0]['lr'],
                    ))
                train_metrics = None 
                train_steps = 0

            if global_step % save_steps == 0:
                if args.local_rank != 0:
                    dist.barrier()
                if args.local_rank == 0:
                    save_model_path = os.path.join(
                        os.path.abspath(args.model_save_directory), 
                        'checkpoint-{}.pt'.format(global_step)
                    )
                    output = {}
                    for name, param in model.module.named_parameters():
                        output[name] = param.data.cpu().numpy()
                    dump_file(output, save_model_path)
                    print('Succeed save model in {}'.format(save_model_path))
                if args.local_rank == 0:
                    dist.barrier()
                model.train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='add argument to lstm trainer')
    
    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--bert-config-file', type=str, default='configs/config_bert_vit.json')
    parser.add_argument('--text-encoder-model', type=str, default='bert-base-chinese')
    parser.add_argument('--vit-encoder-model', type=str, default='google/vit-base-patch16-224-in21k')

    parser.add_argument('--load-model-path', type=str, default='./save_checkpoints/bert_vit_base_models/checkpoint-best.pt')
    parser.add_argument('--model-save-directory', type=str, default='./save_checkpoints/bert_vit_base_models')
    parser.add_argument('--train-dataset-path', type=str, default='./data/train')
    parser.add_argument('--train-dir-path', type=str, default='./data/train_dir')
    parser.add_argument('--tensorboard-dir', type=str, default='./logs/bert_vit_base_models')
    parser.add_argument('--num-thread', type=int, default=20)
    parser.add_argument('--queue-size', type=int, default=200)

    parser.add_argument('--lr_schedule', type=str, default='LE', help='Choices LE, EE, EP (L: Linear, E: Exponetial, P: Polynomial warmup and decay)')
    parser.add_argument('--lr_offset', type=float, default=0.0, help='Offset added to lr.')
    parser.add_argument("--cpu_optimizer", default=False, action='store_true',help="Whether to use cpu optimizer for training")
    
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lr-scheduler-name', type=str, default='warmup_linear', help='learning rate scheduler name')
    parser.add_argument('--warmup-proportion', type=float, default=0.001, help='learning rate warmup proportion')
    parser.add_argument('--epochs', type=int, default=50, help='maximum epochs to train')
    parser.add_argument('--batch-size', type=int, default=6, help='batch size to train/valid')
    parser.add_argument('--print-every', type=int, default=10, help='every steps to print train log')
    parser.add_argument('--global-step', type=int, default=0, help='train procedure global step')
    parser.add_argument('--train-loader-size', type=int, default=180, help='train loader size')
    parser.add_argument('--save-steps', type=int, default=60, help='save model steps')
    parser.add_argument('--max-grad-norm', type=float, default=1.0, help='max grad norm')
    parser.add_argument("--weight-decay-rate", default=0.01, type=float, help='weight_decay_rate')
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--sync_bn', action='store_true')
    parser.add_argument('--opt-level', type=str, default='O2')
    parser.add_argument('--loss-scale', type=int, default=None)
    parser.add_argument("--deepspeed_sparse_attention", default=False, action='store_true',help="Whether to use sparse attention for training")
    parser.add_argument('--job_name',type=str,default=None,help="This is the path to store the output and TensorBoard results.")
    parser.add_argument('--deepspeed_transformer_kernel', default=False, action='store_true', help='Use DeepSpeed transformer kernel to accelerate.')

    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    deepspeed.init_distributed()

    if args.local_rank == 0:
        # print(args)
        model_save_directory = os.path.abspath(args.model_save_directory)
        if not os.path.exists(model_save_directory):
            os.makedirs(model_save_directory, exist_ok=True)

    run()
