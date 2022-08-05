# -*- coding: utf-8 -*-
import torch 
import argparse
import os 
from utils import load_file
from transformers import AutoTokenizer
from dataset.bert_detr_dataset import (
    read_qpic_test_line, collect_qpic_test_data 
)
from models.bert_detr import BertDetrQPicModel
from apex import amp 
from concurrent.futures import ThreadPoolExecutor
from functools import partial 


def run(args):

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    args.vocab_size = tokenizer.vocab_size
    read_qpic_test_line_p = partial(read_qpic_test_line, tokenizer=tokenizer)

    model = BertDetrQPicModel(args)

    load_model_path = os.path.abspath(args.load_model_path)
    if os.path.exists(load_model_path):
        model = load_file(load_model_path, model, args)
        print('{} model load succeed from {}'.format(model.__class__.__name__, load_model_path))
    else:
        print('{} model not exist in {}'.format(model.__class__.__name__, load_model_path))
    print('num. model params: {}'.format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )) 
    model.eval()
    model.to(device)

    test_prediction_path = os.path.abspath(args.test_prediction_path)
    test_dataset_path = os.path.abspath(args.test_dataset_path)
    batch = []

    with open(test_prediction_path, 'w', encoding='utf-8') as f_test:
        with open(test_dataset_path, 'r', encoding='utf-8') as f_ori:
            for lin in f_ori:
                batch.append(lin)
                if len(batch) < args.batch_size:
                    continue
                inputs = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for d in executor.map(read_qpic_test_line_p, batch):
                        inputs.append(d)
                inputs = collect_qpic_test_data(inputs)
                inputs = {k: inputs[k].to(device) for k in inputs}
                with torch.no_grad(): 
                    probs = model(**inputs) # B  
                    probs = probs.cpu().numpy().tolist() # B 
                
                for p in probs:
                    new_lin = '{}'.format(
                        p 
                    )
                    f_test.write(new_lin + '\n')
                batch = []
            if len(batch) > 0:
                inputs = []
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for d in executor.map(read_qpic_test_line_p, batch):
                        inputs.append(d)
                inputs = collect_qpic_test_data(inputs)
                inputs = {k: inputs[k].to(device) for k in inputs}
                with torch.no_grad():
                    probs = model(**inputs) # B 
                    probs = probs.cpu().numpy().tolist() # B 
                
                for p in probs:
                    new_lin = '{}'.format(
                        p 
                    )
                    f_test.write(new_lin + '\n')
                batch = []


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='add argument to test')

    parser.add_argument('--hidden-dim', type=int, default=768)
    parser.add_argument('--bert-config-file', type=str, default='configs/config_bert_detr.json')
    parser.add_argument('--text-encoder-model', type=str, default='bert-base-chinese')
    parser.add_argument('--image-encoder-model', type=str, default='facebook/detr-resnet-50')

    parser.add_argument('--load-model-path', type=str, default='')
    parser.add_argument('--test-dataset-path', type=str, default='')
    parser.add_argument('--test-prediction-path', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size to test')
    parser.add_argument('--cpu', action='store_true', help='whether run model in cpu')

    args = parser.parse_args()
    print(args)

    run(args)
