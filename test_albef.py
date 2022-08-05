# -*- coding: utf-8 -*-
import torch 
import argparse
import os 
from transformers import AutoTokenizer
from dataset.albef import (
    read_qpic_test_line, collect_qpic_test_data
)
from models.albef import ALBEF
from concurrent.futures import ThreadPoolExecutor
import ruamel.yaml as yaml
from utils import load_file


def run(args):

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_model)
    args.vocab_size = tokenizer.vocab_size

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    model = ALBEF(config=config, text_encoder=args.text_encoder_model, tokenizer=tokenizer)

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
                    for d in executor.map(read_qpic_test_line, batch):
                        inputs.append(d)
                inputs = collect_qpic_test_data(inputs, tokenizer)
                
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
                    for d in executor.map(read_qpic_test_line, batch):
                        inputs.append(d)
                inputs = collect_qpic_test_data(inputs, tokenizer)
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

    parser = argparse.ArgumentParser(description='add argument to lstm trainer')

    parser.add_argument('--config', default='./configs/albef.yaml')
    parser.add_argument('--text-encoder-model', type=str, default='bert-base-chinese')

    parser.add_argument('--load-model-path', type=str, default='save_checkpoints/null')
    parser.add_argument('--test-dataset-path', type=str, default='')
    parser.add_argument('--test-prediction-path', type=str, default='')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size to test')
    parser.add_argument('--cpu', action='store_true', help='whether run model in cpu')

    args = parser.parse_args()
    print(args)

    run(args)
