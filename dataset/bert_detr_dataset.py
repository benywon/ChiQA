# -*- coding: utf-8 -*-
from PIL import Image 
from dataset.dataset_utils import (
    list_to_tensor, train_transform,
    test_transform
)
import torch 


def read_qpic_line(line, tokenizer, max_len=64):
    query, _, label, img_path = line[:-1].split('\t')
    image = Image.open(img_path).convert('RGB') 
    image = train_transform(image)
    label = float(label) / 2
    query = tokenizer.encode(query, add_special_tokens=True)[:64]
    input_ids = [101,] + query + [102,]
    attention_mask = [1,] * len(input_ids)
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image': image,
        'label': label,
    }
    return inputs 


def collect_qpic_data(data):
    input_ids = [d['input_ids'] for d in data]
    attention_mask = [d['attention_mask'] for d in data]
    images = [d['image'].unsqueeze(0) for d in data]
    labels = [d['label'] for d in data]

    max_steps = max([len(d) for d in input_ids])

    input_ids = list_to_tensor(input_ids, max_steps)
    attention_mask = list_to_tensor(attention_mask, max_steps)
    image = torch.cat(images, dim=0) # B x C x H x W
    labels = torch.tensor(labels, dtype=torch.float)
    
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image': image,
        'labels': labels
    }
    return inputs 


def read_qpic_test_line(line, tokenizer=None, max_len=64):
    query, _, label, img_path = line[:-1].split('\t')
    image = Image.open(img_path).convert('RGB') 
    image = test_transform(image)
    query = tokenizer.encode(query, add_special_tokens=True)[:64]
    input_ids = [101,] + query + [102,]
    attention_mask = [1,] * len(input_ids)
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image': image,
        'label': label,
    }
    return inputs 


def collect_qpic_test_data(data):
    input_ids = [d['input_ids'] for d in data]
    attention_mask = [d['attention_mask'] for d in data]
    images = [d['image'].unsqueeze(0) for d in data]

    max_steps = max([len(d) for d in input_ids])

    input_ids = list_to_tensor(input_ids, max_steps)
    attention_mask = list_to_tensor(attention_mask, max_steps)
    image = torch.cat(images, dim=0) # B x C x H x W
    
    inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image': image,
    }
    return inputs 