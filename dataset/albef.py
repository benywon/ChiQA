# -*- coding: utf-8 -*-
import json 
import torch 
from PIL import Image 
from torchvision import transforms
from dataset.randaugment import RandomAugment
from dataset.dataset_utils import list_to_tensor

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    
train_transform = transforms.Compose([                        
    transforms.RandomResizedCrop(224,scale=(0.5, 1.0), interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(),
    RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                        'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
    transforms.ToTensor(),
    normalize,
])

test_transform = transforms.Compose([
    transforms.Resize((224,224),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
])


def read_qpic_line(line):
    query, _, label, img_path = line[:-1].split('\t')
    image = Image.open(img_path).convert('RGB')
    image = train_transform(image)
    label = float(label) / 2
    inputs = {
        'image': image,
        'text': query,
        'label': label,
    }
    return inputs


def collect_qpic_data(data, tokenizer):
    images = [d['image'].unsqueeze(0) for d in data] 
    texts = [d['text'] for d in data]
    labels = [d['label'] for d in data]

    text_input = tokenizer(texts, padding='longest', truncation=True, max_length=64, return_tensors="pt")
    image = torch.cat(images, dim=0) # B x T x C
    labels = torch.tensor(labels, dtype=torch.float)

    inputs = {
        'image': image,
        'input_ids': text_input.input_ids,
        'attention_mask': text_input.attention_mask,
        'labels': labels
    }
    return inputs


def read_qpic_test_line(line):
    query, _, label, img_path= line[:-1].split('\t')
    image = Image.open(img_path).convert('RGB')
    image = test_transform(image)
    label = float(label) / 2
    inputs = {
        'image': image,
        'text': query,
        'itm_label': label,
    }
    return inputs 


def collect_qpic_test_data(data, tokenizer):
    images = [d['image'].unsqueeze(0) for d in data] 
    texts = [d['text'] for d in data]

    text_input = tokenizer(texts, padding='longest', truncation=True, max_length=64, return_tensors="pt")
    image = torch.cat(images, dim=0) # B x T x C

    inputs = {
        'image': image,
        'input_ids': text_input.input_ids,
        'attention_mask': text_input.attention_mask
    }
    return inputs