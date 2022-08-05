# -*- coding: utf-8 -*-
import torch 
import torch.nn.functional as F 
from torchvision import transforms
from dataset.randaugment import RandomAugment
from PIL import Image 


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


def list_to_tensor(ids_lst, max_steps, dtype=torch.long, value=0):
    ids = [torch.tensor(lst[:max_steps], dtype=dtype) for lst in ids_lst]
    ids = [F.pad(t, pad=(0, max_steps-t.size(0)), value=value).reshape(1, -1) for t in ids]
    ids = torch.cat(ids, dim=0)
    return ids 