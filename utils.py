# -*- coding: utf-8 -*-
import base64
from io import BytesIO
from PIL import Image
import math 
import time 
import pickle
import torch 


def load_file(filename, model, args):
    with open(filename, 'rb') as f:
        state_dict = pickle.load(f)
    for name, para in model.named_parameters():
        if name not in state_dict:
            if args.local_rank <= 0:
                print('{} not load'.format(name))
            continue
        para.data = torch.FloatTensor(state_dict[name])
    return model 


def dump_file(obj, filename):
    f = open(filename, 'wb')
    pickle.dump(obj, f)


def base64_pillow(base64_str):
    image = base64.b64decode(base64_str)
    image = BytesIO(image)
    image = Image.open(image).convert('RGB')
    return image


def as_minutes(s):
    m = math.floor(s / 60)
    h = math.floor(m / 60)
    s  -= m * 60
    m -= h * 60

    return '%dh %dm %ds' % (h, m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since 
    es = s / (percent)
    rs = es - s 
    
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def union_metrics(ori_m, new_m):
    if ori_m is None:
        return new_m
    for k in ori_m:
        ori_m[k] += new_m[k]
    return ori_m


def average_metrics(metrics, steps):
    m = {}
    for k in metrics:
        m[k] = metrics[k] / steps
    return m 