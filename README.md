[**中文说明**](./README.zh.md) | [**English**](./README.md)

This repository contains the data, baseline implementations for our CIKM 2022 long paper [ChiQA: A Large Scale Image-based Real-World Question Answering Dataset for Multi-Modal Understanding](https://arxiv.org/pdf/2208.03030.pdf).

## prerequisite for running the baselines
`pip install -r requirement.txt`

## Training and inference
An example of training and testing bert-detr model, which utilize the pre-trained language model [BERT](https://arxiv.org/abs/1810.04805) and a SOTA object detection model [DETR](https://arxiv.org/abs/2005.12872) for cross-model representation.   
`sh run_bert_detr.sh -e 5 -p 10 -t 800 -s 200 -l 2e-5 -w 0.1 -n 4`  
arguments：
- e: epochs
- p: print_steps, how many steps to show a log
- t: train_loader_size, batch size per epoch
- s: how many batches should we save the checkpoint.
- l: learning rate
- w: warmup_proportion
- n: num threads to process the data

### BERT+ViT
`sh run_bert_vit.sh -e 5 -p 10 -t 800 -s 200 -l 2e-5 -w 0.1 -n 4`

### ALBEF  
`sh run_albef.sh -e 5 -p 10 -t 800 -s 200 -l 2e-5 -w 0.1 -n 4`  

## Data processing
The raw data of ChiQA will be released soon.