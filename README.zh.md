# ChiQA
ChiQA: A Large Scale Image-based Real-World Question Answering Dataset for Multi-Modal Understanding

## 运行依赖
安装运行依赖命令  
`pip install -r requirement.txt`

## 模型训练和测试
训练并测试bert-detr模型  
`sh run_bert_detr.sh -e 5 -p 10 -t 800 -s 200 -l 2e-5 -w 0.1 -n 4`  
参数说明：
- e: epochs
- p: print_steps，训练多少batch打印一下日志
- t: train_loader_size，一个epoch里训练多少个batch
- s: 训练多少个batch保存一次模型参数
- l: learning rate，学习率大小
- w: warmup_proportion，学习率线性增加的训练时间占比
- n: 生产训练数据的线程个数

训练并测试bert-vit模型   
`sh run_bert_vit.sh -e 5 -p 10 -t 800 -s 200 -l 2e-5 -w 0.1 -n 4`  
训练参数说明同上。

训练并测试albef模型  
`sh run_albef.sh -e 5 -p 10 -t 800 -s 200 -l 2e-5 -w 0.1 -n 4`  
训练参数说明同上。