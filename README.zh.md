# ChiQA
CIKM2022长文论文 [ChiQA: A Large Scale Image-based Real-World Question Answering Dataset for Multi-Modal Understanding](https://arxiv.org/pdf/2208.03030.pdf)

## 关于ChiQA
### 背景
随着互联网的发展，用户在搜索引擎中越来越不满足于简单的文本结果，以图片、视频等为载体的“新问答”越来越受到关注。事实上，在问答系统中，很多答案的结果都可以用一个图片来简单的回答。如下图：
<p align="center">
    <br>
    <img src="medias/ir.example.png" width="360"/>
    <br>
</p>

### ChiQA
图片问答越来越重要，但是传统的VQA（visual question answering）数据很难应用在实际场景中。原因有三：

+ 传统数据的问题是人工生成的，会存在标注人员的主观或者问法的偏置
+ 传统的数据集往往是image-dependent的，即问题是看了图之后才被问的，例如：“这个图上坐在长椅上的人穿的什么颜色衣服”。
+ 传统的VQA数据重点在entity上，往往存在一定的领域单一化。

针对上面问题，我们提出了一个ChiQA数据集，包含有20万中文query-image对。我们人工标注了每条数据的相关性。

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