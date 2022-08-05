#!/bin/bash

epochs=5
print_every=10
train_loader_size=800
save_steps=200
lr=2e-5
warmup_proportion=0.1
num_thread=4

while getopts ":e:p:t:s:l:w:n:" opt
do 
    case $opt in
        e)
            epochs=$OPTARG
            ;;
        p)
            print_every=$OPTARG
            ;;
        t)
            train_loader_size=$OPTARG
            ;;
        s)
            save_steps=$OPTARG
            ;;
        l)
            lr=$OPTARG
            ;;
        w)
            warmup_proportion=$OPTARG
            ;;
        n)
            num_thread=$OPTARG
            ;;
        ?)
            echo "未知参数"
            exit 1;;
    esac
done 

echo $epochs
echo $print_every
echo $train_loader_size
echo $save_steps
echo $lr
echo $warmup_proportion
echo $num_thread

num_checkpoints=`expr $train_loader_size \* $epochs / $save_steps`
echo $num_checkpoints