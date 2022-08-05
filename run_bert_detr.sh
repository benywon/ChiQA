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

num_checkpoints=`expr $train_loader_size \* $epochs / $save_steps`

base_dir=`pwd`
echo $base_dir
data_dir=$base_dir/data 
shuf $data_dir/train > $data_dir/zz
mkdir -p $data_dir/train_dir
split -l 1000 $data_dir/zz $data_dir/train_dir/part_

deepspeed --include='localhost:0,1,2,3,4,5,6,7' --master_addr 'localhost' --master_port 10015 ${base_dir}/train_bert_detr.py \
    --deepspeed \
    --deepspeed_config ${base_dir}/configs/ds_config.json \
    --epochs $epochs \
    --print-every $print_every \
    --train-loader-size $train_loader_size \
    --save-steps $save_steps \
    --lr $lr \
    --warmup-proportion $warmup_proportion \
    --num-thread $num_thread

ps -ef|grep train_bert_detr|awk '{print $2}'|xargs kill -9
sleep 10

data_name=test
test_data_dir=${base_dir}/data
echo $test_data_dir
model_name=bert_base_detr_models
echo $model_name
metrics_file=$test_data_dir/${data_name}.metrics.all.$model_name
echo $metrics_file
test_pred_dir=${test_data_dir}/${model_name}
echo $test_pred_dir
if [ ! -d $test_pred_dir ];
then 
    mkdir $test_pred_dir
else 
    echo "${test_pred_dir} exists."
fi 

checkpoints_dir=${base_dir}/save_checkpoints/${model_name}
echo ${checkpoints_dir}

save_step=$save_steps

for i in {1..$num_checkpoints}
do
    checkpoint_idx=`expr $i \* $save_step`
    checkpoint_path=$checkpoints_dir/checkpoint-${checkpoint_idx}.pt
    echo $checkpoint_path
    cuda_idx=`expr $i % 8`
    echo $cuda_idx
    prediction_path=$test_pred_dir/${data_name}.prediction.$checkpoint_idx    
    echo $prediction_path

    CUDA_VISIBLE_DEVICES=$cuda_idx python3 -u test_bert_detr.py \
        --load-model-path $checkpoint_path \
        --test-dataset-path $test_data_dir/${data_name} \
        --test-prediction-path $prediction_path \
        --batch-size 64 
done 

for i in {1..$num_checkpoints}
do
    checkpoint_idx=`expr $i \* $save_step`
    prediction_path=${test_pred_dir}/${data_name}.prediction.$checkpoint_idx
    echo $prediction_path
    python3 recall_metrics.py $test_data_dir/${data_name} $prediction_path >> $metrics_file
done 
