#!/usr/bin/env bash
train_file="../STAC/split/train.json"
eval_file="../STAC/split/eval.json"
test_file="../STAC/split/test.json"
glove_file="../glove.6B.100d.txt"
dataset_dir="../dataset/STAC"
model_dir="./STAC"

GPU=0
model_name=model
task=student

CUDA_VISIBLE_DEVICES=${GPU} python main.py --train_file=$train_file --test_file=$test_file \
                                               --dataset_dir=$dataset_dir \
                                               --eval_pool_size 10 --glove_embedding_size 100 \
                                               --model_path "${model_dir}/${model_name}.pt" \
                                               --teacher_model_path "${model_dir}/teacher_model.pt" \
                                               --num_layers 3 --max_edu_dist 16 \
                                               --task ${task}

#log_file=test
#CUDA_VISIBLE_DEVICES=${GPU} nohup python -u main.py --train_file=$train_file --test_file=$test_file \
#                                               --dataset_dir=$dataset_dir \
#                                               --eval_pool_size 10 --glove_embedding_size 100 \
#                                               --model_path "${model_dir}/${model_name}.pt" \
#                                               --teacher_model_path "${model_dir}/teacher_model.pt" \
#                                               --num_layers 3 \
#                                               --task ${task} > ${log_file}.log 2>&1 &
