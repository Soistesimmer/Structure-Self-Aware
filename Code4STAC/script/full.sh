#!/usr/bin/env bash
train_file="../STAC/split/train.json"
eval_file="../STAC/split/eval.json"
test_file="../STAC/split/test.json"
glove_file="../glove.6B.100d.txt"
dataset_dir="../dataset/STAC"
model_dir="./STAC"

if [ ! -d "${model_dir}" ]; then mkdir -p "${model_dir}"; fi

GPU=0
model_name=full
task=distill  # student, teacher or distill
CUDA_VISIBLE_DEVICES=${GPU} python main.py --train_file=$train_file --eval_file=$eval_file --test_file=$test_file \
                                    --dataset_dir=$dataset_dir --glove_vocab_path $glove_file \
                                    --epoches 50 --batch_size 40 --pool_size 40 \
                                    --eval_pool_size 10 --report_step 30 \
                                    --save_model --overwrite --do_train \
                                    --model_path "${model_dir}/${model_name}.pt" \
                                    --teacher_model_path "${model_dir}/teacher_model.pt" \
                                    --learning_rate 0.1 --glove_embedding_size 100 \
                                    --num_layers 3 --split_long_dialogue \
                                    --task ${task} --classify_loss

#CUDA_VISIBLE_DEVICES=${GPU} nohup python -u main.py --train_file=$train_file --eval_file=$eval_file --test_file=$test_file \
#                                    --dataset_dir=$dataset_dir --glove_vocab_path $glove_file \
#                                    --epoches 50 --batch_size 40 --pool_size 40 \
#                                    --eval_pool_size 10 --report_step 30 \
#                                    --save_model --overwrite --do_train \
#                                    --model_path "${model_dir}/${model_name}.pt" \
#                                    --teacher_model_path "${model_dir}/teacher_model.pt" \
#                                    --learning_rate 0.1 --glove_embedding_size 100 \
#                                    --num_layers 3 --split_long_dialogue \
#                                    --task ${task} > ${model_name}.log 2>&1 &