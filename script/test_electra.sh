train_file="../Molweni/DP(500)/train.json"
eval_file="../Molweni/DP(500)/dev.json"
test_file="../Molweni/DP(500)/test.json"
glove_file="../glove.6B.200d.txt"
dataset_dir="../dataset/Molweni"
model_dir="./Molweni"

GPU=0
model_name=model
task=student

CUDA_VISIBLE_DEVICES=${GPU} python main_electra.py --train_file=$train_file --test_file=$test_file \
                                               --dataset_dir=$dataset_dir \
                                               --eval_pool_size 10 --glove_embedding_size 200 \
                                               --model_path "${model_dir}/${model_name}.pt" \
                                               --teacher_model_path "${model_dir}/teacher_model.pt" \
                                               --num_layers 3 --max_edu_dist 16 \
                                               --task ${task}
