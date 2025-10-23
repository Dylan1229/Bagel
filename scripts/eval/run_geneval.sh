# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

conda init
conda activate /hdd/yuke/fanjiang/conda_env/bagel

set -x
cd /scr/dataset/yuke/fanjiang/repo/unified-model/Bagel

export GPUS=1
export CUDA_VISIBLE_DEVICES=4
model_path="/scr/dataset/yuke/fanjiang/repo/unified-model/Bagel/models/BAGEL-7B-MoT"
output_path="/scr/dataset/yuke/fanjiang/repo/unified-model/Bagel/results/imggen"
# generate images
torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS \
    --master_addr=127.0.0.1 \
    --master_port=12345 \
    ./eval/gen/gen_images_mp.py \
    --output_dir $output_path/images \
    --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata_long.jsonl \
    --batch_size 1 \
    --num_images 1 \
    --resolution 1024 \
    --max_latent_size 64 \
    --model-path $model_path \
    # --metadata_file ./eval/gen/geneval/prompts/evaluation_metadata.jsonl \


# calculate score
# torchrun \
#     --nnodes=1 \
#     --node_rank=0 \
#     --nproc_per_node=$GPUS \
#     --master_addr=127.0.0.1 \
#     --master_port=12345 \
#     ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
#     $output_path/images \
#     --outfile $output_path/results.jsonl \
#     --model-path ./eval/gen/geneval/model


# summarize score
# python ./eval/gen/geneval/evaluation/summary_scores.py $output_path/results.jsonl