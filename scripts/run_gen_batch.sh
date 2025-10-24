#!/bin/bash

conda init
conda activate /hdd/yuke/fanjiang/conda_env/bagel
cd /scr/dataset/yuke/fanjiang/repo/unified-model/Bagel

set -x 
export PYTHONPATH=$PWD:$PYTHONPATH
export cuda_visible_devices=0

timestamp=$(date +%Y%m%d_%H%M%S)
LOGFILE="./scripts/logs/run_gen_batch_${timestamp}.log"

exec >"$LOGFILE" 2>&1
  python ./scripts/inference_batch/image_gen_batch.py \
    --model_path ./models/BAGEL-7B-MoT \
    --prompt_pth ./data_profile/prompt/text2image.csv \
    --output ./results/image_gen \
    --seed 42 \
    --think False \
    --do_sample False \
    --image_shapes 256 256



