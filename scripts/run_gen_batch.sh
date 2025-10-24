#!/bin/bash

conda init
conda activate /hdd/yuke/fanjiang/conda_env/bagel
cd /scr/dataset/yuke/fanjiang/repo/unified-model/Bagel
export PYTHONPATH=$PWD:$PYTHONPATH

export cuda_visible_devices=0

python ./scripts/inference_batch/image_gen_batch.py \
  --model_path ./models/BAGEL-7B-MoT \
  --prompt_pth ./data_profile/prompt/text2image.csv \
  --output ./results/image_gen \
  --seed 42 \
  --think False \
  --do_sample False \
  --image_shapes 256 256


