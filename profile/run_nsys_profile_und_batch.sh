#!/bin/bash

export cuda_visible_devices=0
project_root="/workspace/Bagel"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root
nsys profile \
  -o ./profile/profile_result/Image_Und_without_think-512x512-short_question_prompt-$(date +%Y%m%d_%H%M%S) \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --python-sampling=true \
  --sample=cpu \
  --force-overwrite=true \
  --stats=true \
  --cuda-memory-usage=true \
  python ./scripts/inference/image_und_batch.py \
    --model_path ./models/BAGEL-7B-MoT \
    --prompt_pth ./data_profile/prompt/understand.csv \
    --output ./results/image_und \
    --seed 42 \
    # --think \
    # --do_sample \


echo ""
echo "============================================"
echo "Profiling finished"