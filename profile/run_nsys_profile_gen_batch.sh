#!/bin/bash
# export cuda_visible_devices=0
project_root="/workspace/Bagel"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root
nsys profile \
  -o ./profile/profile_result/T2I_Gen_with_think-512x512-short_prompt-$(date +%Y%m%d_%H%M%S) \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --python-sampling=true \
  --sample=cpu \
  --force-overwrite=true \
  --stats=true \
  --cuda-memory-usage=true \
  python ./scripts/inference/image_gen_batch.py \
    --model_path ./models/BAGEL-7B-MoT \
    --prompt_pth ./data_profile/prompt/text2image.csv \
    --output ./results/image_gen \
    --seed 42 \
    --image_shapes 512 512 \
    --think \
    # --do_sample \
    # --do_sample \


echo ""
echo "============================================"
echo "Profiling finished"

# In the docker /workspace/Bagel, run 'bash profile/run_nsys_profile_gen_batch.sh' 
