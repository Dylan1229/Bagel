#!/bin/bash
export cuda_visible_devices=0
project_root="/mnt"
if [[ ":$PYTHONPATH:" != *":$project_root:"* ]]; then
  export PYTHONPATH="$project_root:$PYTHONPATH"
fi

cd $project_root
ncu --set speedOfLight    \
    --export "./profile/profile_result/nsys_profile_gen_batch"  \
    --kernel-name "regex:(sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas|index_elementwise_kernel|flash_fwd_kernel|vectorized_elementwise_kernel|elementwise_kernel|unrolled_elementwise_kernel|sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas|sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x256x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas|reduce_kernel|)" \
    -f \
    -c 20 \
  python ./scripts/inference_batch/image_gen_batch.py \
    --model_path ./models/BAGEL-7B-MoT \
    --prompt_pth ./data_profile/prompt/text2image.csv \
    --output ./results/image_gen \
    --seed 42 \
    --think True \
    --do_sample False

echo "============================================"
echo "Ncu Profiling finished"