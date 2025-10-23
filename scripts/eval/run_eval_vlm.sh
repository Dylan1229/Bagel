# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

conda init
conda activate /hdd/yuke/fanjiang/conda_env/bagel

set -x

cd /scr/dataset/yuke/fanjiang/repo/unified-model/Bagel
# Set proxy and API key
export OPENAI_API_KEY=$openai_api_key
model_path="/scr/dataset/yuke/fanjiang/repo/unified-model/Bagel/models/BAGEL-7B-MoT"
output_path="/scr/dataset/yuke/fanjiang/repo/unified-model/Bagel/results"
export GPUS=1
export CUDA_VISIBLE_DEVICES=1

# DATASETS=("mme" "mmbench-dev-en" "mmvet" "mmmu-val" "mathvista-testmini" "mmvp")
# DATASETS=("mmmu-val_cot")
DATASETS=("mme")
DATASETS_STR="${DATASETS[*]}"
export DATASETS_STR

# Create log directory if not exists
log_dir="/scr/dataset/yuke/fanjiang/repo/unified-model/Bagel/eval/logs"
mkdir -p "$log_dir"

# Generate log file name with timestamp
log_file="$log_dir/eval_vlm_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: $log_file"

bash scripts/eval/eval_vlm.sh \
    $output_path \
    --model-path $model_path \
    2>&1 | tee "$log_file"
