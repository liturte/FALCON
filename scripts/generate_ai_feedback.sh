#!/bin/bash

# run_evaluation.sh
# 设置默认值
INPUT_DIR="/data/coding/CodeRL/outputs/deep_codes"
OUTPUT_DIR="/data/coding/CodeRL/outputs/AI_Feedback"

# 创建输出目录（如果不存在）
mkdir -p $OUTPUT_DIR

# 运行Python脚本
echo "Starting code evaluation..."
python /data/coding/CodeRL/AI_Feedback/ai_feedback_generate.py \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \


echo "Evaluation complete!"