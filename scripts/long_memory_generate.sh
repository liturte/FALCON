

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

BASE_PATH="/data/coding/FALCON"

TEST_PATH="/data/coding/FALCON/data/APPS/test"  
OUTPUT_DIR="${BASE_PATH}/outputs/long_term_generate"  
MODEL_PATH="/data/coding/model/codeT5"  

NUM_SEQS=5
START_IDX=0
END_IDX=10
TEMPERATURE=0.6

python generate_with_long_memory.py \
    --test_path "${TEST_PATH}" \
    --output_path "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --code_output "${BASE_PATH}/outputs/deepseek_test_result" \
    --feedback_path "${BASE_PATH}/outputs/deep_ai_feedback" \
    --task_path "${BASE_PATH}/outputs/deep_codes" \
    --num_seqs ${NUM_SEQS} \
    --start ${START_IDX} \
    --end ${END_IDX} \
    --temperature ${TEMPERATURE}

    