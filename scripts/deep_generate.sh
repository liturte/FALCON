##
## Copyright (c) 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
##
model_path=/data/coding/model/deepseek_instruct
tokenizer_path=/data/coding/model/deepseek_instruct
test_path=/data/coding/CodeRL/data/APPS/test/

start=0
end=4999
num_seqs_per_iter=5
num_seqs=5
temp=0.6

output_path=/data/coding/CodeRL/outputs/deep_codes

CUDA_VISIBLE_DEVICES=4,5,6,7 python deep_generate.py \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp \