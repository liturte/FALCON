##
## Copyright (c) 2023 Tencent, inc, 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
##
model_path=/data/coding/model/deepseek/deepseek-coder-6___7b-base
tokenizer_path=/data/coding/model/deepseek/deepseek-coder-6___7b-base
test_path=/data/coding/RLTF/data/APPS/APPS/test/

start=501
end=1500
num_seqs_per_iter=1
num_seqs=1
temp=0.6

output_path=/data/coding/RLTF/outputs/deepseek_outputs/deepseek-coder-6___7b-base/

CUDA_VISIBLE_DEVICES=2 python generate.py \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp