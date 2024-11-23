##
## Copyright (c) 2022, salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
##
model_path=/data/coding/model/codeT5
tokenizer_path=/data/coding/model/codeT5/
test_path=/data/coding/CodeRL/data/APPS/test/

start=0
end=5000
num_seqs_per_iter=10
num_seqs=10
temp=0.6

output_path=/data/coding/CodeRL/outputs/codes

CUDA_VISIBLE_DEVICES=6,7 python generate.py \
    --model_path $model_path \
    --tokenizer_path $tokenizer_path \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp \