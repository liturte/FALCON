#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import torch
import glob
import logging
import random
import fnmatch
import numpy as np
import gc
import os
from tqdm import tqdm 
from collections import Counter
import pickle as pkl 
import json, pdb 
from collections import defaultdict
from multiprocessing import Manager
import transformers

import datasets.utils as dsutils


class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, model, max_tokens, sample_mode, 
                 tuning_mode, max_src_tokens, relative_returns, args):
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs 

        self.model = model
        self.sample_mode = sample_mode
        self.tuning_mode = tuning_mode
        self.relative_returns = relative_returns
        
        self.max_tokens = max_tokens
        self.max_src_tokens = max_src_tokens
        self.args = args
        self.samples = []           
        self.all_error_types, self.all_error_subtypes, self.all_baseline_error_types = [], [], []
        self.initialize()

        if self.model in ['codet5-base', 'codet5-large']:
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained(args.tokenizer_path)
    def load_pkl_file(self, file_path, baseline_error_type=-1):
        gen_samples = []
        samples_info = []

        if os.path.isfile(file_path) and file_path.endswith('.pkl'):
            try:
                with open(file_path, 'rb') as file:
                    pkl_data = pkl.load(file)

                for key, inner_data in pkl_data.items():
                    if isinstance(inner_data, dict) and 'results' in inner_data and 'sols' in inner_data:
                        inner_data['code'] = inner_data['sols']
                        samples, info = self.new_load_rl_samples(inner_data, baseline_error_type)
                        gen_samples += samples
                        samples_info += info
                    else:
                        print(f"Missing required fields in file {file_path}, key: {key}. Skipping.")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        else:
            print(f"Invalid file path or non-pkl file: {file_path}")

        return gen_samples, samples_info



    
    def load_gen_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        info = []
        
        for idx, sol in enumerate(sols):
            sol_str = sol['code']  
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample) 
            
            result = sol['result']
            error_type = sol['error_type']
            
            info.append((result, error_type))
            
        return samples, info 
    
    def load_rl_samples(self, sols, baseline_error_type): 
        samples = []
        info = []
        
        for idx, code in enumerate(sols['code']):   
            samples.append((sols['prompt'][idx], code))
            info.append((sols['gt_error_type'][idx], sols['error_hidden_states'][idx], baseline_error_type))
            
        return samples, info 

    def new_load_rl_samples(self, sols, baseline_error_type):
        samples = []
        info = []

        for idx, code in enumerate(sols['code']):
            samples.append(("111", code))

            # Use 'results' as gt_error_type, set others to None
            gt_error_type = sols['results'][idx] if idx < len(sols['results']) else None
            info.append((gt_error_type, None, baseline_error_type))

        return samples, info

    def load_gt_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        
        for sol_str in sols:
            sol_str = dsutils.reindent_code(sol_str)
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample)
        
        return samples 
    
    def get_gt_info(self):
        return (1, None)

    def get_baseline_error_type(self, sols): 
        return dsutils.get_error_type(sols[0]['result'])
    
    
    def update_error_stat(self, info):
        for i in info:
            error_type = dsutils.get_error_type(i[0])
            error_subtype = i[1]
            self.all_error_types.append(error_type)
            self.all_error_subtypes.append(error_subtype) 
            
    def update_error_stat_rl(self, info):
        for i in info:
            error_type = i[0]
            baseline_error_type = i[-1]
            self.all_error_types.append(error_type)   
            self.all_baseline_error_types.append(baseline_error_type) 
    
    def initialize(self):
        all_samples = []
        skipped_problems = []
        samples_info = [] 
        gen_samples = [] 
        
        all_samples_dict = {} 

        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):           
            if self.tuning_mode in ['critic']:                
                gen_sols_fname = [os.path.join(self.dataroot, problem_name, "gen_solutions.json")]       

            elif self.tuning_mode in ['rl']:
                gen_sols_fname = [os.path.join(self.dataroot, problem_name, "gen_solutions_critic_scores.pkl")]   
                
                if self.relative_returns: 
                    baseline_fname = os.path.join(self.dataroot, problem_name, "baseline_solutions.json")
                
            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")            
            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue
                
            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()
            
            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")    
            if (os.path.isfile(starter_code)):
                answer_type = "\nUse Call-Based format\n"
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                answer_type = "\nUse Standard Input format\n"
                starter_code = ""

            sols_str_list = json.load(open(sols_fname, 'r'))
            gt_samples = self.load_gt_samples(sols_str_list, answer_type, starter_code, question_str)
            all_samples += gt_samples 
                
            # Read all the solutions
            if self.tuning_mode in ['critic']: 
                for fname in gen_sols_fname:
                    if os.path.exists(fname):
                        gen_sols = json.load(open(fname, 'r'))
                        samples, info = self.load_gen_samples(gen_sols, answer_type, starter_code, question_str) 
                        self.update_error_stat(info)
                        gen_samples += samples
                        samples_info += info
                
                # also include ground-truth samples to train critic model; assume success test outcomes 
                gen_samples += gt_samples
                info = [self.get_gt_info() for s in gt_samples]
                samples_info += info
                
            elif self.tuning_mode in ['rl']: 
                
                if self.relative_returns:
                    baseline_sample = json.load(open(baseline_fname, 'r'))
                    baseline_error_type = self.get_baseline_error_type(baseline_sample)
                else:
                    baseline_error_type = -1 
            new_name = problem_name.lstrip('0')
            new_name = new_name if new_name else '0'
            pkl_path = os.path.join(self.args.unit_test_path, f"{new_name}.pkl")
            
            samples, info = self.load_pkl_file(
                pkl_path,
                baseline_error_type=-1 
            )
            # 加载评分文件
            score_file = os.path.join(self.args.ai_feedback_path, f"{new_name}.pkl")
            if os.path.exists(score_file):
                with open(score_file, 'rb') as f:
                    scores = pkl.load(f)
                coding_style_scores = scores['Coding Style']
                complexity_scores = scores['Complexity']
                instruction_scores = scores['Instruction Following']

                for idx, item in enumerate(info):
                    original_info = list(item)  
                    additional_scores = [
                        coding_style_scores[idx],      
                        complexity_scores[idx],        
                        instruction_scores[idx]        
                    ]
                    original_info.append(additional_scores)  
                    info[idx] = tuple(original_info)         #


            gen_samples += samples
            samples_info += info
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        
        if self.tuning_mode in ['critic', 'rl']:
            if len(gen_samples) != len(samples_info): pdb.set_trace()
            print(f"Loaded {len(gen_samples)} generated samples from {self.dataroot}.")
            print("Error type distribution: {}".format(sorted(Counter(self.all_error_types).items())))
            if self.tuning_mode == 'critic':
                print("Error subtype distribution: {}".format(sorted(Counter(self.all_error_subtypes).items())))
            else:
                print("Baseline error distribution: {}".format(sorted(Counter(self.all_baseline_error_types).items())))
        else:
            print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
            
        self.samples = all_samples
        self.samples_info = samples_info 
        self.gen_samples = gen_samples 
        
        if self.relative_returns: 
            self.all_error_types = np.array(self.all_error_types)
            self.all_baseline_error_types = np.array(self.all_baseline_error_types)
            print('Sampled Error > Baseline error: {}/{}'.format((self.all_error_types>self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            print('Sampled Error = Baseline error: {}/{}'.format((self.all_error_types==self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            print('Sampled Error < Baseline error: {}/{}'.format((self.all_error_types<self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            
            sample_rewards = np.array([dsutils.get_reward_from_error_type(e) for e in self.all_error_types])
            baseline_rewards = np.array([dsutils.get_reward_from_error_type(e) for e in self.all_baseline_error_types])
            print("Relative returns distribution: {}".format(sorted(Counter(sample_rewards-baseline_rewards).items())))
            
    def __len__(self):
        if self.tuning_mode in ['critic', 'rl']:
            return len(self.gen_samples)
        return len(self.samples)

    def __getitem__(self, idx):
        
        if self.tuning_mode in ['critic']:
            raw_samples = self.pack_samples(idx, 'gen')
            inputs = self.sample_task(raw_samples, 'gen')
         
        elif self.tuning_mode in ['rl']:
            gt_sample_idx = random.randint(0, len(self.samples)-1)
            raw_gt_samples = self.pack_samples(gt_sample_idx)
            inputs = self.sample_task(raw_gt_samples)
            
            item = self.gen_samples[idx]
            info = self.samples_info[idx]

            gen_inputs = self.sample_rl_task(item, info)
            for k,v in gen_inputs.items():
                inputs['rl_{}'.format(k)] = v 
            
        else:
            raw_samples = self.pack_samples(idx)
            inputs = self.sample_task(raw_samples)

        gc.collect()
        return inputs
    def get_reward_from_error_type(self, error_type):
        if error_type == -2:
            # Compile error or critical failure
            return -1
        elif error_type == -1:
            # Runtime error
            return -0.6
        elif error_type == 0:
            # Failed unit tests
            return -0.3
        elif error_type == 1:
            # Passed all unit tests
            return 1
        else:
            raise NotImplementedError(f"Error type {error_type} not handled.")
    def pack_samples(self, idx, sample_type=None):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = [] 
        
        if sample_type == 'gen':
            sample_pool = self.gen_samples
        else:
            sample_pool = self.samples
        
        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[idx]
            if self.tuning_mode in ['critic'] and sample_type=='gen': 
                curr_result, curr_error_subtype = self.samples_info[idx] 
                
        elif self.sample_mode == 'uniform_prob':
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:

            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))            
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))
            
            if self.tuning_mode in ['critic'] and sample_type=='gen': 
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix, 
                                     curr_result, curr_error_subtype))
                break 
     
            else:
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))
                
                # only pack 1 sample each sequence for codeT5 
                if self.model in ['codet5-base', 'codet5-large']:
                    break 

            if self.sample_mode == 'uniform_sol':
                new_idx = random.randint(0, len(sample_pool)-1)
                curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[new_idx] 
            elif self.sample_mode == 'uniform_prob':
                raise NotImplementedError()

        return curr_samples

    
    def sample_task(self, samples, sample_type=None):

        input_ids = []
        label_ids = []
                    
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            error_types = [] 
                    
        for sample in samples:
            if self.tuning_mode in ['critic'] and sample_type == 'gen': 
                q_str, s_str, a_str, answer_type, result, error_subtype = sample
            else:
                q_str, s_str, a_str, answer_type = sample
            
            q_str =  "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            input_ids.extend(question_token_ids)
             
            answer_token_ids = self.tokenizer.encode(a_str, verbose=False)
            if self.model not in ['codet5-base', 'codet5-large']:
                label_ids.extend([-100] * len(question_token_ids))
                answer_token_ids.append(self.tokenizer.eos_token_id)
                input_ids.extend(answer_token_ids)
            label_ids.extend(answer_token_ids)

            if self.tuning_mode in ['critic'] and sample_type == 'gen':
                error_types.append(dsutils.get_error_type(result))
                
        # Sanity checks and padding 
        input_ids_max_len = self.max_src_tokens if self.model in ['codet5-base', 'codet5-large'] else self.max_tokens 
        if len(input_ids) < input_ids_max_len: 
            new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids 
            
            if self.model not in ['codet5-base', 'codet5-large']:
                new_label_ids = [-100] * input_ids_max_len 
                new_label_ids[:len(label_ids)] = label_ids
                label_ids = new_label_ids
                
        if self.model in ['codet5-base', 'codet5-large'] and len(label_ids) < self.max_tokens:
            new_label_ids = [-100] * self.max_tokens 
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids
        
        if self.model not in ['codet5-base', 'codet5-large'] and len(input_ids) != len(label_ids): pdb.set_trace()
            
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            assert len(error_types) == 1 
            
        # Cut off the excess
        input_ids = input_ids[:input_ids_max_len]
        label_ids = label_ids[:self.max_tokens]
        
        out_sample = {
            "input_ids" : torch.LongTensor(input_ids),
            "labels" :  torch.LongTensor(label_ids)
        }
        
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            out_sample['error_types'] = torch.LongTensor(error_types)            
            
        return out_sample 

    def sample_rl_task(self, item, info):
        input_ids, labels = item

        if len(info) == 3:
            gt_error, _, baseline_error = info
            additional_scores = [0, 0, 0]
        elif len(info) == 4:
            gt_error, _, baseline_error, additional_scores = info
        else:
            raise ValueError(f"Unexpected info structure: {info}")

        if isinstance(gt_error, list):
            gt_error = gt_error[0] if len(gt_error) > 0 else -1

        if gt_error is None:
            gt_error = -1
        if labels is None or not isinstance(labels, (list, np.ndarray)):
            labels = [-100] * 512  

        rewards = torch.tensor([self.get_reward_from_error_type(gt_error)] * len(labels), dtype=torch.float32)
        additional_reward = sum(additional_scores) if isinstance(additional_scores, list) else 0
        rewards += additional_reward

        if self.relative_returns:
            baseline_reward = baseline_error if baseline_error != -1 else 0
            rewards *= (rewards[0].item() - baseline_reward)

        max_length = 512

        if input_ids is None:
            input_ids = [self.tokenizer.eos_token_id] * max_length
        elif isinstance(input_ids, str):
            input_ids = self.tokenizer.encode(input_ids, add_special_tokens=False)[:max_length]
        elif not isinstance(input_ids, (list, np.ndarray)):
            input_ids = [self.tokenizer.eos_token_id] * max_length

        if labels is None:
            labels = [-100] * len(input_ids)
        elif isinstance(labels, str):
            labels = self.tokenizer.encode(labels, add_special_tokens=False)[:max_length]
        elif not isinstance(labels, (list, np.ndarray)):
            labels = [-100] * len(input_ids)

        if len(input_ids) < max_length:
            input_ids.extend([self.tokenizer.eos_token_id] * (max_length - len(input_ids)))
            labels.extend([-100] * (max_length - len(labels)))

            if len(rewards) < max_length:
                rewards = torch.cat((rewards, torch.zeros(max_length - len(rewards))), dim=0)

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        rewards = rewards[:max_length]

        return {
            "input_ids": torch.LongTensor(input_ids),
            "rewards": rewards,
            "label_ids": torch.LongTensor(labels),
        }
