import json
import os
import openai
from openai import OpenAI
from typing import List, Dict
import pickle as pkl
# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-b7a1e5676ec34a46b71e589d283c8d84"  # 替换为你的实际 OpenAI API 密钥
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)

# 文件路径
PROBLEM_SOLUTIONS_FILE = '/data/coding/RLTF/data/Python_Seeds_with_Problem_Descriptions_and_Solutions.jsonl'
UNITTEST_RESULTS_FILE = '/data/coding/RLTF/data/test_example/output_results_3.pkl'
NEW_DATASET_FILE = '/data/coding/RLTF/data/new_dataset.jsonl'

def load_problem_and_solution(file_path, index):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    return None

def load_unittest_results(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)

def get_function_name(solution_code):
    import re
    match = re.search(r'def (\w+)\(', solution_code)
    if match:
        return match.group(1)
    return None

def update_code_with_deepseek(problem_description, solution_code, unittest_results):
    input_prompt = {
        "role": "user",
        "content": f"Update the following code based on the unittest results:\n\nProblem description:\n{problem_description}\n\nSolution code:\n{solution_code}\n\nUnittest results:\n{unittest_results}"
    }
    
    response = deepseek_client.chat.completions.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are a code assistant. Update the code based on the unittest results and provide a valid Python solution."},
            input_prompt
        ],
        temperature=0
    )

    updated_code = response.choices[0].message.content.strip()
    return updated_code, input_prompt, response.choices[0].message.content

def save_new_dataset(new_dataset: List[Dict], file_path):
    with open(file_path, 'w') as f:
        for entry in new_dataset:
            f.write(json.dumps(entry) + '\n')

def main():
    problem_index = 2  # 第三个问题的索引为2（从0开始计数）
    
    # 加载问题和解决方案
    data = load_problem_and_solution(PROBLEM_SOLUTIONS_FILE, problem_index)
    if data is None:
        print(f"Problem at index {problem_index} not found.")
        return
    problem_description = data['problem_description']
    solution_code = data['solution']
    
    # 加载单元测试结果
    unittest_results = load_unittest_results(UNITTEST_RESULTS_FILE)
    
    # 调用 DeepSeek 更新代码
    updated_code, input_prompt, response_content = update_code_with_deepseek(
        problem_description, solution_code, unittest_results
    )

    # 记录交互
    interaction_record = [
        {"role": "user", "content": input_prompt['content']},
        {"role": "assistant", "content": response_content}
    ]
    
    # 构建新数据集条目
    new_dataset_entry = {
        "problem_description": problem_description,
        "updated_code": updated_code,
        "interaction_record": interaction_record,
        "error_type": unittest_results[0].get('result', 'Unknown Error')
    }
    
    # 保存新数据集
    save_new_dataset([new_dataset_entry], NEW_DATASET_FILE)

if __name__ == '__main__':
    main()
    print(f"New dataset entry saved to {NEW_DATASET_FILE}")
