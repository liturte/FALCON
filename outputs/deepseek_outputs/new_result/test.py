import openai
import os
import json
import pickle

# 设置OpenAI API密钥
openai.api_key = 'sk-cfc0ea9b09604b09821bde52cae856ca'
openai.api_base = 'https://api.deepseek.com/v1'

# 定义一个函数，使用deepseek-coder来分析代码错误并改进
def analyze_and_improve_code(nl, code_snippet, error_message):
    prompt = f"""
    The following is a problem description and a Python code snippet. The code snippet contains an error, as indicated by the provided error message. Please identify the issue, explain it in a single paragraph, and provide an improved version of the code.

    Problem Description:
    {nl}

    Code:
    {code_snippet}

    Error Message:
    {error_message}

    Analysis:
    Improved Code:
    """
    response = openai.ChatCompletion.create(
        model="deepseek-coder",
        messages=[
            {"role": "system", "content": "You are a Python programming assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,  # 增加max_tokens以确保有足够的空间容纳分析和改进代码
        temperature=0
    )

    response_text = response['choices'][0]['message']['content']
    print(response_text)
    # 使用'Improved Code:'作为分隔符分离分析和改进代码
    if 'Improved Code:' in response_text:
        parts = response_text.split('Improved Code:')
        analysis = parts[0].strip()
        improved_code = parts[1].strip().replace("```python", "").replace("```", "").strip()
    else:
        analysis = response_text.strip()
        improved_code = ""

    return analysis, improved_code

# 读取问题描述
def read_question_description(index):
    question_path = f"/data/coding/RLTF/data/APPS/APPS/test/{index:04d}/question.txt"
    with open(question_path, 'r') as file:
        question_description = file.read().strip()
    return question_description

# 处理0-100文件并构建新数据集
directory = '/data/coding/RLTF/outputs/deepseek_outputs/new_result/'

# Function to process and analyze each pickle file from 0-100 and save results to JSONL
def process_and_save_as_jsonl(directory, output_file):
    with open(output_file, 'w') as jsonl_file:
        for i in range(101):  # 读取0-100的pkl文件
            filename = f"{i}.pkl"
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                nl = read_question_description(i)  # 从指定路径读取问题描述
                with open(file_path, 'rb') as file:
                    data = pickle.load(file)
                    for key, value in data.items():
                        code_snippet = value['sols'][0]
                        error_message = str(value['errors'][0]) if value['results'][0][0] != True else 'No error.'
                        analysis, improved_code = analyze_and_improve_code(nl, code_snippet, error_message)
                        json_obj = {
                            "raw_index": key,
                            "nl": nl,
                            "code_snippet": code_snippet,
                            "error_message": error_message,
                            "analysis": analysis,
                            "improved_code": improved_code
                        }
                        jsonl_file.write(json.dumps(json_obj) + '\n')

# 运行处理函数并保存为jsonl文件
output_file = 'updated_data_0_100.jsonl'
process_and_save_as_jsonl(directory, output_file)

print(f"Data saved to {output_file}")
