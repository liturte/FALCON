import json
import pickle
import os
import time
from openai import OpenAI
import re
import argparse

def create_evaluation_prompt(code, original_prompt):
   """创建评估prompt"""
   prompt = f"""Please evaluate the following code solution based on three assessment criteria.
Provide EXACTLY three numbers between -1 and 2 separated by commas in this format: X,Y,Z
DO NOT include any other text or explanation in your response.

Coding Style Assessment (-1 to 2):
-1. Poor Adherence: The code significantly deviates from standard practices, showing poor readability, maintainability, and efficiency.
0. Basic Adherence: The code makes some effort to follow language conventions but lacks consistency in readability, maintainability, or efficiency.
1. Good Adherence: The code generally follows standards, demonstrating adequate readability, maintainability, and efficiency, though with room for improvement.
2. Excellent Adherence: The code exemplifies best practices, with high readability, maintainability, and efficiency, fully adhering to idiomatic conventions.

Complexity Assessment (-1 to 2):
-1. Overly Complex: The code is unnecessarily complicated, with a high level of complexity that makes it hard to understand or maintain.
0. Acceptable Complexity: The code has a reasonable level of complexity, but there may be opportunities to simplify further.
1. Moderately Simple: The code is simple and well-organized, with minimal complexity and clear logic.
2. Optimal Simplicity: The code exemplifies the best practices in minimizing complexity while ensuring functionality.

Instruction Following Assessment (-1 to 2):
-1. Non-Compliant: The assistant frequently deviates from instructions without necessity or user consent.
0. Acceptable: The assistant shows some adherence to instructions but deviates without strong justification.
1. Compliant with Justified Deviations: The assistant generally follows instructions, with deviations occurring but justified by necessity or user request.
2. Fully Compliant: The assistant follows instructions closely, with minimal deviations, all of which are well justified.

Original Problem/Instructions:
{original_prompt}

Code to Evaluate:
{code}

Your response should be EXACTLY in this format: X,Y,Z
Where:
X = Coding Style score (-1 to 2)
Y = Complexity score (-1 to 2)
Z = Instruction Following score (-1 to 2)
Example valid response: 1,0,2"""

   return prompt

def call_deepseek_api(prompt, api_key, max_retries=3):
   """调用DeepSeek API，带重试机制"""
   client = OpenAI(
       api_key="sk-27461bf6eee242cfb6cfc779b16036f0",
       base_url="https://api.deepseek.com"
   )
   
   for attempt in range(max_retries):
       try:
           response = client.chat.completions.create(
               model="deepseek-chat",
               messages=[
                   {"role": "system", "content": "You are a code evaluation assistant. Always respond with exactly three numbers between -1 and 2, separated by commas."},
                   {"role": "user", "content": prompt}
               ],
               stream=False,
               temperature=0.1  # 降低随机性
           )
           return response.choices[0].message.content
       except Exception as e:
           print(f"API call attempt {attempt + 1} failed: {e}")
           if attempt == max_retries - 1:
               print("All API call attempts failed")
               return None
           time.sleep(1)  # 失败后等待1秒再重试

def parse_scores(response):
   """解析API返回的评分"""
   if response is None:
       print("Error: Received None response from API")
       return [-1, -1, -1]
   
   try:
       # 清理response，只保留数字和逗号
       cleaned_response = response.strip()
       
       # 调试输出
       print(f"Raw API response: {response}")
       
       # 尝试不同的分隔模式
       if ',' in cleaned_response:
           # 按逗号分割
           scores = [int(score.strip()) for score in cleaned_response.split(',')]
       else:
           # 尝试提取数字
           scores = [int(num) for num in re.findall(r'-?\d+', cleaned_response)]
       
       # 验证分数
       if len(scores) != 3:
           print(f"Warning: Expected 3 scores, got {len(scores)}: {scores}")
           return [-1, -1, -1]
           
       # 验证分数范围
       for score in scores:
           if score not in [-1, 0, 1, 2]:
               print(f"Warning: Score {score} out of valid range [-1, 2]")
               return [-1, -1, -1]
       
       return scores
   except Exception as e:
       print(f"Score parsing error: {e}")
       print(f"Response was: {response}")
       return [-1, -1, -1]

def process_single_file(file_path, output_dir, api_key):
   """处理单个JSON文件并保存评估结果"""
   try:
       # 读取JSON文件
       with open(file_path, 'r') as f:
           data = json.load(f)
       
       # 获取文件编号
       file_num = os.path.basename(file_path).split('.')[0]
       print(f"Processing file {file_num}")
       
       # 修改结果存储格式，移除results
       results = {
           'code': [],        # 存储代码
           'Coding Style': [], # 存储代码风格评分
           'Complexity': [],   # 存储复杂度评分
           'Instruction Following': [], # 存储指令遵循度评分
       }
       
       # 从data中获取代码
       first_key = list(data.keys())[0]
       codes = data[first_key].get('code', [])
       prompt = data[first_key].get('prompt', '')
       
       if not codes:
           print(f"Warning: No code found in {file_path}")
           return None
       
       print(f"Found {len(codes)} code samples")
       
       for i, code in enumerate(codes):
           print(f"Processing code {i+1}/{len(codes)}")
           
           if code is None:
               print(f"Warning: Code {i+1} is None, skipping...")
               continue
               
           results['code'].append(code)
           
           evaluation_prompt = create_evaluation_prompt(code, prompt)
           response = call_deepseek_api(evaluation_prompt, api_key)
           scores = parse_scores(response)
           
           results['Coding Style'].append(scores[0])
           results['Complexity'].append(scores[1])
           results['Instruction Following'].append(scores[2])
           
           print(f"Code {i+1} scores: Style={scores[0]}, Complexity={scores[1]}, Instruction={scores[2]}")
       
       # 保存为pkl文件
       output_path = os.path.join(output_dir, f"{file_num}.pkl")
       with open(output_path, 'wb') as f:
           pickle.dump(results, f)
       
       print(f"Results saved to {output_path}")
       return results
   
   except Exception as e:
       print(f"Error processing file {file_path}: {e}")
       import traceback
       traceback.print_exc()
       return None

def process_directory(input_dir, output_dir, api_key):
   """处理整个目录的文件"""
   # 确保输出目录存在
   os.makedirs(output_dir, exist_ok=True)
   
   # 获取所有json文件并排序
   json_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.json')])
   total_files = len(json_files)
   
   print(f"Found {total_files} files to process")
   
   # 处理所有文件
   for i, filename in enumerate(json_files, 1):
       file_path = os.path.join(input_dir, filename)
       print(f"\nProcessing file {i}/{total_files}: {filename}")
       
       results = process_single_file(file_path, output_dir, api_key)
       if results is None:
           print(f"Failed to process {filename}")
       
       # 每处理10个文件打印一次进度
       if i % 10 == 0:
           print(f"Progress: {i}/{total_files} files processed")

def main():
   parser = argparse.ArgumentParser(description='Process code files with DeepSeek API')
   parser.add_argument('--input_dir', type=str, required=True,
                       default="/data/coding/CodeRL/outputs/deep_codes",
                       help='Input directory containing JSON files')
   parser.add_argument('--output_dir', type=str, required=True,
                       default="/data/coding/CodeRL/outputs/AI_Feedback",
                       help='Output directory for PKL files')
   
   args = parser.parse_args()
   
   print(f"Processing files from {args.input_dir}")
   print(f"Saving results to {args.output_dir}")
   
   process_directory(args.input_dir, args.output_dir, args.api_key)
   
   print("\nProcessing complete!")

if __name__ == "__main__":
   main()