import os
import json

def extract_and_replace_code(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    answer_key = "ANSWER(only use python):"
    modified_data = {}

    for key, value in data.items():
        code = value["code"]
        modified_code = []
        for item in code:
            if answer_key in item:
                answer_start = item.find(answer_key) + len(answer_key)
                extracted_answer = item[answer_start:].strip()
                modified_code.append(extracted_answer)
            else:
                modified_code.append(item)
        value["code"] = modified_code
        modified_data[key] = value
    
    return modified_data

def process_directory(input_directory):
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.json'):
            file_path = os.path.join(input_directory, file_name)
            modified_data = extract_and_replace_code(file_path)
            
            with open(file_path, 'w') as f:
                json.dump(modified_data, f, indent=4)

if __name__ == "__main__":
    input_directory = "/data/coding/RLTF/outputs/deepseek_outputs/deepseek-coder-6___7b-base"  # 修改为你的JSON文件目录

    process_directory(input_directory)
    print("Processing complete.")
