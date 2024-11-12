import os
import pickle
from collections import Counter


def extract_errors_and_tracebacks(directory):
    error_counter = Counter()
    traceback_counter = Counter()

    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            file_path = os.path.join(directory, filename)

            with open(file_path, 'rb') as f:
                data = pickle.load(f)


            if 1 in data:
                entry = data[1]
                errors = entry.get('errors', [])
                tracebacks = entry.get('results', [])  # 假设tracebacks存储在results中

                # 展开嵌套列表并提取错误信息的字符串表示
                errors = [str(error) for sublist in errors for error in sublist if error is not None]
                tracebacks = [str(traceback) for sublist in tracebacks for traceback in sublist if
                              traceback is not None]

                error_counter.update(errors)
                traceback_counter.update(tracebacks)

    return error_counter, traceback_counter


# 指定文件所在目录
directory = '/data/coding/RLTF/outputs/deepseek_outputs/new_result/'

# 提取错误信息和traceback并统计
error_counter, traceback_counter = extract_errors_and_tracebacks(directory)

# 输出统计结果
print("Error Statistics:")
for error, count in error_counter.items():
    print(f"{error}: {count}")

print("\nTraceback Statistics:")
for traceback, count in traceback_counter.items():
    print(f"{traceback}: {count}")
