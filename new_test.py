import json
import os
import pickle as pkl
from tqdm import tqdm

# 假设 run_test 是一个已经定义好的函数，用于运行单个测试用例
# 文件路径
PROBLEM_SOLUTIONS_FILE = '/data/coding/RLTF/data/Python_Seeds_with_Problem_Descriptions_and_Solutions.jsonl'
INPUT_OUTPUT_FILE = '/data/coding/RLTF/data/input_output/input_output_3.json'
OUTPUT_FILE = '/data/coding/RLTF/data/test_example/output_results_3.pkl'
class TimeoutException(Exception):
    pass

def load_problem_and_solution(file_path, index):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    return None

def load_test_cases(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def get_function_name(solution_code):
    import re
    match = re.search(r'def (\w+)\(', solution_code)
    if match:
        return match.group(1)
    return None

def run_test(solution_code, function_name, input_data):
    import signal
    from pyext import RuntimeModule
    import tempfile as tfile
    import subprocess
    
    timeout = 4
    def timeout_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGALRM, timeout_handler)

    sol = solution_code
    sol += f'\ninput_data = """{input_data}"""'
    sol = sol.replace('print(', 'pass # print(')
    sol += f'\nprint({function_name}(input_data))'

    try:
        signal.alarm(timeout)
        with tfile.NamedTemporaryFile(mode="w+", suffix='.py', delete=True, encoding='utf-8') as tf:
            tf.write(sol)
            tf.flush()
            file_path = tf.name

            render_cmd = 'python ' + file_path
            p = subprocess.Popen(render_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, )
            out, err = p.communicate()
            returncode = p.returncode

            p.wait()
        signal.alarm(0)
        if returncode == 0:
            out = out.decode().strip()
            return out
        else:
            raise RuntimeError(err.decode())
    except TimeoutException:
        raise TimeoutException("Timed out!")
    except Exception as e:
        raise e
    finally:
        signal.alarm(0)

def run_tests(solution, test_cases):
    function_name = get_function_name(solution)
    if function_name is None:
        raise ValueError("No function name found in the solution code.")

    results = []
    inputs = test_cases['inputs']
    outputs = test_cases['outputs']
    for input_data, expected_output in tqdm(zip(inputs, outputs), desc="Running tests", total=len(inputs)):
        try:
            result = run_test(solution, function_name, input_data)
            passed = result == expected_output
            results.append({'input': input_data, 'expected': expected_output, 'result': result, 'passed': passed})
        except Exception as e:
            results.append({'input': input_data, 'expected': expected_output, 'result': str(e), 'passed': False})
    return results

def save_results(results, file_path):
    with open(file_path, 'wb') as f:
        pkl.dump(results, f)

def main():
    # 加载第三个问题和解决方案
    problem_index = 2  # 第三个问题的索引为2（从0开始计数）
    data = load_problem_and_solution(PROBLEM_SOLUTIONS_FILE, problem_index)
    
    if data is None:
        print(f"Problem at index {problem_index} not found.")
        return
    
    solution_code = data['solution']

    # 加载测试用例
    test_cases = load_test_cases(INPUT_OUTPUT_FILE)

    # 运行测试
    results = run_tests(solution_code, test_cases)

    # 保存结果
    save_results(results, OUTPUT_FILE)

if __name__ == '__main__':
    main()
    print(f'Tests complete, results saved to {OUTPUT_FILE}')