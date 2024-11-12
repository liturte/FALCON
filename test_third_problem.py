import json
import numpy as np
import os
import glob
from tqdm import tqdm
import pickle as pkl
from io import StringIO
from typing import get_type_hints
from typing import List, Tuple

def run_test(prob_path:str=None, problem_list:List[str]=None, prob_index:int=None,
        test:str=None, debug:bool=False, example_tests:bool=False, debug_compare=False, go_on=False):
    """
    if test is not None it'll try to run the code.
    otherwise it'll just return an input and output pair.
    """
    
    if prob_path is None and problem_list is None:
        print("please provide either prob_path or problem_list")
        exit()

    if debug:
        print(f"start = {datetime.now().time()}")
    if prob_path is not None:
        root = prob_path
    elif problem_list is not None:
        root = problem_list[prob_index]

    if os.path.exists(os.path.join(root, "input_output.json")):
        with open(os.path.join(root, "input_output.json")) as f:
            in_outs = json.load(f)
            if debug:
                print(f"test cases json = {in_outs['inputs']} {in_outs['outputs']}")
            
            if in_outs.get("fn_name") is None:
                which_type = CODE_TYPE.standard_input  # Standard input
                method_name = None
            else:
                which_type = CODE_TYPE.call_based  # Call-based
                method_name = in_outs["fn_name"]
    elif not example_tests:
        return [], [], [], None 
    elif example_tests: 
        which_type = CODE_TYPE.standard_input  # assuming this method type 
        method_name = None
    
    if example_tests:
        if os.path.exists(os.path.join(root, "example_input_output.json")):
            with open(os.path.join(root, "example_input_output.json")) as f:
                in_outs = json.load(f)
                if in_outs is None: 
                    return [], [], [], None
        else:
            return [], [], [], None
    
    if debug:
        print(f"loaded json = {datetime.now().time()}")
    #else:
    #    continue
    if test is None:
        print('test is None')
        return [], [], [], None 
    elif test is not None:
        # reliability_guard()

        results = []
        errors = []
        outputs = []
        sol = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n"


        if debug:
            print(f"loading test code = {datetime.now().time()}")
        
        if which_type == CODE_TYPE.call_based:
            sol += test
            if debug: # or True:
                print(f"sol = {sol}")
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                if "class Solution" not in test:
                    tmp = tmp_sol
                else:
                    tmp = tmp_sol.Solution()
                signal.alarm(0)
            
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 0 compilation error = {e}")
                if isinstance(e, SyntaxError):
                    results.append(-2)
                    errors.append(e)
                    outputs.append(None)

                    return results, errors, outputs, sol
            signal.alarm(0)

        elif which_type == CODE_TYPE.standard_input:
            # sol
            tmp_test = test.split("\n")

            new_test = []
            for x in tmp_test:
                if (not x.startswith("from ")) and (not x.startswith("import ")):
                    new_test.append("    " + x + "\n")
                else:
                    new_test.append(x + "\n")
            tmp_test = new_test
            
            new_test = ""
            started = False
            for i in tmp_test:
                if (i.startswith("    ") or i.startswith("\t")) and not started:
                    new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                    new_test += "def code():\n"
                    new_test += i
                    started = True
                elif started and ((i.startswith("from ")) or (i.startswith("import "))): 
                    new_test += "    " + i
                else:
                    new_test += i
            tmp_test = new_test

            sol += tmp_test
            if debug:
                print(f"sol = {sol}")
                # print(f"{o}") 
            method_name = "code"
            signal.alarm(timeout)
            try:
                tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
                tmp = tmp_sol
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                if debug:
                    print(f"type 1 compilation error = {e}")
                if isinstance(e, SyntaxError):
                    results.append(-2)
                    errors.append(e)
                    outputs.append(None)
                    return results, errors, outputs, sol
            signal.alarm(0)
        if debug:
            print(f"get method = {datetime.now().time()}")
        if debug_compare:
            print(sol)
        for index, inputs in enumerate(in_outs["inputs"]):
            sol_str = copy.deepcopy(sol)
            try:
                if isinstance(inputs[0], dict):
                    inputs = [{int(k): v for k, v in inputs[0].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index], dict):
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index].items()}]
            except:
                True
            try:
                if isinstance(in_outs["outputs"][index][0], dict):
                    in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index][0].items()}]
            except:
                True


            returncode = None
            if which_type == CODE_TYPE.call_based:
                sol_str += f'\ninputs = {inputs}'.replace('inf', 'math.inf')
                sol_str = sol_str.replace('print(', 'pass # print(')
                if 'Solution' in sol_str:
                    sol_str += f'\nsol = Solution()'
                    sol_str += f'\nprint(sol.{method_name}(*inputs))'
                else:
                    sol_str += f'\nprint({method_name}(*inputs))'

                try:
                    signal.alarm(4)
                    with tfile.NamedTemporaryFile(mode="w+", suffix='.py', delete=True, encoding='utf-8') as tf:
                        tf.write(sol_str)
                        tf.flush()
                        file_path = tf.name

                        render_cmd = 'python ' + file_path
                        p = subprocess.Popen(render_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, )
                        out, err = p.communicate()
                        returncode = p.returncode

                        p.wait()
                    signal.alarm(0)
                    if returncode == 1:
                        results.append(-1)
                        # errors.append(None)
                        errors.append(err.decode())
                        outputs.append(None)

                        return results, errors, outputs, sol
                    elif returncode == 0:
                        out = out.decode()
                        out = out.split('\n')
                        if out[-1] == '':
                            out = out[:-1]
                        out = '\n'.join(out)

                        # print('11111')
                        # print(out)
                        # if isinstance(in_outs['outputs'][index], bool) or isinstance(in_outs['outputs'][index], int) \
                        #     or isinstance(in_outs['outputs'][index], list):
                        #     in_outs['outputs'][index] = str(in_outs['outputs'][index])

                        res = stripped_string_compare(out, str(in_outs['outputs'][index]))
                        res = res or (stripped_string_compare(f'[{out}]', str(in_outs['outputs'][index])))
                        res = res or (stripped_string_compare(f'({out})', str(in_outs['outputs'][index])))
                        res = res or (stripped_string_compare(str([out]), str(in_outs['outputs'][index])))
                        res = res or (stripped_string_compare(str((out)), str(in_outs['outputs'][index])))

                        if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                            try:
                                res = res or stripped_string_compare(out, str(in_outs['outputs'][index][0]))
                                res = res or (stripped_string_compare(f'[{out}]', str(in_outs['outputs'][index][0])))
                                res = res or (stripped_string_compare(f'({out})', str(in_outs['outputs'][index][0])))
                                res = res or (stripped_string_compare(str([out]), str(in_outs['outputs'][index][0])))
                                res = res or (stripped_string_compare(str((out)), str(in_outs['outputs'][index][0])))
                                res = res or stripped_string_compare(out, str(tuple(in_outs['outputs'][index][0])))
                                res = res or (stripped_string_compare(f'[{out}]', str(tuple(in_outs['outputs'][index][0]))))
                                res = res or (stripped_string_compare(f'({out})', str(tuple(in_outs['outputs'][index][0]))))
                                res = res or (stripped_string_compare(str([out]), str(tuple(in_outs['outputs'][index][0]))))
                                res = res or (stripped_string_compare(str((out)), str(tuple(in_outs['outputs'][index][0]))))
                            except:
                                pass

                        try:
                            res = res or (float(out) == in_outs['outputs'][index])
                        except:
                            pass

                        try:
                            res = res or (float(out) == in_outs['outputs'][index][0])
                        except:
                            pass

                        # if '[' in out or ']' in out or ',' in out or '(' in out or ')' in out:
                        #     try:
                        #         tmp_out = out.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                        #         tmp_out = tmp_out.split(',')
                        #
                        #         res = res or (tmp_out == in_outs['outputs'][index])
                        #         res = res or (tmp_out == in_outs['outputs'][index][0])
                        #
                        #         tmp_out = [float(i) for i in tmp_out]
                        #         res = res or (tmp_out == in_outs['outputs'][index])
                        #         res = res or (tmp_out == in_outs['outputs'][index][0])
                        #
                        #     except:
                        #         pass

                        try:
                            tmp_out = eval(out)
                            res = res or (tmp_out == in_outs['outputs'][index])
                            res = res or (tmp_out == in_outs['outputs'][index][0])
                            if isinstance(tmp_out[0], tuple):
                                res = res or ([list(x) for x in tmp_out] == in_outs["outputs"][index][0])

                        except:
                            pass

                        if debug_compare:
                            print("--- Call based ---")
                            print("inputs: ", inputs)
                            print("outputs: ", type(out), out)
                            print("gt_outputs: ", type(in_outs['outputs'][index]), in_outs['outputs'][index])
                            print("result: ", res)

                        results.append(res)
                        errors.append(None)
                        outputs.append(out)

                        if res == False and not go_on:
                            return results, errors, outputs, sol

                    else:
                        raise RuntimeError('error returncode')

                except Exception as e:
                    signal.alarm(0)
                    p.kill()
                    # print('Time out')
                    results.append(-1)
                    errors.append(e)
                    outputs.append(None)
                    return results, errors, outputs, sol
                signal.alarm(0)

            elif which_type == CODE_TYPE.standard_input:
                if isinstance(inputs, list):
                    inputs = "\n".join(inputs)
                if isinstance(in_outs['outputs'][index], list):
                        in_outs['outputs'][index] = "\n".join(in_outs['outputs'][index])

                # sol_str += '\nsys.setrecursionlimit(100000)'
                sol_str += f'\n{method_name}()'
                # print(sol_str)

                try:
                    signal.alarm(4)
                    with tfile.NamedTemporaryFile(mode="w+", suffix='.py', delete=True, encoding='utf-8') as tf:
                        tf.write(sol_str)
                        tf.flush()
                        file_path = tf.name

                        render_cmd = 'python ' + file_path
                        p = subprocess.Popen(render_cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
                        out, err = p.communicate(input=inputs.encode())
                        returncode = p.returncode

                        # print("err")
                        # print(err)
                        # print("output")
                        # print(out)
                        # print('returncode: ', returncode)

                        p.wait()
                    signal.alarm(0)
                    if returncode == 1:
                        results.append(-1)
                        # errors.append(None)
                        errors.append(err.decode())
                        outputs.append(None)

                        # signal.alarm(timeout)
                        # try:
                        #     call_method(method, inputs)
                        #     # reset the alarm
                        #     signal.alarm(0)
                        # except Exception as e:
                        #     signal.alarm(0)
                        #     errors.append(e)
                        # signal.alarm(0)

                        return results, errors, outputs, sol
                    elif returncode == 0:
                        out = out.decode()
                        output = out.split('\n')
                        if output[-1] == '':
                            output = output[:-1]

                        if debug_compare:
                            print("--- Standard Input ---")
                            print("inputs: ", inputs)
                            print("outputs: ", type(output), output)
                            print("gt_outputs: ", type(in_outs['outputs'][index]), in_outs['outputs'][index])
                            print("custom_compare_: ", custom_compare_(output, in_outs["outputs"][index]))

                        if custom_compare_(output, in_outs['outputs'][index]):
                            tmp_result = True
                            results.append(tmp_result)
                            errors.append(None)
                            outputs.append(output)
                            continue

                        # ground truth sequences are expressed as lists not tuples
                        if isinstance(output, tuple):
                            output = list(output)

                        tmp_result = False
                        try:
                            tmp_result = (output == [in_outs["outputs"][index]])
                            if isinstance(in_outs["outputs"][index], list):
                                tmp_result = tmp_result or (output == in_outs["outputs"][index])
                                if isinstance(output[0], str):
                                    tmp_result = tmp_result or (
                                                [e.strip() for e in output] == in_outs["outputs"][index])
                        except Exception as e:
                            if debug:
                                print(f"Failed check1 exception = {e}")
                            pass

                        if debug_compare:
                            print(f'check1  result: {tmp_result}')
                            print("outputs: ", type(output), output)

                        if tmp_result == True:
                            results.append(tmp_result)
                            errors.append(None)
                            outputs.append(output)
                            continue

                        # try one more time without \n
                        if isinstance(in_outs["outputs"][index], list):
                            for tmp_index, i in enumerate(in_outs["outputs"][index]):
                                in_outs["outputs"][index][tmp_index] = i.split("\n")
                                in_outs["outputs"][index][tmp_index] = [x.strip() for x in
                                                                        in_outs["outputs"][index][tmp_index] if x]
                        else:
                            in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                            in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                            in_outs["outputs"][index] = list(map(lambda x: x.strip(), in_outs["outputs"][index]))

                        try:
                            tmp_result = (output == [in_outs["outputs"][index]])
                            if isinstance(in_outs["outputs"][index], list):
                                tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        except Exception as e:
                            if debug:
                                print(f"Failed check2 exception = {e}")
                            pass

                        if debug_compare:
                            print(f'check2  result: {tmp_result}')

                        if tmp_result == True:
                            results.append(tmp_result)
                            errors.append(None)
                            outputs.append(output)
                            continue

                        # try by converting the output into a split up list too
                        if isinstance(output, list):
                            output = list(filter(len, output))

                        if debug:
                            nl = "\n"
                            if not isinstance(inputs, list):
                                print(
                                    f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs.replace(nl, ' new-line ')}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")
                            else:
                                print(
                                    f"output = {output}, test outputs = {in_outs['outputs'][index]}, inputs = {inputs}, {type(inputs)}, {output == [in_outs['outputs'][index]]}")

                        if tmp_result == True:
                            results.append(tmp_result)
                            errors.append(None)
                            outputs.append(output)
                            continue

                        try:
                            tmp_result = (output == [in_outs["outputs"][index]])
                            if isinstance(in_outs["outputs"][index], list):
                                tmp_result = tmp_result or (output == in_outs["outputs"][index])
                        except Exception as e:
                            if debug:
                                print(f"Failed check3 exception = {e}")
                            pass

                        if debug_compare:
                            print(f'check3  result: {tmp_result}')

                        try:
                            output_float = [float(e) for e in output]
                            gt_float = [float(e) for e in in_outs['outputs'][index]]
                            tmp_result = tmp_result or (
                                        (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float))
                        except Exception as e:
                            pass
                        try:
                            if isinstance(output[0], list):
                                output_float = [float(e) for e in output[0]]
                                gt_float = [float(e) for e in in_outs['outputs'][index][0]]
                                tmp_result = tmp_result or (
                                            (len(output_float) == len(gt_float)) and np.allclose(output_float,
                                                                                                 gt_float))
                        except Exception as e:
                            pass

                        if tmp_result == True:
                            results.append(tmp_result)
                            errors.append(None)
                            outputs.append(output)
                            continue

                        # try by converting the stuff into split up list
                        if isinstance(in_outs["outputs"][index], list):
                            for tmp_index, i in enumerate(in_outs["outputs"][index]):
                                in_outs["outputs"][index][tmp_index] = set(i.split())
                        else:
                            in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

                        try:
                            tmp_result = (output == in_outs["outputs"][index])
                        except Exception as e:
                            if debug:
                                print(f"Failed check4 exception = {e}")
                            continue

                        if debug_compare:
                            print(f'check4  result: {tmp_result}')

                        if tmp_result == True:
                            results.append(tmp_result)
                            errors.append(None)
                            outputs.append(output)
                            continue

                        # try by converting the output into a split up list too
                        if isinstance(output, list):
                            for tmp_index, i in enumerate(output):
                                output[tmp_index] = i.split()
                            output = list(filter(len, output))
                            for tmp_index, i in enumerate(output):
                                output[tmp_index] = set(i)
                        else:
                            output = output.split()
                            output = list(filter(len, output))
                            output = set(output)

                        try:
                            tmp_result = (set(frozenset(s) for s in output) == set(
                                frozenset(s) for s in in_outs["outputs"][index]))
                        except Exception as e:
                            if debug:
                                print(f"Failed check5 exception = {e}")

                        if debug_compare:
                            print(f'check5  result: {tmp_result}')

                        # if they are all numbers, round so that similar numbers are treated as identical
                        try:
                            tmp_result = tmp_result or (set(frozenset(round(float(t), 3) for t in s) for s in output) == \
                                                        set(frozenset(round(float(t), 3) for t in s) for s in
                                                            in_outs["outputs"][index]))
                        except Exception as e:
                            if debug:
                                print(f"Failed check6 exception = {e}")

                        if debug_compare:
                            print(f'check6  result: {tmp_result}')

                        if tmp_result == True and debug:
                            print("PASSED")

                        results.append(tmp_result)
                        errors.append(None)
                        outputs.append(output)

                        if tmp_result != True and not go_on:
                            ## TESTING TRICK: exit loop if not pass a test case
                            return results, errors, outputs, sol

                    else:
                        raise RuntimeError('error returncode')

                except Exception as e:
                    signal.alarm(0)
                    p.kill()
                    # print('Time out')
                    results.append(-1)
                    errors.append(e)
                    outputs.append(None)
                    return results, errors, outputs, sol
                signal.alarm(0)

            if returncode == -11:
                print('Segmentation Fault')
                results.append(False)
                errors.append('Segmentation Fault')
                outputs.append(None)
                return results, errors, outputs, sol
    return results, errors, outputs, sol


# 指定问题描述文件路径和输出文件路径
PROBLEM_SOLUTIONS_FILE = '/data/coding/RLTF/data/Python_Seeds_with_Problem_Descriptions_and_Solutions.jsonl'
TEST_OUTPUT_DIR = '/data/coding/RLTF/data/test_example'  # 输出文件的目录
INPUT_OUTPUT_FILE = '/data/coding/RLTF/data/input_output/input_output_3.json'  # 测试用例文件

def load_problem_and_solution(file_path, index):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    return None

def eval_and_save_problem(problem, solution, output_path, test_path):
    problem_id = 3
    print(f'Testing sample {problem_id}')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print(f'Saving to {output_path}/{problem_id}.pkl')

    all_results, all_errors, all_sols = [], [], []

    for o_idx, gen_code in tqdm(enumerate(solution), total=len(solution), ncols=0, leave=False):

        curr_results = []
        curr_errors = []
        curr_sol = None
        try:
            curr_results, curr_errors, _, curr_sol = run_test(prob_path=test_path, test=gen_code, debug=False,
                                          example_tests=False)

        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_results, list)
            all_results.append(curr_results)
            all_errors.append(curr_errors)
            all_sols.append(curr_sol)

        save_results = {problem_id: {'results': all_results, 'errors': all_errors, 'sols': all_sols}}
        with open(output_path + f'/{problem_id}.pkl', "wb") as file:
            pkl.dump(save_results, file)

    '''
    How to read results:
    [-2] = compile error, 
    [-1] = runtime error 
    [False] = failed test case 
    [True] = passed test case
    '''

    save_results = {problem_id: {'results': all_results, 'errors': all_errors, 'sols': all_sols}} 
    pkl.dump(save_results,  open(output_path + f'/{problem_id}.pkl', "wb"))

def main():
    # 加载第三个问题和解决方案
    problem_index = 2  # 第三个问题的索引为2（从0开始计数）
    data = load_problem_and_solution(PROBLEM_SOLUTIONS_FILE, problem_index)
    
    if data is None:
        print(f"Problem at index {problem_index} not found.")
        return
    
    problem_description = data['problem_description']
    solution = data['solution']
    print(solution)
    # 加载本地测试用例
    if not os.path.isfile(INPUT_OUTPUT_FILE):
        print(f"Test file {INPUT_OUTPUT_FILE} not found.")
        return

    with open(INPUT_OUTPUT_FILE, 'r') as infile:
        test_cases = json.load(infile)

    # 运行测试并保存结果
    eval_and_save_problem(problem_description, solution, TEST_OUTPUT_DIR, INPUT_OUTPUT_FILE)

if __name__ == '__main__':
    main()
    print(f'Tests complete, results saved in directory: {TEST_OUTPUT_DIR}')
