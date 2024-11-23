import pickle

def read_and_print_pkl(file_path):
    """
    读取 .pkl 文件并打印其完整结构。
    
    :param file_path: str, .pkl 文件的路径
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        def recursive_print(data, level=0):
            """
            递归打印数据结构。
            
            :param data: 要打印的对象
            :param level: 当前递归的深度
            """
            indent = "    " * level
            if isinstance(data, dict):
                print(f"{indent}{{")
                for key, value in data.items():
                    print(f"{indent}  {repr(key)}: ", end="")
                    recursive_print(value, level + 1)
                print(f"{indent}}}")
            elif isinstance(data, list):
                print(f"{indent}[")
                for item in data:
                    recursive_print(item, level + 1)
                print(f"{indent}]")
            elif isinstance(data, tuple):
                print(f"{indent}(")
                for item in data:
                    recursive_print(item, level + 1)
                print(f"{indent})")
            else:
                print(f"{indent}{repr(data)}")
        
        print("文件内容如下：")
        recursive_print(data)
    
    except Exception as e:
        print(f"读取 .pkl 文件时出错: {e}")

# 使用方法
# 替换 'your_file.pkl' 为你的文件路径
read_and_print_pkl('/data/coding/CodeRL/outputs/deep_ai_feedback/2.pkl')
