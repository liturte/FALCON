import pickle

# 指定pickle文件的路径
pkl_file_path = '/data/coding/RLTF/data/test_example/output_results_3.pkl'

# 读取pickle文件内容
with open(pkl_file_path, 'rb') as file:
    data = pickle.load(file)

# 打印内容
print(data)
