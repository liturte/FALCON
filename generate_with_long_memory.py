import json
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import glob
from transformers import RobertaTokenizer, T5ForConditionalGeneration, T5EncoderModel
import faiss

class FAISSRetriever:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_path)
        self.encoder = self.model.encoder  
        self.encoder.to(self.device)
        self.tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
        
        # 初始化或加载FAISS索引
        if os.path.exists(args.index_path):
            self.index = faiss.read_index(args.index_path)
            with open(f"{args.index_path}_mapping.pkl", 'rb') as f:
                self.id_mapping = pickle.load(f)
        else:
            self.build_index()
            
    def text_to_vector(self, text):
        """将文本转换为向量表示"""
        inputs = self.tokenizer(text, truncation=True, max_length=self.args.source_len, 
                              return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            vector = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return vector

    def build_index(self):
        """构建FAISS索引"""
        print("Building FAISS index...")
        
        self.index = faiss.IndexFlatL2(self.args.embedding_dim)
        self.id_mapping = {}
        
        tasks_path = self.args.task_path
        vectors = []
        ids = []
        
        for filename in os.listdir(tasks_path):
            if not filename.endswith('.json'):
                continue
                
            task_id = filename.split('.')[0]
            with open(os.path.join(tasks_path, filename), 'r') as f:
                task_data = json.load(f)
            
            if not task_data or task_id not in task_data:
                continue
                
            prompt = task_data[task_id].get('prompt', '')
            if not prompt:
                continue
                
            vector = self.text_to_vector(prompt)
            vectors.append(vector)
            ids.append(task_id)
            
        vectors = np.vstack(vectors)
        self.index.add(vectors)
        
        faiss.write_index(self.index, self.args.index_path)
        self.id_mapping = {i: task_id for i, task_id in enumerate(ids)}
        with open(f"{self.args.index_path}_mapping.pkl", 'wb') as f:
            pickle.dump(self.id_mapping, f)
            
        print(f"Built index with {len(ids)} vectors")

    def retrieve(self, query, k=1):
        """检索最相似的任务"""
        query_vector = self.text_to_vector(query)
        D, I = self.index.search(query_vector, k)
        results = []
        for i, dist in zip(I[0], D[0]):
            task_id = self.id_mapping[i]
            results.append((task_id, float(dist)))
        return results

def load_pkl(file_path):
    """加载pkl文件"""
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def load_json(file_path):
    """加载json文件"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def retrieve_similar_task(question, args, retriever=None):
    """检索最相似的任务及其相关信息"""
    if not args.use_faiss:
        return None
        
    if retriever is None:
        retriever = FAISSRetriever(args)
        
    similar_tasks = retriever.retrieve(question, args.num_retrieve)
    if not similar_tasks:
        return None
        
    task_id = similar_tasks[0][0]
    
    # 1. 读取测试结果
    test_data = load_pkl(os.path.join(args.code_output, f"{task_id}.pkl"))
    if not test_data:
        return None
        
    best_solution = test_data['sols'][0] if test_data.get('sols') else ""
    main_errors = []
    if test_data.get('errors'):
        for err in test_data['errors']:
            if err and len(err) > 2:
                main_errors.append(err[2])

    # 2. 读取评分反馈
    feedback_data = load_pkl(os.path.join(args.feedback_path, f"{task_id}.pkl"))
    if not feedback_data:
        return None
        
    style_score = feedback_data['Coding Style'][0] if feedback_data.get('Coding Style') else 0
    complexity_score = feedback_data['Complexity'][0] if feedback_data.get('Complexity') else 0
    instruction_score = feedback_data['Instruction Following'][0] if feedback_data.get('Instruction Following') else 0

    # 3. 读取任务描述
    task_data = load_json(os.path.join(args.task_path, f"{task_id}.json"))
    if not task_data or task_id not in task_data:
        return None
        
    task_prompt = task_data[task_id].get('prompt', '')

    return {
        'task': task_prompt,
        'code': best_solution,
        'feedback': {
            'style_score': style_score,
            'efficiency_score': complexity_score,
            'instruction_score': instruction_score
        },
        'errors': "; ".join(main_errors) if main_errors else "No common errors found"
    }
def generate_prompt(args, test_case_path, prompt_path, solutions_path, tokenizer, starter_path=None):
    """生成包含历史信息的prompt"""
    # 1. 获取原始问题描述
    _input = "\nQUESTION:\n"
    try:
        with open(prompt_path, "r") as f:
            data = f.readlines()
            question_text = "".join(data)
        _input += question_text
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        return None

    # 2. 添加starter code(如果有)
    if starter_path and os.path.exists(starter_path):
        try:
            with open(starter_path, "r") as f:
                data = f.readlines()
                data = "".join(data)
                data = "\n" + data 
            _input += data
        except Exception as e:
            print(f"Error reading starter code: {e}")
    
    # 3. 添加输入格式说明
    if os.path.exists(test_case_path):
        try:
            with open(test_case_path, "r") as f:
                data = json.load(f)
            if not data.get("fn_name"):
                _input += "\nUse Standard Input format"
            else:
                _input += "\nUse Call-Based format"
        except Exception as e:
            print(f"Error reading test case: {e}")
            _input += "\nUse Standard Input format"
    elif starter_path is not None and os.path.exists(starter_path):
        _input += "\nUse Call-Based format"
    else:
        _input += "\nUse Standard Input format"
        
    # 4. 检索相似任务 - 修改这里，传入question_text而不是data
    similar_task = retrieve_similar_task(question_text, args)
    print(f"Retrieved similar task: {True if similar_task else False}")
    if similar_task:
        # 5. 添加历史任务上下文
        _input += "\n\nContext of relevant code:"
        _input += f"\n- Historical Task: {similar_task['task']}"
        _input += f"\n- Code: {similar_task['code']}"
        _input += f"\n- Style Score: {similar_task['feedback']['style_score']}"
        _input += f"\n- Efficiency Score: {similar_task['feedback']['efficiency_score']}"
        _input += f"\n- Instruction Following Score: {similar_task['feedback']['instruction_score']}"
        
        # 6. 添加要求和需要避免的错误
        _input += "\n\nRequirements:"
        _input += "\n1. Ensure the generated code adheres to best practices for Python, including proper naming conventions, consistent formatting, and coding standards."
        _input += "\n2. Optimize the code for performance, avoiding unnecessary recursion or nested loops."
        _input += "\n3. Use built-in or efficient library functions whenever applicable to improve both readability and performance."
        
        _input += "\n\nAvoid the following errors:"
        _input += f"\n- {similar_task['errors']}"
    
    # 7. 保持原有的ANSWER标记
    _input += "\nANSWER:\n"
    
    return _input

def main(args):
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载tokenizer
    try:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 加载模型
    print(f"Loading model from {args.model_path}...")
    try:
        if args.critic_scores:
            model = T5ForConditionalGeneration.from_pretrained(args.model_path, tuning_mode='critic')
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.model_path)
        model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # 初始化FAISS检索器
    if args.use_faiss:
        retriever = FAISSRetriever(args)
    else:
        retriever = None

    # 处理问题列表
    problems = sorted(glob.glob(os.path.join(args.test_path, '*')))
    if args.start >= len(problems) or args.start < 0:
        print(f"Start index {args.start} > number of problems {len(problems)}")
        return
        
    if args.end is None or args.end > len(problems):
        end = len(problems)
    else:
        end = args.end
    problems = problems[args.start:end]

    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Saving results to {args.output_path}")

    # 处理每个问题
    for index, problem in tqdm(enumerate(problems), ncols=0, total=len(problems)):
        prob_path = os.path.join(problem)
        problem_id = int(os.path.basename(problem))
        print(f"Processing problem {problem_id}")

        # 检查是否已经处理过
        if args.critic_scores and \
            os.path.exists(os.path.join(args.output_path, f"{problem_id}_gt{args.gt_solutions}.pkl")):
            continue
        elif os.path.exists(os.path.join(args.output_path, f"{problem_id}.json")):
            continue

        # 准备文件路径
        test_case_path = os.path.join(prob_path, "input_output.json")
        prompt_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")
        if not os.path.exists(starter_path):
            starter_path = None

        solutions_path = os.path.join(prob_path, 
            "gen_solutions.json" if args.critic_scores and not args.gt_solutions else "solutions.json")

        # 生成prompt
        input_text = generate_prompt(args, test_case_path, prompt_path, solutions_path, 
                                   tokenizer, starter_path)
        if not input_text:
            continue
        print(input_text)
        with torch.no_grad():
            try:
                # 编码输入
                input_ids = torch.LongTensor(tokenizer.encode(input_text, 
                                                            verbose=False, 
                                                            max_length=args.source_len)).unsqueeze(0).to(device)

                
                num_loops = int(args.num_seqs / args.num_seqs_per_iter)
                output_programs = []
                
                for i in range(num_loops):
                    output_ids = model.generate(
                        input_ids,
                        do_sample=True,
                        temperature=args.temperature,
                        max_length=args.max_len,
                        num_return_sequences=args.num_seqs_per_iter,
                        top_p=0.95
                    )

                    for output_id in output_ids:
                        output_programs.append(tokenizer.decode(output_id, skip_special_tokens=True))

                # 保存生成的代码
                saved_codes = {problem_id: {'code': output_programs, 'prompt': input_text}}
                codes_loc = os.path.join(args.output_path, f"{problem_id}.json")
                
                with open(codes_loc, "w") as f:
                    json.dump(saved_codes, f)
                    
            except Exception as e:
                print(f"Error processing problem {problem_id}: {e}")
                continue

if __name__ == "__main__":
    from configs.FAISS_config import get_new_args
    args = get_new_args()
    main(args)