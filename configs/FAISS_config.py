import argparse

def get_new_args():
    parser = argparse.ArgumentParser(description="Run an enhanced CodeT5 model with retrieval.")
    
    # 继承原有参数，保持生成和保存逻辑不变
    parser.add_argument("-t","--test_path", default="data/APPS/test/", type=str, 
                       help='Path to test samples')
    parser.add_argument("--output_path", type=str, required=True,
                       help='Path to save generated programs, saved as {problem_id}.json')
    parser.add_argument("--model_path", type=str, required=True,
                       help='Path of trained model')
    parser.add_argument("--tokenizer_path", default="/data/coding/model/codeT5", type=str, 
                       help='Path to the tokenizer')
    parser.add_argument("--critic_scores", default=False, action='store_true',
                       help='if model is a critic model, enable this to output critic scores')
    parser.add_argument("--binary_prediction", default=False, action='store_true',
                       help='if model is a critic model, enable this for binary classification')
    parser.add_argument("--num_seqs", default=5, type=int,
                       help='Number of total generated programs per test sample')
    parser.add_argument('--num_seqs_per_iter', default=5, type=int,
                       help='Number of possible minibatch to generate programs per iteration')
    parser.add_argument("--max_len", default=512, type=int,
                       help='Maximum length of output sequence')
    parser.add_argument('--source_len', default=600, type=int,
                       help='Maximum length of input sequence')
    parser.add_argument('--gt_solutions', default=False, action='store_true',
                       help='Only when critic is used, enable this to estimate returns/rewards for ground-truth programs')
    parser.add_argument("--temperature", default=0.6, type=float,
                       help='temperature for sampling tokens')
    parser.add_argument("-s","--start", default=0, type=int,
                       help='start index of test samples')
    parser.add_argument("-e","--end", default=10000, type=int,
                       help='end index of test samples')

    # 检索相关的路径参数 - 这些是用来读取历史数据的
    parser.add_argument("--code_output", default="/data/coding/FALCON/outputs/deepseek_test_result", type=str,
                       help='Path to historical code test results, contains *.pkl files')
    parser.add_argument("--feedback_path", default="/data/coding/FALCON/outputs/AI_Feedback", type=str,
                       help='Path to historical AI feedback data, contains *.pkl files')
    parser.add_argument("--task_path", default="/data/coding/FALCON/outputs/deep_codes", type=str,
                       help='Path to historical task descriptions, contains *.json files')
    
    # FAISS相关参数
    parser.add_argument("--embedding_dim", default=768, type=int,
                       help='Dimension of embedding vectors')
    parser.add_argument("--index_path", default="/data/coding/FALCON/outputs/faiss_index", type=str,
                       help='Path for FAISS index file')
    parser.add_argument("--use_faiss", default=True, action='store_true',
                       help='Whether to use FAISS retrieval')
    parser.add_argument("--num_retrieve", default=1, type=int,
                       help='Number of similar tasks to retrieve')
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_new_args()