import argparse
import os
from BayesianOptimization import PromptOptimizer
import openai

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai.api_base = "https://api.chatanywhere.com.cn/v1"
openai.api_key = "sk-4LbX8s4Tlb3UfNhyWEbyZZSFF6qqklBVQXs3sHZpdhMQpjeP"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='', help='task')
    parser.add_argument('--tokens_num', type=int, default=5, help='tokens_num')
    parser.add_argument('--vec_dim', type=int, default=10, help='vec_dim')
    parser.add_argument('--model_hf_dir', type=str, default='lmsys/vicuna-7b-v1.3', help='model_hf_dir')
    args = parser.parse_args()
    meta_prompt, success_fail, success_id = PromptOptimizer(args)
    success_task = 0

    for i in success_fail:
        if i == 1:
            success_task = success_task + 1
    success_rate = f'Task: {args.task}; Success_rate: {success_task/10}'
    for i in range(10):
        if min(success_id[i])!=10000:
            with open(args.task + '_ENV_trail_Result.txt', 'a') as f:
                f.write(f'Environment #{i} Trial #{min(success_id[i])} : SUCCESS')
                f.write('\n')
        else:
            with open(args.task + '_ENV_trail_Result.txt', 'a') as f:
                f.write(f'Environment #{i} : Fail')
                f.write('\n')

    print(success_rate)
    with open('./DORA_result.txt', 'a') as f:
        f.write(success_rate)
        f.write('\n')
