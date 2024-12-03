import copy
import torch
from evaluation import GetScore
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


class OpenSourceLLMAPI:
    def __init__(self, init_prompt=None, vec_dim=None, tokens_num=None, model_hf_dir=None):
        kwargs = {
            'torch_dtype': torch.float16,
            'use_cache': True
        }

        self.model = AutoModelForCausalLM.from_pretrained(model_hf_dir, low_cpu_mem_usage=True, **kwargs).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(model_hf_dir, model_max_length=1024, padding_side="left", use_fast=False,)
        self.init_prompt_str = init_prompt
        self.embedding = self.model.get_input_embeddings().weight.clone()
        self.init_prompt = self.embedding[self.tokenizer(init_prompt, return_tensors="pt").input_ids.cuda()]
        self.tokens_num = tokens_num
        self.hidden_size = self.init_prompt.shape[-1]
        self.system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        self.role = ['USER:', 'ASSISTANT:']
        self.best_eval_score = 0.0
        self.best_prompt = None
        self.best_prompt = None
        self.instruction_id = [0]  # 记录指令id
        self.memory = [[], [], [], [], [], [], [], [], [], []]  # 保存每个任务实例的反思建议
        self.success_fail = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 记录10个任务实例中成功或失败，0表示失败，1表示成功
        self.success_id = [[10000], [10000], [10000], [10000], [10000], [10000], [10000], [10000], [10000], [10000]]  # 记录每个任务实例成功的具体轮次
        #随机投影（均匀分布）
        self.linear = torch.nn.Linear(vec_dim, self.tokens_num * self.hidden_size, bias=False)
        for p in self.linear.parameters():
            torch.nn.init.uniform_(p, -1, 1)


    def eval(self, prompt_embedding = None, llm_input = None, iterations = 0, args = None):
        self.init_token = self.init_prompt_str[0] + llm_input[0]
        if prompt_embedding is None:
            prompt_embedding = self.best_prompt
        tmp_prompt = copy.deepcopy(prompt_embedding)
        prompt_embedding = prompt_embedding.type(torch.float32)
        prompt_embedding = self.linear(prompt_embedding)  # 随机投影Az
        prompt_embedding = prompt_embedding.reshape(1, self.tokens_num, -1)

        input_text = f"{self.system_prompt} USER:{self.init_token} ASSISTANT:"
        input_embedding = self.embedding[self.tokenizer(input_text, return_tensors="pt").input_ids.cuda()]
        prompt_embedding = prompt_embedding.to(device=input_embedding.device, dtype=input_embedding.dtype)
        input_embedding = torch.cat((prompt_embedding, input_embedding), 1)
        outputs = self.model.generate(inputs_embeds=input_embedding, max_new_tokens=128)
        prompt = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        with open(args.task + '_prompt.json', 'a') as f:
            json.dump(prompt, f)
        #对指令进行评估得到评分
        eval_score, prompt_score = GetScore(prompt=prompt, task=args.task, memory=self.memory, success_fail=self.success_fail, instruction_id=self.instruction_id, success_id=self.success_id, iterations=iterations)
        eval_score = eval_score.sorted()[1][0]

        if eval_score >= self.best_eval_score:
            self.best_eval_score = eval_score
            self.best_prompt = copy.deepcopy(tmp_prompt)
            self.best_prompt = prompt

        prompt_info = f"prompt: {prompt[0]}, eval_score: {eval_score}, prompt_score: {prompt_score}"
        with open(args.task + '_prompt_info.json', 'a') as f:
            json.dump(prompt_info, f)

        return eval_score, prompt_score
