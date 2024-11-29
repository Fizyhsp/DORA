import math
import re
import json
import numpy as np
from langchain import PromptTemplate
from agents import ReflectionAgent
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from utils import make_instance
from envs import MiniwobEnv
from check_action import check_action

class ExecAccuracyEvaluationResult:
    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def _agg_scores(self):
        """计算soft prompt得分"""
        return [np.mean(s) for s in self.scores]


    def sorted(self):
        scores = self._agg_scores()
        # 根据评分排序
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores


def compute_cosine(text1, text2):
    """计算两条反思指令余弦相似度"""
    words1 = text1.split()
    words2 = text2.split()
    words1_dict = {}
    words2_dict = {}
    for word in words1:
        word = re.sub('[^a-zA-Z]', '', word)
        word = word.lower()
        if word != '' and word in words1_dict:
            num = words1_dict[word]
            words1_dict[word] = num + 1
        elif word != '':
            words1_dict[word] = 1
        else:
            continue
    for word in words2:
        word = re.sub('[^a-zA-Z]', '', word)
        word = word.lower()
        if word != '' and word in words2_dict:
            num = words2_dict[word]
            words2_dict[word] = num + 1
        elif word != '':
            words2_dict[word] = 1
        else:
            continue
    dic1 = sorted(words1_dict.items(), key=lambda x: x[1], reverse=True)
    dic2 = sorted(words2_dict.items(), key=lambda x: x[1], reverse=True)

    words_key = []
    list(map(lambda x: words_key.append(x[0]), dic1))
    list(map(lambda x: words_key.append(x[0]), filter(lambda x: x[0] not in words_key, dic2)))


    vect1 = []
    vect2 = []
    for word in words_key:
        if word in words1_dict:
            vect1.append(words1_dict[word])
        else:
            vect1.append(0)
        if word in words2_dict:
            vect2.append(words2_dict[word])
        else:
            vect2.append(0)

    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(vect1)):
        sum += vect1[i] * vect2[i]
        sq1 += pow(vect1[i], 2)
        sq2 += pow(vect2[i], 2)
    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
    except ZeroDivisionError:
        result = 0.0

    return result


def GetScore(prompt, task, memory, success_fail, instruction_id, success_id, iterations):
    """对生成的反思指令进行评估"""
    llm = ChatOpenAI(openai_api_base="https://api.chatanywhere.com.cn/v1", temperature=0.5,
                     openai_api_key="sk-4LbX8s4Tlb3UfNhyWEbyZZSFF6qqklBVQXs3sHZpdhMQpjeP",
                     model_name="gpt-3.5-turbo-1106")
    with open("prompts/reflection.txt", 'r') as f:
        reflection_template = f.read()
    reflection_prompt = PromptTemplate(template=reflection_template, input_variables=["meta_prompt", "task",
                                                                                      "trajectory"],
                                       template_format="jinja2")
    reflection_chain = LLMChain(prompt=reflection_prompt, llm=llm)
    env: MiniwobEnv = make_instance(
        MiniwobEnv,
        EPISODE_MAX_TIME=100000000,
        name="miniwob/"+task,
        env_type="static",
    )

    agent = ReflectionAgent(
        name="llm_agent",
        env=env,
        api_base="https://api.chatanywhere.com.cn/v1",
        api_key="sk-4LbX8s4Tlb3UfNhyWEbyZZSFF6qqklBVQXs3sHZpdhMQpjeP",
        model_name="gpt-3.5-turbo-1106",
        template="prompts/reflection_agent.txt",
        task='',
        proxy="127.0.0.1:7890",
    )
    obs_buffer = {}
    traj_buffer = {}
    real_step = 0
    index = 0
    step = 0
    with open("./reflection_suggestions_pool/"+task + "_reflection_example.json", "r", encoding="utf-8") as f:
        reflection_history = json.load(f)

    num_episodes = 1
    reflection_step = 1
    scores = []
    trajectory = ''
    for i in range(num_episodes):
        agent.reset(memory=memory[i])
        seed = i
        for j in range(reflection_step):
            real_step = j
            agent.trajectory = []
            traj_temp = []
            obs, info = env.reset(seed=seed)
            obs_prev = {'dom_elements': {}}
            agent.task = obs['utterance']
            while len(obs['dom_elements']) != len(obs_prev['dom_elements']):
                obs_prev = obs
                obs_buffer[index] = obs['dom_elements']
                agent.index = index
                action_plan_list, think_list = agent.plan(obs['dom_elements'])
                think_num = 0
                if(action_plan_list==[]):
                    reward = 0
                for action_plan in action_plan_list:
                    if check_action(action_plan):
                        try:
                            obs, reward, terminated, truncated, info = env.step(
                                action_plan[0],
                                **action_plan[1],
                            )
                        except Exception as e:
                            continue
                        agent.trajectory.append(
                            {'think': think_list[think_num], 'action': {'name': action_plan[0].name, 'input_args_schema': action_plan[1]}})
                        agent.trajectory.pop(0) if len(
                            agent.trajectory,
                        ) > agent.trajectory_max_length else None
                        traj_temp.append(
                            {'think': think_list[think_num], 'action': {'name': action_plan[0].name, 'input_args_schema': action_plan[1]}})
                        traj_temp.pop(0) if len(
                            traj_temp,
                        ) > agent.trajectory_max_length else None
                        traj_buffer[index] = traj_temp
                        step = step + 1
                        if len(obs['dom_elements']) != len(obs_prev['dom_elements']):
                            index = index + 1
                            traj_temp = []
                            break
                        if reward != 0:
                            break
                    else:
                        reward = 0
                    think_num = think_num + 1
                if reward != 0:
                    break
                if step > 10:
                    break
            if reward > 0:
                step = 0
                obs_buffer = {}
                traj_buffer = {}
                index = 0
                success_fail[i] = 1
                success_id[i].append(j+instruction_id[0]*3)
                #将成功结果写入文件
                data = {'BO_ITERATION': iterations,'Environment':i, 'Trial':j+instruction_id[0] * 3, 'prompt':prompt[0], 'is_success': 'SUCCESS', 'Trajectory':str(agent.trajectory),'Reflection':str(agent.memory)}
                with open(task + "_result.json", "a", encoding="utf-8") as f:
                    json.dump(data, f)
                    f.write("\n")
                break
            # 将失败结果写入文件
            data = {'BO_ITERATION': iterations,'Environment':i, 'Trial':j+instruction_id[0] * 3, 'prompt':prompt[0], 'is_success': 'FAIL', 'Trajectory':str(agent.trajectory),'Reflection':str(agent.memory)}
            with open(task + "_result.json", "a", encoding="utf-8") as f:
                json.dump(data, f)
                f.write("\n")
            for key in obs_buffer:
                if key not in list(traj_buffer):
                    traj_buffer[key] = []
            #生成智能体轨迹
            for key in obs_buffer:
                trajectory = trajectory + 'The index=' + str(key) + ' screen:' + str(obs_buffer[key]) + ', actions ' \
                                                                                                        'for this ' \
                                                                                                        'screen: ' + \
                             str(traj_buffer[key])+' .'
            obs_buffer = {}
            traj_buffer = {}
            step = 0
            index = 0
            reflection = reflection_chain.predict(meta_prompt=prompt[0], task=obs['utterance'],
                                                  trajectory=trajectory)
            reflection_history.append({"input": 'task:' + str(agent.task) + '; trajectory:' + str(agent.trajectory), "output": reflection})
            memory[i].append(reflection)
            #写入反思样例池
            with open("./reflection_suggestions_pool/"+task + "_reflection_example.json", "w", encoding="utf-8") as f:
                json.dump(reflection_history, f)

            trajectory = ''
            agent.update_memory(memory[i])
        #环境反馈reward
        scores.append((reward+1)/2)

        #步骤数
        if real_step == 0:
            scores.append(1.0)
        else:
            scores.append(1/real_step)
        #反思指令相似度
        if len(agent.memory) >= 2:
            scores.append(float(1.0 - compute_cosine(str(agent.memory[-1]), str(agent.memory[-2]))))
        else:
            scores.append(float(1.0))

    env.close()
    instruction_id[0] = instruction_id[0] + 1

    scores = np.array(scores).reshape(len(prompt), 3*num_episodes)

    res = ExecAccuracyEvaluationResult(prompt, scores)
    return res, scores