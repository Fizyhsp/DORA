"""Adapted from https://github.com/ysymyth/ReAct/blob/master/alfworld.ipynb"""

import os
import sys
import json
import yaml
import copy
import importlib
import alfworld
import alfworld.agents.environment
from alfworld_runs.LLM import LLMAPI
from alfworld_runs.env_history import EnvironmentHistory
import math
import re
from typing import List, Dict, Any, Tuple


FOLDER = './alfworld_runs/prompts'
PROMPT_FILE = 'alfworld_3prompts.json'
results = {}
similarity = {}
step = {}
with open(os.path.join(FOLDER, PROMPT_FILE), 'r') as f:
    d = json.load(f)

def llm(prompt: str, model, stop: List[str] = ["\n"]):
    try:
        cur_try = 0
        while cur_try < 6:
            text = LLMAPI(prompt=prompt, model=model, temperature=cur_try * 0.2, stop_strs=stop)
            # dumb way to do this
            if len(text.strip()) >= 5:
                return text
            cur_try += 1
        return ""
    except Exception as e:
        print(prompt)
        print(e)
        import sys
        sys.exit(1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

def alfworld_run(env, base_prompt, memory: List[str], to_print=True, ob='', model=None) -> Tuple[EnvironmentHistory, bool]:
    if len(memory) > 3:
        env_history = EnvironmentHistory(base_prompt, ob, memory[-3:], [])
    else:
        env_history = EnvironmentHistory(base_prompt, ob, memory, [])
    env_history.reset()
    if to_print:
        print(ob)
        sys.stdout.flush()
    cur_step = 0
    while cur_step < 30:
        action = llm(str(env_history) + ">", stop=['\n'], model=model).strip()
        env_history.add("action", action)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        env_history.add("observation", observation)
        if to_print:
            print(f'> {action}\n{observation}')
            sys.stdout.flush()
        if done:
            return env_history, True, cur_step
        elif env_history.check_is_exhausted():
            return env_history, False, cur_step
        cur_step += 1
        print('cur_step:',cur_step)
    return env_history, False, cur_step

PREFIXES = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}


def compute_cosine(text1, text2):
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


def run_trial(
        env_id,
        trial_log_path: str,
        world_log_path: str,
        trial_idx: int,
        env_configs: List[Dict[str, Any]],
        model: str
    ) -> List[Dict[str, Any]]:
    flag = 0
    flag_break = 0
    importlib.reload(alfworld)
    importlib.reload(alfworld.agents.environment)
    env_id_copy = copy.deepcopy(env_id)


    with open('./alfworld_runs/base_config.yaml') as reader:
        config = yaml.safe_load(reader)
    split = "eval_out_of_distribution"
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    print(env)

    num_successes: int = 0
    num_additional_successes: int = 0
    num_envs: int = len(env_configs)
    for i, env_config in enumerate(env_configs):
        results[i] = 0
        similarity[i] = 0
        step[i] = 0

    for z, env_config in enumerate(env_configs):
        for num in env_id_copy:
            for nu in range(flag, 90):
                if nu != num:
                    ob, info = env.reset()
                else:
                    env_id_copy.remove(num)
                    flag = nu + 1
                    flag_break = 1
                    break
            if flag_break == 1:
                flag_break = 0
                break
        ob, info = env.reset()
        print(z)
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])


        print(f"using {name}")


        for i, (k, v) in enumerate(PREFIXES.items()):
            if name.startswith(k):
                base_prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0']
                final_env_history, is_success, cur_step = alfworld_run(env, base_prompt, env_config["memory"], to_print=True, ob=ob, model=model)

                # update env config
                if is_success:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: SUCCESS'
                    results[z] = 1
                    env_configs[z]['is_success'] = True
                    num_successes += 1
                    num_additional_successes += 1
                    if cur_step == 0:
                        step[z] = 1.0
                    else:
                        step[z] = 1 / cur_step
                else:
                    status_str: str = f'Environment #{z} Trial #{trial_idx}: FAIL'
                    results[z] = 0
                    env_configs[z]['is_success'] = False
                    if cur_step == 0:
                        step[z] = 1.0
                    else:
                        step[z] = 1 / cur_step

                # log to world log
                with open(world_log_path, 'a') as f:
                    f.write(status_str + '\n')

                # log env results to trial log
                with open(trial_log_path, 'a') as wf:
                    wf.write(f'\n#####\n\nEnvironment #{z}:\n{str(final_env_history)}\n\nSTATUS: {"OK" if is_success else "FAIL"}\n\n#####\n')

        if len(env_config["memory"]) >= 2:
            similarity[z] = float(1.0 - compute_cosine(str(env_config["memory"][-1]), str(env_config["memory"][-2])))
        else:
            similarity[z] = float(1.0)

    # close environment object
    env.close()

    # log trial results to trial and world logs
    log_str: str = f"""
-----
SUCCESS: {num_successes}
ADDITIONAL SUCCESS: {num_additional_successes}
FAIL: {num_envs - num_successes}
TOTAL: {num_envs}
ACCURACY: {round(num_successes / num_envs, 2)}
-----"""
    with open(trial_log_path, 'a') as wf:
        wf.write(log_str)
    with open(world_log_path, 'a') as wf:
        wf.write(log_str + '\n')

    return final_env_history, env_configs, results, similarity, step
