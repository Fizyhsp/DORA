import numpy as np
import os
import json
from alfworld_runs.alfworld_trial import run_trial
from alfworld_runs.generate_reflections import update_memory

from typing import Any, List, Dict
class ExecAccuracyEvaluationResult:

    def __init__(self, prompts, scores):
        self.prompts = prompts
        self.scores = scores

    def _agg_scores(self, method):
        """For each prompt, compute a statistic of the scores (e.g., mean, median)"""
        return [np.mean(s) for s in self.scores]


    def sorted(self):
        scores = self._agg_scores('mean')
        # Sort prompts by score
        sorted_prompts = [p for _, p in sorted(zip(scores, self.prompts))]
        sorted_scores = sorted(scores)
        # Reverse both and convert to lists
        sorted_prompts = list(reversed(sorted_prompts))
        sorted_scores = list(reversed(sorted_scores))
        return sorted_prompts, sorted_scores

    def in_place(self, method='default'):
        if method == 'default':
            scores = self._agg_scores('mean')
        else:
            scores = self._agg_scores(method)
        return self.prompts, scores


def GetScore(meta_prompt, task, memory, success_fail, instruction_id, success_id, ite):
    scores = []
    if not os.path.exists('eval_score_log'):
        os.makedirs('eval_score_log')
    logging_dir = task + '_DORA_log'
    with open(task + "_reflection_example.json", "r", encoding="utf-8") as f:
        reflection_history = json.load(f)
    # initialize environment configs
    env_configs: List[Dict[str, Any]] = []
    if task == 'pick_and_place':
        env_id = [0, 16, 19, 20, 21, 22, 23, 26, 36, 37]
    elif task == 'pick_cool_then_place':
        env_id = [1, 3, 18, 30, 38, 48, 52, 58, 59, 65]
    elif task == 'pick_heat_then_place':
        env_id = [2, 4, 11, 24, 31, 32, 40, 45, 51, 57]
    elif task == 'pick_two_obj':
        env_id = [8, 10, 35, 42, 43, 47, 50, 55, 61, 63]
    elif task == 'pick_clean_then_place':
        env_id = [5, 6, 7, 9, 12, 14, 15, 17, 25, 27]
    elif task == 'look_at_obj':
        env_id = [13, 33, 41, 53, 54, 62, 64, 66, 82, 87]
    for i in range(len(env_id)):
        env_configs += [{
            'name': f'env_{i}',
            'memory': memory[i],
            'is_success': False,
            'skip': False
        }]
    print(memory[i])

    world_log_path: str = os.path.join(logging_dir, 'world.log')

    # print start status to user
    print(f"""
-----
Starting run with the following parameters:
Run name: {logging_dir}
Number of trials: {3}
Number of environments: {10}
Use memory: {True}

Sending all logs to `{'DORA_log'}`
-----
""")

    # run trials
    trial_idx = 0
    while trial_idx < 2:
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** Start Trial #{trial_idx+instruction_id[0]*2} *****\n\n')

        # set paths to log files
        trial_log_path: str = os.path.join(logging_dir, f'trial_{trial_idx+instruction_id[0]*2}.log')
        trial_env_configs_log_path: str = os.path.join(logging_dir, f'env_results_trial_{trial_idx+instruction_id[0]*2}.json')
        if os.path.exists(trial_log_path):
            open(trial_log_path, 'w').close()
        if os.path.exists(trial_env_configs_log_path):
            open(trial_env_configs_log_path, 'w').close()

        # run trial
        env_history, _, results, similarity, step = run_trial(env_id, trial_log_path, world_log_path, trial_idx+instruction_id[0]*2, env_configs, True, 'gpt-3.5-turbo')

        # update memory if needed
        env_configs: List[Dict[str, Any]] = update_memory(meta_prompt[0], trial_log_path, env_configs)
        for z, env_config in enumerate(env_configs):
            reflection_history.append(
                {"input": str(env_history._history[0]), "output": str(env_config['memory'][-1])})
            with open(task + "_reflection_example.json", "w", encoding="utf-8") as f:
                json.dump(reflection_history, f)
            print(z)
            memory[z] = env_config["memory"]
            if env_config['is_success'] == True:
                success_fail[z] = 1
                success_id[z].append(trial_idx+instruction_id[0]*2)
                data = {'BO_ITERATION': ite,'Environment':env_config['name'], 'Trial':trial_idx+instruction_id[0] * 2, 'prompt':meta_prompt[0], 'is_success': 'SUCCESS', 'Trajectory':str(env_history._history),'Reflection':str(env_config['memory'])}
                with open(task + "_result.json", "a", encoding="utf-8") as f:
                    json.dump(data, f)
                    f.write("\n")
            else:
                data = {'BO_ITERATION': ite,'Environment':env_config['name'], 'Trial':trial_idx+instruction_id[0] * 2, 'prompt':meta_prompt[0], 'is_success': 'FAIL', 'Trajectory':str(env_history._history),'Reflection':str(env_config['memory'])}
                with open(task + "_result.json", "a", encoding="utf-8") as f:
                    json.dump(data, f)
                    f.write("\n")

        # log env configs for trial
        with open(trial_env_configs_log_path, 'a') as wf:
            json.dump(env_configs, wf, indent=4)

        # log world for trial
        with open(world_log_path, 'a') as wf:
            wf.write(f'\n\n***** End Trial #{trial_idx+instruction_id[0]*2} *****\n\n')
        trial_idx += 1

    for envid in results:
        scores.append(results[envid])
        scores.append(similarity[envid])
        scores.append(step[envid])


    instruction_id[0] = instruction_id[0] + 1
    # Reshape the scores so that it is num_prompts x num_samples
    scores = np.array(scores).reshape(len(meta_prompt), 30)

    res = ExecAccuracyEvaluationResult(meta_prompt, scores)
    return res, scores