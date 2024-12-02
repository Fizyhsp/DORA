from alfworld_runs.LLM import LLMAPI

from typing import List, Dict, Any

with open("./alfworld_runs/reflexion_few_shot_examples.txt", 'r') as f:
    FEW_SHOT_EXAMPLES = f.read()

def _get_scenario(s: str) -> str:
    """Parses the relevant scenario from the experience log."""
    return s.split("Here is the task:")[-1].strip()

def _generate_reflection_query(meta_prompt, log_str: str, memory: List[str]) -> str:
    """Allows the Agent to reflect upon a past experience."""
    scenario: str = _get_scenario(log_str)
    query: str = meta_prompt + f"""Here are two examples:

{FEW_SHOT_EXAMPLES}

{scenario}"""

    if len(memory) > 0:
        query += '\n\nPlans from past attempts:\n'
        for i, m in enumerate(memory):
            query += f'Trial #{i}: {m}\n'

    query += '\n\nNew plan:'
    return query

def update_memory(meta_prompt, trial_log_path: str, env_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Updates the given env_config with the appropriate reflections."""
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()
        
    env_logs: List[str] = full_log.split('#####\n\n#####')
    assert len(env_logs) == len(env_configs), print(f'bad: {len(env_logs)}, {len(env_configs)}')
    for i, env in enumerate(env_configs):
        # if unsolved, get reflection and update env config
        if not env['is_success'] and not env['skip']:
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']
            reflection_query: str = _generate_reflection_query(meta_prompt, env_logs[i], memory)
            reflection: str = LLMAPI(reflection_query) # type: ignore
            env_configs[i]['memory'] += [reflection]
                
    return env_configs
