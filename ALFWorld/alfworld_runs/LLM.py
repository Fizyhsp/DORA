import openai
from typing import Optional, List


def LLMAPI(prompt: str, model = 'gpt-3.5-turbo', temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stop=stop_strs,
        temperature=temperature,
    )
    return response.choices[0]["message"]["content"]
