import math

import numpy as np
from openai import OpenAI


def ppl_verify(verifier_prompt, verifier_output, use_verifier_output=False):
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

    if use_verifier_output:
        prefix = verifier_prompt + verifier_output
    else:
        prefix = verifier_prompt

    perplexities = []
    for bbox in ["\\bbox{{0}}", "\\bbox{{1}}"]:
        full_prompt = prefix + bbox
        res_echo = client.completions.create(model="qwen3_8b", prompt=full_prompt, max_tokens=1, logprobs=1, echo=True)
        cumulative_text = ""
        tokens = res_echo.choices[0].logprobs.tokens
        tokens_logprobs = []
        for i in range(len(tokens)):
            token = tokens[i]
            cumulative_text += token
            if len(cumulative_text) > len(verifier_prompt):
                logprobs = res_echo.choices[0].logprobs.token_logprobs[i]
                tokens_logprobs.append(logprobs)
        # calculate the perplexity
        avg_logprobs = np.mean(tokens_logprobs)
        perplexity = math.exp(-avg_logprobs)
        perplexities.append(perplexity)
    
    return perplexities
