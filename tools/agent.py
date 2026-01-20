import os
import sys
import numpy as np
import time
import logging
import yaml
import fire
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
from collections import defaultdict
# Ensure repository root is on PYTHONPATH so local packages can be imported
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# from agent_system.environments.env_manager import *
from openai import OpenAI
import json
import pandas as pd

class Agent:
    def __init__(self, vllm_config: dict):
        api_key = vllm_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "EMPTY")
        base_url = vllm_config.get("base_url", "http://localhost:8000/v1")
        self.model_name = vllm_config.get("model_name", "qwen3-4b")
        self.temperature = vllm_config.get("temperature", 0.4)
        self.max_tokens = vllm_config.get("max_tokens", 16384)
        self.system_prompt = vllm_config.get("system_prompt", None)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def get_action_from_gpt(self, obs):
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        messages.append({
            "role": "user",
            "content": obs
        })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=1,
            stop=None
        )
        action = response.choices[0].message.content.strip()
        return action