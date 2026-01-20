import logging
import os
import sys
from pathlib import Path

# Ensure repository root is on PYTHONPATH so local packages can be imported
project_root = Path(__file__).parent.parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

from openai import OpenAI

from agent_system.environments import *


class Agent:
    def __init__(self, vllm_config: dict):
        api_key = vllm_config.get("api_key") or os.environ.get("OPENAI_API_KEY", "EMPTY")
        base_url = vllm_config.get("base_url", "http://localhost:8000/v1")
        self.model_name = vllm_config.get("model_name", "qwen3-4b")
        self.temperature = vllm_config.get("temperature", 0.4)
        self.max_tokens = vllm_config.get("max_tokens", 16384)

        # Read system prompt configuration
        # Default to None for backward compatibility
        self.system_prompt = vllm_config.get("system_prompt", None)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=5,
            timeout=300.0,
        )

    def get_action_from_gpt(self, obs):
        messages = []
        # Add system message only if system_prompt is configured
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
        
        # print(response.model_dump_json(indent=2))
        # ALign the content will vllm.
        reasoning_text = ""
        reasoning_payload = None
        if hasattr(response.choices[0].message, 'reasoning_content'):
            reasoning_payload = response.choices[0].message.reasoning_content
        elif hasattr(response.choices[0].message, 'reasoning'):
            reasoning_payload = response.choices[0].message.reasoning

        if isinstance(reasoning_payload, str) and reasoning_payload.strip():
            reasoning_text = "<think>\n" + reasoning_payload.strip() + "\n</think>\n"

        action = response.choices[0].message.content.strip()
        return reasoning_text + action