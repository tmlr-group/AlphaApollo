import json
import re
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import DictConfig

from agent_system.environments.informal_math_evolving.base_text_env import (
    BaseTextEnv, BaseTextEnvStepOutput, ConversationType)
from agent_system.environments.informal_math_evolving.utils.qwen_math import \
    compute_score
from tools import \
    InformalMathToolGroup


class InformalMathEvolvingEnv(BaseTextEnv):
    """
    Environment for Informal Math tasks.
    """

    def __init__(self, env_config: DictConfig):
        super().__init__()
        
        # Build tool_config dict for tool group initialization
        tool_config = {
            "enable_python_code": getattr(env_config, "enable_python_code", True),
            "enable_local_rag": getattr(env_config, "enable_local_rag", False),
            "python_code_timeout": getattr(env_config, "python_code_timeout", 30),
            "rag_cfg": getattr(env_config, "rag", None),
        }
        
        # Initialize the tools
        self.tool_group = InformalMathToolGroup(
            log_requests=getattr(env_config, "log_requests", False),
            vllm_cfg=getattr(env_config, "vllm", None),
            verifier_cfg=getattr(env_config, "verifier", None),
            tool_config=tool_config,
        )
        self.init_tool_groups([self.tool_group])

    def reset(self, extras: Optional[Dict[str, Any]] = None) -> None:
        # NOTE: using the information in "extra_info" of the data field to initialize the environment
        extras = extras or {}
        for key in ["question", "ground_truth"]:
            assert key in extras, f"{key} is required in extras field"

        self.question = extras["question"]
        # self.gt_traj = extras.get("gt_traj", None)
        self.ground_truth = extras["ground_truth"]
        self.max_steps = extras.get("max_steps", 3)
        self.data_source = extras.get("data_source", "unknown")

        self.policy_solution = extras.get("policy_solution", None) # for verifier only

        # Set ground truth in tool group for Python verification
        self.tool_group.set_ground_truth(self.ground_truth)

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []
        self.done = False
        self.done_reason: Optional[str] = None
        self.turns = 0
        self._score_list = []

    def _get_reward(self, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["text_actions"] for item in self.chat_history])
            # print(f"\n=====Chat History:=====\n{chat_history_str}", flush=True)
            # solution_str = extract_answer_segment(chat_history_str) # match <answer>...</answer>
            solution_str = chat_history_str
            return compute_score(solution_str=solution_str, ground_truth=self.ground_truth)
        else:
            # No reward for intermediate steps for Informal Math tasks
            return 0
    
    
    def _is_done(self, action: str) -> bool:
        """
        Termination rules:
        - Hit max_steps
        - Empty action (invalid)
        - Contains <answer>...</answer> (policy)
        - Contains <report>...</report> (verifier)
        """
        self.done_reason = None
        if self.turns >= self.max_steps:
            self.done_reason = "max_steps"
            return True

        if not action or not str(action).strip():
            self.done_reason = "empty_action"
            return True

        if "<answer>" in action and "</answer>" in action:  # for policy agent only
            self.done_reason = "answer"
            return True
        if "<report>" in action and "</report>" in action:  # for verifier only
            self.done_reason = "report"
            return True

        return False

    def _execute_tool(self, tool_group_name: str, tool_name: str, tool_inputs: Any, return_score: bool = False) -> Any:
        tool_output = super()._execute_tool(tool_group_name, tool_name, tool_inputs)
        text_result = tool_output.get("text_result", "")
        score = tool_output.get("score", None)
        if return_score:
            return "\n<tool_response>" + text_result + "</tool_response>\n", score
        else:
            return "\n<tool_response>" + text_result + "</tool_response>\n"

    # Support multiple tool calling: python_code
    def _parse_action(self, action: str) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Parse action to extract tool calls.
        Returns a list of tuples: (tool_name, tool_input)
        """
        tool_calls = []
        
        # Check for python_code
        if "<python_code>" in action and "</python_code>" in action:
            match = re.search(r"<python_code>(.*?)</python_code>", action, re.DOTALL)
            if match:
                tool_calls.append(("python_code", match.group(1).strip()))
        
        # Check for informalmath_verify
        if "<informalmath_verify>" in action and "</informalmath_verify>" in action:
            match = re.search(r"<informalmath_verify>(.*?)</informalmath_verify>", action, re.DOTALL)
            if match:
                tool_calls.append(("informalmath_verify", match.group(1).strip()))

        # Check for local_rag
        if "<local_rag>" in action and "</local_rag>" in action:
            match = re.search(r"<local_rag>(.*?)</local_rag>", action, re.DOTALL)
            if match:
                tool_calls.append(("local_rag", match.group(1).strip()))

        # Return None if no tool calls found (for backward compatibility)
        if not tool_calls:
            return [(None, None)]
        
        return tool_calls

    def step(self, action: StopIteration, text_actions: List[str]) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.chat_history.append({"role": "assistant", 
                                  "content": action, 
                                  "text_actions": text_actions
                                  })

        # parse action to tool calls
        try:
            tool_calls = self._parse_action(action)
        except Exception as e:
            raise Exception(f"Error parsing action: {e}, action: {action}")
        
        self.done = self._is_done(action)

        reward = self._get_reward(self.done)

        if self.done:
            tool_calling = any(t[0] is not None for t in tool_calls)
            metadata = {
                "data_source": self.data_source,
                "tool_calling": tool_calling,
                "done_reason": self.done_reason,
            }
            if self.done_reason == "empty_action":
                metadata["is_action_valid"] = 0
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=self.done,
                metadata=metadata,
                postprocessed_action=action
            )


        observations = []
        tool_infos = []
        
        for tool_call in tool_calls:
            observation = None
            tool_info = None
            if tool_call[0] is not None:
                tool_name, tool_input = tool_call
                
                if tool_name == "python_code":
                    observation = self._execute_tool(
                        "InformalMathToolGroup",
                        "python_code",
                        {"code": tool_input}
                    )
                    tool_info = {
                        "tool_calling": True,
                        "tool_group": "InformalMathToolGroup",
                        "tool_name": "python_code",
                        "tool_input": tool_input,
                        "data_source": self.data_source,
                    }
                elif tool_name == "informalmath_verify":
                    observation = self._execute_tool(
                        "InformalMathToolGroup",
                        "informalmath_verify",
                        {
                            "question": self.question,
                            "policy_solution": self.policy_solution,
                        },
                    )
                    tool_info = {
                        "tool_calling": True,
                        "tool_group": "InformalMathToolGroup",
                        "tool_name": "informalmath_verify",
                        "tool_input": tool_input,
                        "data_source": self.data_source,
                    }
                elif tool_name == "local_rag":
                    # Parse JSON input for local_rag
                    try:
                        rag_params = json.loads(tool_input)
                        repo_name = rag_params.get("repo_name", "")
                        query = rag_params.get("query", "")
                    except (json.JSONDecodeError, TypeError):
                        repo_name = ""
                        query = tool_input
                    # DEBUG: local_rag execution
                    print(f"\n[DEBUG env.py] Executing local_rag: repo={repo_name}, query={query[:50]}...")
                    observation = self._execute_tool(
                        "InformalMathToolGroup",
                        "local_rag",
                        {"repo_name": repo_name, "query": query},
                    )
                    # DEBUG: local_rag result
                    print(f"[DEBUG env.py] local_rag result length: {len(observation) if observation else 0} chars")
                    print(f"[DEBUG env.py] local_rag result: {observation}")
                    tool_info = {
                        "tool_calling": True,
                        "tool_group": "InformalMathToolGroup",
                        "tool_name": "local_rag",
                        "tool_input": tool_input,
                        "data_source": self.data_source,
                    }

            # Wrap the observation properly as a message
            if observation:
                new_obs = {"role": "user", "content": observation, "text_actions": text_actions}
                self.chat_history.append(new_obs)
            else:
                new_obs = None
            if tool_info is None:
                tool_info = {
                    "tool_calling": False,
                    "tool_group": "InformalMathToolGroup",
                    "tool_name": None,
                    "tool_input": None,
                    "data_source": self.data_source,
                    }
            observations.append(new_obs)
            tool_infos.append(tool_info)


        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=self.done,
            metadata=tool_infos,
            postprocessed_action=action,
        )
