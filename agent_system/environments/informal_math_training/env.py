from agent_system.environments.informal_math_training.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from typing import Any
from agent_system.environments.informal_math_training.utils.qwen_math import compute_score, extract_answer_segment
from tools.manager import InformalMathToolGroup
import re
from typing import Dict, Optional, List, Tuple
from omegaconf import DictConfig
import json

# Tool pattern definitions for extensibility
# To add a new tool, append ("tool_name", r"<tool_name>(.*?)</tool_name>") to this list
TOOL_PATTERNS = [
    ("python_code", r"<python_code>(.*?)</python_code>"),
    ("informalmath_verify", r"<informalmath_verify>(.*?)</informalmath_verify>"),
    ("local_rag", r"<local_rag>(.*?)</local_rag>"),
]

class InformalMathTrainingEnv(BaseTextEnv):
    """
    Environment for Informal Math tasks.
    """

    def __init__(self, env_config: DictConfig):
        super().__init__()
        
        # Build tool_config dict for tool group initialization
        tool_config = {
            "enable_python_code": getattr(env_config, "enable_python_code", True),
            "enable_local_rag": getattr(env_config, "enable_local_rag", True),
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

        # Set ground truth in tool group for Python verification
        self.tool_group.set_ground_truth(self.ground_truth)

        # Chat history
        # role (user, assistant), content (tool observation or LLM response)
        self.chat_history: ConversationType = []
        self.done = False
        self.turns = 0
        self._score_list = []

    def _get_reward(self, done: bool) -> float:
        if done:
            # Concat all chat history into a single string and compute reward
            chat_history_str = "".join([item["text_actions"] for item in self.chat_history])
            solution_str = chat_history_str

            return compute_score(solution_str=solution_str, ground_truth=self.ground_truth)
        else:
            # No reward for intermediate steps for Informal Math tasks
            return 0
    
    
    def _is_done(self, tool_calls: List[Tuple[Optional[str], Optional[str]]]) -> bool:
        # 1. exceed max steps
        if self.turns >= self.max_steps:
            return True
        
        # 2. no tool calls
        if not tool_calls or all(tool_call == (None, None) for tool_call in tool_calls):
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

    # Support multiple tool calling: python_code and informalmath_verify
    def _parse_action(self, action: str) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Parse action to extract tool calls using unified TOOL_PATTERNS.
        Returns a list of tuples: (tool_name, tool_input)
        
        To add a new tool, simply add its pattern to the TOOL_PATTERNS list at module level.
        """
        tool_calls = []
        
        for tool_name, pattern in TOOL_PATTERNS:
            if f"<{tool_name}>" in action and f"</{tool_name}>" in action:
                match = re.search(pattern, action, re.DOTALL)
                if match:
                    tool_calls.append((tool_name, match.group(1).strip()))
        
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
        # NOTE: Use text_actions (original model output) instead of action (projection-truncated)
        # to correctly extract ALL tool calls including local_rag when multiple tools are enabled
        raw_action = text_actions if isinstance(text_actions, str) else action
        try:
            tool_calls = self._parse_action(raw_action)
        except Exception as e:
            raise Exception(f"Error parsing action: {e}, action: {raw_action}")
        
        self.done = self._is_done(tool_calls)

        reward = self._get_reward(self.done)

        if self.done:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=reward,
                done=self.done,
                metadata={"data_source": self.data_source, "tool_calling": False},
                postprocessed_action=action
            )


        observations = []
        tool_infos = []
        
        for tool_call in tool_calls:
            if tool_call[0] is not None:
                observation = None
                tool_info = None
                tool_name, tool_input = tool_call
                
                if tool_name == "python_code":
                    # Get raw tool output as dict to check score
                    tool_output = super()._execute_tool(
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
                    # Check score and modify text_result if needed
                    if self.tool_group.enable_local_rag and tool_output.get("score", None) == 0:
                        inner = json.loads(tool_output.get("text_result", ""))
                        inner["result"] = (
                            str(inner.get("result", ""))
                            + "\n\nPlease use the local_rag tool to query relevant information and resolve the code issue."
                        )
                        tool_output["text_result"] = json.dumps(inner)
                    # Convert to string format for observation
                    text_result = tool_output.get("text_result", "")
                    observation = "\n<tool_response>" + text_result + "</tool_response>\n"
                elif tool_name == "informalmath_verify":
                    observation = self._execute_tool(
                        "InformalMathToolGroup",
                        "informalmath_verify",
                        {
                            "question": self.question,
                            "solution": tool_input,
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
                    try:
                        tool_input_dict = json.loads(tool_input)
                        observation = self._execute_tool(
                            "InformalMathToolGroup",
                            "local_rag",
                            tool_input_dict
                        )
                        tool_info = {
                            "tool_calling": True,
                            "tool_group": "InformalMathToolGroup",
                            "tool_name": "local_rag",
                            "tool_input": tool_input,
                            "data_source": self.data_source,
                        }
                    except json.JSONDecodeError:
                        observation = "\n<tool_response>Error: Invalid JSON input for local_rag</tool_response>\n"
                        tool_info = {
                            "tool_calling": True,
                            "tool_group": "InformalMathToolGroup",
                            "tool_name": "local_rag",
                            "tool_input": tool_input,
                            "data_source": self.data_source,
                            "error": "Invalid JSON"
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
