# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import fire
import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts import *
from agent_system.memory import (EvolvingMemory,
                                 NDimensionalMemory, SearchMemory,
                                 SimpleMemory)

logger = logging.getLogger(__name__)

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos


class InformalMathEvolvingEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for InformalMathEnv.
    """
    def __init__(self, envs, projection_f, config):
        if config.env.informal_math_evolving.memory_type == "score":
            self.memory = EvolvingMemory(sort_key="score", descending=True)
        elif config.env.informal_math_evolving.memory_type == "ndimensional":
            self.memory = NDimensionalMemory(
                dimensions=["complexity", "performance"],
                performance_key="score", complexity_key="action"
            )
        elif config.env.informal_math_evolving.memory_type == "simple":
            self.memory = SimpleMemory()
        else:
            raise ValueError(f"Invalid memory type: {config.env.informal_math_evolving.memory_type}")
        super().__init__(envs, projection_f, config)

    @staticmethod
    def _sanitize_action_for_memory(action: str, reason: str | None) -> str:
        """Ensure we never persist empty actions or raw <report> blocks into memory."""
        if reason == "empty_action":
            return "[EMPTY_ACTION]"
        if reason == "report":
            return "[REPORT_REDACTED]"
        return action if action else "[UNSPECIFIED_ACTION]"

    def reset(self, kwargs, verifier: bool = False, use_previous_solutions: bool = False) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        # self.tasks = obs
        self.tasks = [k.get("question", "") for k in kwargs]

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(
                obs, 
                init=True, 
                verifier=verifier, 
                use_previous_solutions=use_previous_solutions,
                previous_solutions=kwargs[0].get("previous_solutions", "")
            ),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(
        self, 
        text_actions: List[str], 
        store_full_action: bool = False,
        verifier: bool = False, 
        use_previous_solutions: bool = False,
        previous_solutions: str = "",
        env_dones: List[bool] | None = None,
    ):
        actions, valids = self.projection_f(text_actions)

        # Track illegal actions so we can force termination and keep memory clean.
        force_done = [False] * len(actions)
        done_reasons: List[str | None] = [None] * len(actions)
        sanitized_proj_actions: List[str] = []
        sanitized_full_actions: List[str] = []

        for idx, (projected, raw) in enumerate(zip(actions, text_actions)):
            projected_str = projected or ""
            has_report = ("<report>" in projected_str.lower() and "</report>" in projected_str.lower())
            is_empty = not projected_str.strip()

            if is_empty:
                force_done[idx] = True
                done_reasons[idx] = "empty_action"
                valids[idx] = 0  # mark as invalid step
            elif has_report:
                force_done[idx] = True
                done_reasons[idx] = "report"

            sanitized_proj_actions.append(
                self._sanitize_action_for_memory(projected_str, done_reasons[idx])
            )
            sanitized_full_actions.append(
                self._sanitize_action_for_memory(raw or "", done_reasons[idx])
            )

        next_obs, rewards, dones, infos = self.envs.step(actions, text_actions)

        # If caller says an env is already done, keep it done and avoid changing its obs/reward.
        if env_dones is not None:
            for i, already_done in enumerate(env_dones):
                if already_done:
                    dones[i] = True
                    rewards[i] = 0
                    infos[i]["already_done"] = True
                    # Keep previous observation for that env (no new text)
                    if i < len(next_obs):
                        next_obs[i] = ""  # placeholder; upstream should ignore for done envs

        # Force termination for empty actions or report responses from verifier.
        for i in range(len(dones)):
            if force_done[i]:
                dones[i] = True
                infos[i]["forced_done"] = True
                infos[i]["done_reason"] = done_reasons[i]

        if self.config.env.informal_math_evolving.memory_type == "score" or self.config.env.informal_math_evolving.memory_type == "ndimensional":
            self.memory.store({
                "text_obs": next_obs,
                "action": sanitized_full_actions if store_full_action else sanitized_proj_actions,
                "meta_info": {
                    "score": rewards
                },
            })
        else:
            self.memory.store({
                "text_obs": next_obs,
                "action": sanitized_full_actions if store_full_action else sanitized_proj_actions,
            })

        next_observations = {
            "text": self.build_text_obs(next_obs, verifier=verifier, use_previous_solutions=use_previous_solutions, previous_solutions=previous_solutions),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False,
        verifier: bool = False,
        use_previous_solutions: bool = False,
        previous_solutions: str = ""
    ) -> List[str]:
        postprocess_text_obs: List[str] = []
        history_length = 0
        if hasattr(self.config, 'env') and hasattr(self.config.env, 'history_length'):
            history_length = self.config.env.history_length

        memory_ctx = None
        if not init and history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                history_length,
                obs_key="text_obs",
                action_key="action"
            )
        elif not init:
            memory_ctx = ["" for _ in range(len(text_obs))]
        
        # Set python code and local_rag flags
        enable_python_code = bool(OmegaConf.select(self.config, "env.informal_math_evolving.enable_python_code") or False)
        enable_local_rag = bool(OmegaConf.select(self.config, "env.informal_math_evolving.enable_local_rag") or False)

        for i in range(len(text_obs)):
            if init:
                if verifier:
                    # prompt for verifier agent
                    template = get_verifier_prompt(enable_python_code, use_history=False)
                    obs_i = template.format(
                        question=self.tasks[i],
                        policy_solution=text_obs[i],
                    )
                    # Store policy solution for subsequent steps
                    if not hasattr(self, 'policy_solutions'):
                        self.policy_solutions = {}
                    self.policy_solutions[i] = text_obs[i]
                else:
                    # prompt for policy agent
                    template = get_policy_prompt(enable_python_code, use_history=False, use_previous_solutions=use_previous_solutions, enable_local_rag=enable_local_rag)
                    # DEBUG: Check if template contains local_rag
                    has_local_rag_in_prompt = "<local_rag>" in template
                    obs_i = template.format(question=self.tasks[i], previous_solutions=previous_solutions)
            else:
                memory_entry = "" if not memory_ctx else memory_ctx[i]
                if verifier:
                    template = get_verifier_prompt(enable_python_code, use_history=True)
                    # Use stored policy solution
                    policy_sol = self.policy_solutions.get(i, "") if hasattr(self, 'policy_solutions') else ""
                    obs_i = template.format(
                        question=self.tasks[i],
                        policy_solution=policy_sol,
                        memory_context=memory_entry,
                        step_count=len(self.memory[i]),
                    )
                else:
                    # default prompt
                    template = get_policy_prompt(enable_python_code, use_history=True, use_previous_solutions=use_previous_solutions, enable_local_rag=enable_local_rag)
                    # DEBUG: Check if template contains local_rag
                    has_local_rag_in_prompt = "<local_rag>" in template
                    obs_i = template.format(
                        question=self.tasks[i],
                        memory_context=memory_entry,
                        step_count=len(self.memory[i]),
                        previous_solutions=previous_solutions
                    )
                    
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                if data_source is None:
                    tool_infos = info.get("tool_infos")
                    if isinstance(tool_infos, dict):
                        data_source = tool_infos.get("data_source", "unknown")
                    else:
                        data_source = "unknown"
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask


class InformalMathTrainingEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for InformalMathEnv.
    """
    def __init__(self, envs, projection_f, config):
        if config.env.informal_math.memory_type == "score":
            self.memory = EvolvingMemory(sort_key="score", descending=True)
        elif config.env.informal_math.memory_type == "ndimensional":
            self.memory = NDimensionalMemory(
                dimensions=["complexity", "performance"],
                performance_key="score", complexity_key="action"
            )
        elif config.env.informal_math.memory_type == "simple":
            self.memory = SimpleMemory()
        else:
            raise ValueError(f"Invalid memory type: {config.env.informal_math.memory_type}")
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions, text_actions)
        if self.config.env.informal_math.memory_type == "score" or self.config.env.informal_math.memory_type == "ndimensional":
            self.memory.store({
                "text_obs": next_obs,
                "action": text_actions,
                "meta_info": {
                    "score": rewards
                },
            })
        else:
            self.memory.store({
                "text_obs": next_obs,
                "action": text_actions,
            })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False,
        verifier: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []
        history_length = 0
        if hasattr(self.config, 'env') and hasattr(self.config.env, 'history_length'):
            history_length = self.config.env.history_length

        memory_ctx = None
        if not init and history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                history_length,
                obs_key="text_obs",
                action_key="action"
            )
        elif not init:
            memory_ctx = ["" for _ in range(len(text_obs))]
        
        # Set python code flag
        enable_python_code = bool(OmegaConf.select(self.config, "env.informal_math.enable_python_code") or False)
        # Set rag system flag
        enable_local_rag = bool(OmegaConf.select(self.config, "env.informal_math.enable_local_rag") or False)
        # Get execution mode (default to agentic)
        execution_mode = str(OmegaConf.select(self.config, "env.informal_math.execution_mode") or "agentic")
        
        # Build tool_config dict for prompt generation (easier to extend with more tools)
        tool_config = {
            "enable_python_code": enable_python_code,
            "enable_local_rag": enable_local_rag,
        }
        
        for i in range(len(text_obs)):
            if init:
                template = get_policy_training_prompt(use_history=False, max_steps=self.config.env.max_steps, tool_config=tool_config)
                obs_i = template.format(question=self.tasks[i])
            else:
                memory_entry = "" if not memory_ctx else memory_ctx[i]
                template = get_policy_training_prompt(use_history=True, max_steps=self.config.env.max_steps, tool_config=tool_config)
                obs_i = template.format(question=self.tasks[i], memory_context=memory_entry, step_count=len(self.memory[i]))
                    
            postprocess_text_obs.append(obs_i)
        return postprocess_text_obs


    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                if data_source is None:
                    tool_infos = info.get("tool_infos")
                    if isinstance(tool_infos, dict):
                        data_source = tool_infos.get("data_source", "unknown")
                    else:
                        data_source = "unknown"
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask



def make_envs(config):
    """
    Create enviroments 
    """ 
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(config.env.resources_per_worker, resolve=True)

    # ======================= 
    # InformalMath Training
    # ======================= 
    if "informal_math_training" in config.env.env_name.lower():
        from agent_system.environments.informal_math_training import (
            build_informal_math_training_envs,
            informal_math_training_projection)
        _envs = build_informal_math_training_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_informal_math_training_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(informal_math_training_projection)
        envs = InformalMathTrainingEnvironmentManager(_envs, projection_f, config)
        val_envs = InformalMathTrainingEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    
    # ======================= 
    # InformalMath Evolving
    # ======================= 
    elif "informal_math_evolving" in config.env.env_name.lower():
        from agent_system.environments.informal_math_evolving import (
            build_informal_math_evolving_envs,
            informal_math_evolving_projection)
        _envs = build_informal_math_evolving_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_informal_math_evolving_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(informal_math_evolving_projection)
        envs = InformalMathEvolvingEnvironmentManager(_envs, projection_f, config)
        val_envs = InformalMathEvolvingEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    
    else:
        raise ValueError(f"Environment {config.env.env_name} not supported")
    
