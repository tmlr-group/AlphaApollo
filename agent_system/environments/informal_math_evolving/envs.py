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

import asyncio
import concurrent.futures
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, ListConfig


class InformalMathEvolvingMultiProcessEnv(gym.Env):
    """
    - env_num  : Number of groups (logical sharding; keep the parameter for external compatibility)
    - group_n  : Number of environments per group
    - total_envs = env_num * group_n
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: DictConfig | None = None,
    ) -> None:
        super().__init__()

        from agent_system.environments.informal_math_evolving.env import \
            InformalMathEvolvingEnv

        self.env_num   = env_num
        self.group_n   = group_n
        self.batch_size = env_num * group_n
        self.is_train  = is_train
        self.max_steps = env_config.max_steps

        self._rng = np.random.RandomState(seed)

        # ---------- Key changes start ----------
        informal_math_evolving_cfg  = env_config.informal_math_evolving

        # 2) Assign configuration to each env in a round-robin manner
        self.envs = []
        for idx in range(self.batch_size):
            cfg_i = deepcopy(informal_math_evolving_cfg)
            self.envs.append(InformalMathEvolvingEnv(cfg_i))

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Track per-env step counts and emitted intermediate reward sums.
        self._step_counts = np.zeros(self.batch_size, dtype=int)
        self._intermediate_emitted_cumsum = np.zeros(self.batch_size, dtype=float)

    # initialize the environment with dataset information
    def _sync_reset(self, env, kwargs):
        extras = {
            "question": kwargs["question"],
            "ground_truth": kwargs["ground_truth"],
            "gt_traj": kwargs["gt_traj"],
            "max_steps": self.max_steps,
            "data_source": kwargs.get("data_source", "unknown"),
            "policy_solution": kwargs.get("policy_solution", None),
            "previous_solutions": kwargs.get("previous_solutions", None),
        }
        env.reset(extras)
        if kwargs.get("policy_solution", None) is not None:
            obs = extras["policy_solution"]
        else:
            if kwargs.get("previous_solutions", None) is not None:
                obs = extras.get("previous_solutions") + "\n" + extras["question"]
            else:
                obs = extras["question"] # the initial state for policy model should be the question
        info = {
            "data_source": kwargs.get("data_source", "unknown")
        }
        return obs, info
    
    # step the environment with the action
    def _sync_step(self, env, action: str, text_actions: List[str] = None):
        out = env.step(action, text_actions)
        try:
            obs = out["observations"]
            obs = "" if len(obs) == 0 or obs[0] is None else obs[0]["content"].strip()
            reward = out["reward"]
            done = out["done"]
        except Exception as e:
            print(f"Error in step: {e}, action: {action}")
            print(f"text_actions: {text_actions}")
            print(f"out: {out}")
            raise Exception(f"Error in step: {e}, action: {action}")

        info = out.get("metadata", [])
        info = {"tool_infos": info}  # wrap as a dict for external access
        info["postprocessed_action"] = out.get("postprocessed_action")
        info["won"] = bool(done and reward > 0.0)
        return obs, reward, done, info

    def reset(self, kwargs: List[Dict]):
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, but the env was initialised with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {
                    "question": "",
                    "ground_truth": "",
                    "gt_traj": "",
                    "max_steps": self.max_steps,
                    "data_source": "unknown",
                }

        padded_kwargs = list(kwargs) + [dummy_kw] * pad_n
        valid_mask = [True] * len(kwargs) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_reset, env, kw)
            for env, kw in zip(self.envs, padded_kwargs)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, info_list = map(list, zip(*results))

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        # Reset counters for a fresh episode window
        self._step_counts[:] = 0
        self._intermediate_emitted_cumsum[:] = 0.0

        return obs_list, info_list

    def step(self, actions: List[str], text_actions: List[str]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, but the env was initialized with total_envs={self.batch_size}")

        pad_n = self.batch_size - len(actions)
        padded_actions = list(actions) + [""] * pad_n
        valid_mask = [True] * len(actions) + [False] * pad_n

        tasks = [
            self._loop.run_in_executor(self._executor, self._sync_step, env, act, text_acts)
            for env, act, text_acts in zip(self.envs, padded_actions, text_actions)
        ]
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        obs_list, reward_list, done_list, info_list = map(list, zip(*results))

        # Update per-env counters and handle manager-timeout finalization
        for i in range(self.batch_size):
            if not valid_mask[i]:
                continue
            # Increment local step counter for valid envs
            self._step_counts[i] += 1
            if self._step_counts[i] >= self.max_steps and not done_list[i]:
                done_list[i] = True
                cur_r = float(reward_list[i])
                reward_list[i] = cur_r

        obs_list = [o for o, keep in zip(obs_list, valid_mask) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid_mask) if keep]
        done_list = [d for d, keep in zip(done_list, valid_mask) if keep]
        info_list = [i for i, keep in zip(info_list, valid_mask) if keep]

        return obs_list, reward_list, done_list, info_list

    def close(self):
        if getattr(self, "_closed", False):
            return
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True)
        if hasattr(self, '_loop') and self._loop is not None and not self._loop.is_closed():
            self._loop.close()
        self._closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            # Ignore exceptions during garbage collection
            pass


def build_informal_math_evolving_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config=None,
):
    return InformalMathEvolvingMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
    )