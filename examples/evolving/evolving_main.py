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

import copy
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict

import fire
import numpy as np
from omegaconf import OmegaConf

# Ensure repository root is on PYTHONPATH so local packages can be imported
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from agent_system.environments.env_manager import \
    InformalMathEvolvingEnvironmentManager
from agent_system.environments.prompts.informal_math_evolving import (
    INFORMAL_MATH_TEMPLATE_WITH_HIS_FORCE_ANSWER,
    INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_AND_HIS_FORCE_ANSWER,
    VERIFIER_AGENT_TEMPLATE_WITH_HIS_FORCE_REPORT,
    VERIFIER_REPORT_AGGREGATION_TEMPLATE,
    SUMMARIZER_TEMPLATE)
from examples.evolving.utils.agent import Agent
from examples.evolving.utils.dataset_loader import load_informal_math_data
from examples.evolving.utils.utils import (collect_tool_events,
                                           configure_color_logging,
                                           has_tag_block,
                                           load_run_configuration,
                                           save_problem_outputs,
                                           snapshot_memory)
from verl.utils.reward_score.math import (compute_score,
                                          last_boxed_only_string, remove_boxed)
from agent_system.memory.memory import NDimensionalMemory

log = logging.getLogger(__name__)


TAG_PATTERN_CACHE = {}
DEFAULT_CFG_PATH = Path(project_root) / "configs" / "vllm_informal_math.yaml"

# --------- Helper functions for verifier ---------

def extract_tag_block(text: str, tag: str) -> str:
    if not text:
        return ""
    if tag not in TAG_PATTERN_CACHE:
        TAG_PATTERN_CACHE[tag] = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
    match = TAG_PATTERN_CACHE[tag].search(text)
    return match.group(1).strip() if match else ""

def remove_tags_only(text: str, tag: str) -> str:
    """
    Removes only the tags themselves, preserving the content inside.
    Example: AAA<report>Hello</report>AAA -> Hello
    """
    pattern = rf"</?{tag}>"
    return re.sub(pattern, "", text)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} in text, checking answer tags first."""
    if not text:
        return None
    try:
        for tag in ["answer", "report"]:
            content = extract_tag_block(text, tag)
            if content and (boxed := last_boxed_only_string(content)):
                return remove_boxed(boxed)
        if boxed := last_boxed_only_string(text):
            return remove_boxed(boxed)
    except Exception:
        pass
    return None

def compute_answer_correctness(action: str, ground_truth: str, verifier_log: Optional[dict] = None) -> dict:
    """Compute policy answer correctness and verifier judgment correctness.
    
    Returns dict with: policy_answer, policy_answer_correct, verifier_judgment, verifier_judgment_correct
    """
    # Extract and check policy answer
    raw_answer = extract_tag_block(action, "answer") if action else None  # store original content
    boxed_answer = extract_boxed_answer(action)
    has_answer_tag = raw_answer is not None

    if boxed_answer is not None:
        correct = compute_score(f"\\boxed{{{boxed_answer}}}", ground_truth) > 0.5
        answer_for_log = boxed_answer
    elif has_answer_tag:
        # Has <answer> tag but no boxed value -> store raw text, mark incorrect
        correct = False
        answer_for_log = raw_answer
    else:
        correct, answer_for_log = None, None
    
    # Extract and check verifier judgment
    v_judgment, v_correct = None, None
    if verifier_log:
        # Handle both formats: direct dict with "report" or {env_idx: {...}} format
        if 0 in verifier_log or "0" in verifier_log:
            report = verifier_log.get(0, verifier_log.get("0", {})).get("report", "")
        else:
            report = verifier_log.get("report", "")
        
        if report and (judgment_str := extract_boxed_answer(report)) and judgment_str in ["0", "1"]:
            v_judgment = int(judgment_str)
            v_correct = (v_judgment == 1) == correct if correct is not None else None
    
    return {
        "policy_answer": answer_for_log,
        "policy_answer_correct": correct,
        "verifier_judgment": v_judgment,
        "verifier_judgment_correct": v_correct
    }

def ensure_verifier_format(agent, verifier_obs, max_retries=5):
    """
    Ensure the verifier output is in the correct format with fallback retry logic.
    """
    for _ in range(max_retries):
        action = agent.get_action_from_gpt(verifier_obs)
        if has_tag_block(action, "report") and extract_boxed_answer(action) in ["0", "1"]:
            return action
    return "<report>Failed to verify.\nJudgment: \\boxed{0}</report>"

def aggregate_verifier_judgments(verifier_actions, aggregator_config=None, question="", policy_solution=""):
    """
    Aggregate verifier judgments using majority voting and LLM-based report aggregation.
    
    Args:
        verifier_actions: List of verifier action strings
        aggregator_config: Configuration for creating Agent instance (optional, for report aggregation)
        question: The original question (for aggregation prompt)
        policy_solution: The policy solution being verified (for aggregation prompt)
        
    Returns:
        tuple: (majority_judgment: int, representative_report: str, judgment_counts: dict)
            - majority_judgment: 0 or 1 based on majority vote
            - representative_report: Aggregated report from LLM or first matching report
            - judgment_counts: Dict with counts for each judgment
    """
    judgments = []
    reports = []
    
    for action in verifier_actions:
        report = extract_tag_block(action, "report")
        judgment_str = extract_boxed_answer(action)
        judgment = int(judgment_str) if judgment_str in ["0", "1"] else 0
        judgments.append(judgment)
        reports.append(report if report else action)
    
    if not judgments:
        return 0, "<report>Failed to verify.\nJudgment: \\boxed{0}</report>", {0: 0, 1: 0}
    
    # Count judgments
    judgment_counts = Counter(judgments)
    majority_judgment = judgment_counts.most_common(1)[0][0]
    
    # Collect reports matching majority judgment
    majority_reports = [reports[i] for i, j in enumerate(judgments) if j == majority_judgment]
    
    # Aggregate reports using LLM if aggregator_config is provided and multiple reports exist
    if aggregator_config and len(majority_reports) > 1:
        agent = Agent(aggregator_config)
        individual_reports = "\n\n".join(
            [f"--- Verifier {i+1} Report ---\n{r}" for i, r in enumerate(majority_reports)]
        )
        aggregation_prompt = VERIFIER_REPORT_AGGREGATION_TEMPLATE.format(
            question=question,
            policy_solution=policy_solution,
            majority_judgment=majority_judgment,
            individual_reports=individual_reports
        )
        print("===> Aggregation verifier prompt:", aggregation_prompt)
        aggregated_response = agent.get_action_from_gpt(aggregation_prompt)
        print("===> Aggregation verifier report:", aggregated_response)
        representative_report = extract_tag_block(aggregated_response, "report")
        if not representative_report:
            representative_report = majority_reports[0]
    else:
        print("Warning: Aggregator config not provided or single report; using first majority report as representative.")
        representative_report = majority_reports[0] if majority_reports else None
    
    if representative_report is None:
        representative_report = reports[0] if reports else "<report>Failed to verify.\nJudgment: \\boxed{0}</report>"
    
    return majority_judgment, representative_report, dict(judgment_counts)
    
# ----------------------------------------------------

# --------- Helper functions for environment ---------
def load_problems(run_cfg, debug_mode: bool = False, problem_idx=None):
    data_path = os.path.join(run_cfg["data_root"], run_cfg["dataset_name"], run_cfg["file_name"])
    assert os.path.exists(data_path), f"data_path {data_path} does not exist"

    log.info("Loading data from %s", data_path)
    problems = load_informal_math_data(data_path)
    assert problems, "No problems loaded"
    if debug_mode:
        problems = problems[:1]
    if problem_idx is not None:
        problems = [problems[problem_idx]]
    log.info("Loaded %d problems for evaluation", len(problems))
    return problems


def build_env(env_name, env_num=1, group_n=1, env_config=None):
    assert env_name == "informal_math_evolving", f"Unsupported environment name: {env_name}"
    assert env_config, "env.config must be provided via the YAML configuration."
    
    from agent_system.environments.informal_math_evolving import (
        build_informal_math_evolving_envs, informal_math_evolving_projection)

    env_config = OmegaConf.create(env_config)
    envs = build_informal_math_evolving_envs(seed=1, env_num=env_num, group_n=group_n, env_config=env_config)
    manager_config = OmegaConf.create({
        'env': {
            'env_name': 'informal_math_evolving',
            'history_length': int(env_config.history_length),
            'max_steps': int(env_config.max_steps),
            'informal_math_evolving': {
                'memory_type': str(env_config.informal_math_evolving.memory_type),
                "enable_verify": env_config.informal_math_evolving.enable_verify,
                "enable_python_code": env_config.informal_math_evolving.enable_python_code,
                "enable_local_rag": env_config.informal_math_evolving.enable_local_rag,
                "execution_mode": str(OmegaConf.select(env_config, "informal_math_evolving.execution_mode") or "agentic"),
            }
        }
    })
    return InformalMathEvolvingEnvironmentManager(envs, informal_math_evolving_projection, manager_config)


def fetch_contexts(memory, history_length, env_num):
    if history_length <= 0 or not memory:
        return ["" for _ in range(env_num)]
    text_obs, _ = memory.fetch(history_length, "text_obs", "action")
    return text_obs

def collect_actions_from_gpt(agent, obs, env_dones):
    texts = obs.get("text", [])
    return ["None" if env_dones[i] else agent.get_action_from_gpt(texts[i]) for i in range(len(env_dones))]


def collect_actions_from_gpt_parallel(agent, obs, env_dones, max_workers: Optional[int] = None):
    """
    Parallel version of collect_actions_from_gpt using ThreadPoolExecutor.
    
    Args:
        agent: The agent to collect actions from
        obs: Observation dict containing "text" key
        env_dones: List of boolean flags indicating which environments are done
        max_workers: Maximum number of parallel workers (default: number of active environments)
    
    Returns:
        List of actions for each environment
    """
    texts = obs.get("text", [])
    n_envs = len(texts)
    
    # Find active (not done) environment indices
    active_indices = [i for i in range(n_envs) if not env_dones[i]]
    
    if not active_indices:
        return ["None"] * n_envs
    
    # Default workers to number of active environments
    if max_workers is None:
        max_workers = len(active_indices)
    
    actions = ["None"] * n_envs
    
    def fetch_action(idx: int) -> Tuple[int, str]:
        return idx, agent.get_action_from_gpt(texts[idx])
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_action, i): i for i in active_indices}
        for future in as_completed(futures):
            try:
                idx, action = future.result()
                actions[idx] = action
            except Exception as e:
                idx = futures[future]
                log.error(f"Error fetching action for env {idx}: {e}")
                actions[idx] = "None"
    
    return actions


def verifier_repeat_sampling_single(
    verifier_agent, 
    question: str,
    policy_solution: str, 
    memory_context: str,
    step_count: int,
    env_idx: int,
    max_retries: int = 5
) -> Tuple[int, str, bool]:
    """
    Single verifier repeat sampling (thread-safe version without env_manager access).
    
    Returns:
        Tuple of (env_idx, action, success)
    """
    obs = VERIFIER_AGENT_TEMPLATE_WITH_HIS_FORCE_REPORT.format(
        question=question,
        policy_solution=policy_solution,
        memory_context=memory_context,
        step_count=step_count
    )
    for _ in range(max_retries):
        action = verifier_agent.get_action_from_gpt(obs)
        if has_tag_block(action, "report"):
            return env_idx, action, True
    return env_idx, "<report>Failed to generate report.\nReport: \\boxed{{0}}</report>", False


def verifier_repeat_sampling_parallel(
    verifier_agent,
    verifier_env_manager,
    policy_solution: str,
    history_length: int,
    env_indices: List[int],
    final_verifier_actions: List[str],
    max_retries: int = 5,
    max_workers: Optional[int] = None
) -> List[Tuple[int, str, bool]]:
    """
    Parallel verifier repeat sampling for multiple environments.
    
    Args:
        verifier_agent: The verifier agent
        verifier_env_manager: The verifier environment manager
        policy_solution: The policy solution to verify
        history_length: History length for memory context
        env_indices: List of environment indices that need repeat sampling
        final_verifier_actions: Current final actions (will be updated in-place)
        max_retries: Maximum retry attempts per environment
        max_workers: Maximum parallel workers
    
    Returns:
        List of (env_idx, action, success) tuples
    """
    if not env_indices:
        return []
    
    if max_workers is None:
        max_workers = len(env_indices)
    
    # Pre-fetch all required data from env_manager (not thread-safe)
    tasks_data = []
    for env_idx in env_indices:
        if history_length > 0:
            memory_context, _ = verifier_env_manager.memory.fetch(history_length, "text_obs", "action")
            memory_context = memory_context[env_idx]
        else:
            memory_context = ""
        question = verifier_env_manager.tasks[env_idx]
        step_count = len(verifier_env_manager.memory[env_idx])
        tasks_data.append((env_idx, question, memory_context, step_count))
    
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                verifier_repeat_sampling_single,
                verifier_agent,
                question,
                policy_solution,
                memory_context,
                step_count,
                env_idx,
                max_retries
            ): env_idx
            for env_idx, question, memory_context, step_count in tasks_data
        }
        
        for future in as_completed(futures):
            try:
                env_idx, action, success = future.result()
                results.append((env_idx, action, success))
                # Update final_verifier_actions in-place
                final_verifier_actions[env_idx] = action
            except Exception as e:
                env_idx = futures[future]
                log.error(f"Error in verifier repeat sampling for env {env_idx}: {e}")
                results.append((env_idx, "<report>Failed to generate report.\nReport: \\boxed{{0}}</report>", False))
    
    return results

# ---------------------------------------------------------------


# --------- Repeat sampling for answer and report ---------
def policy_repeat_sampling(policy_agent, policy_env_manager, history_length, env_idx, previous_solutions="", max_retries=5) -> str:
    if history_length > 0:
        memory_context, _ = policy_env_manager.memory.fetch(history_length, "text_obs", "action")
        memory_context = memory_context[env_idx]
    else:
        memory_context = ""
    question = policy_env_manager.tasks[env_idx]
    step_count=len(policy_env_manager.memory[env_idx])
    if len(previous_solutions) > 0:
        obs = INFORMAL_MATH_TEMPLATE_WITH_PREVIOUS_SOLUTIONS_AND_HIS_FORCE_ANSWER.format(
            question=question,
            previous_solutions=previous_solutions,
            memory_context=memory_context,
            step_count=step_count,
        )
    else:
        obs = INFORMAL_MATH_TEMPLATE_WITH_HIS_FORCE_ANSWER.format(
        question=question,
        memory_context=memory_context,
        step_count=step_count
    )
    for _ in range(max_retries):
        action = policy_agent.get_action_from_gpt(obs)
        if has_tag_block(action, "answer"):
            return action, True
    return "<answer>Failed to generate answer.\nAnswer: \\boxed{}</answer>", False

def verifier_repeat_sampling(verifier_agent, verifier_env_manager, policy_solution, history_length, env_idx, max_retries=5) -> str:
    if history_length > 0:
        memory_context, _ = verifier_env_manager.memory.fetch(history_length, "text_obs", "action")
        memory_context = memory_context[env_idx]
    else:
        memory_context = ""
    question = verifier_env_manager.tasks[env_idx]
    step_count=len(verifier_env_manager.memory[env_idx])
    obs = VERIFIER_AGENT_TEMPLATE_WITH_HIS_FORCE_REPORT.format(
        question=question,
        policy_solution=policy_solution,
        memory_context=memory_context,
        step_count=step_count
    )
    for _ in range(max_retries):
        action = verifier_agent.get_action_from_gpt(obs)
        if has_tag_block(action, "report"):
            return action, True
    return "<report>Failed to generate report.\nReport: \\boxed{{0}}</report>", False

# -------------------------------------------

# ------ Main functions for policy ---------
def run_problem(problem_idx, current_problem, runtime):
    policy_env_manager = runtime["policy_env_manager"]
    verifier_configs = runtime["verifier_configs"]
    verifier_max_workers = runtime.get("verifier_max_workers", 0)  # 0 = sequential

    if verifier_configs.get("enabled"):
        verifier_env_manager = verifier_configs["verifier_env_manager"]
        verifier_agent = verifier_configs["verifier_agent"]
        verifier_env_num = verifier_configs["verifier_env_num"]
        verifier_env_max_steps = verifier_configs["verifier_env_max_steps"]
        verifier_env_history_length = verifier_configs["verifier_env_history_length"]
        verifier_memory_depth = verifier_env_history_length if verifier_env_history_length > 0 else 2

    policy_env_num = runtime["policy_env_num"]
    policy_env_max_steps = runtime["policy_env_max_steps"]
    policy_agent = runtime["policy_agent"]
    policy_env_history_length = runtime["policy_env_history_length"]
    policy_env_memory_depth = policy_env_history_length if policy_env_history_length > 0 else 2
    solution_memory_depth = runtime["scored_history_length"]

    policy_overall_success = np.zeros(policy_env_num, dtype=bool)
    task_success_cnt = defaultdict(int)
    task_total_cnt = defaultdict(int)
    step_outputs = []

    previous_solutions = []

    solution_memory = NDimensionalMemory(dimensions=["complexity", "performance"], performance_key="score", complexity_key="action")
    solution_memory.reset(batch_size=policy_env_num)

    start_time = time.time()    
    for evolving_round in range(runtime["evolving_round"]):
        print(f"===> Evolving round {evolving_round}, reset all the environments...")
        # reset the policy and verifier memory and environment

        # NOTE: now the all the policies share the same solution memory for the same problem, so the batch size is fixed as 1
        previous_solutions_nd, valid_length = solution_memory.fetch(history_length=solution_memory_depth, obs_key="text_obs", action_key="action")

        if valid_length[0] > 0:
            current_problem["previous_solutions"] = "\n".join(previous_solutions_nd)
            obs, _ = policy_env_manager.reset(kwargs=[current_problem], use_previous_solutions=True)
        else:
            obs, _ = policy_env_manager.reset(kwargs=[current_problem])
        
        print(f"===> Policy model input: {obs['text'][0]}")
        policy_env_dones = [False] * policy_env_num

        for step_idx in range(policy_env_max_steps):
            done_count = int(np.array(policy_env_dones).sum())
            log.info("===> Round %s, Step %s; Dones (%s/%s); SR %.4f", evolving_round, step_idx, done_count, policy_env_num, policy_overall_success.mean().item())

            policy_obs_snapshot = copy.deepcopy(obs)
            # ----- Policy model - Step1: generate actions -----
            actions = collect_actions_from_gpt(policy_agent, obs, policy_env_dones)
            
            # If it is the last step, we must ensure the action contains an answer
            if step_idx == policy_env_max_steps - 1:
                for i in range(policy_env_num):
                    if not policy_env_dones[i] and not has_tag_block(actions[i], "answer"):
                        actions[i], success = policy_repeat_sampling(
                            policy_agent, policy_env_manager, policy_env_history_length, i,
                            previous_solutions=previous_solutions_nd
                        )

            policy_memory_snapshot = snapshot_memory(policy_env_manager.memory, limit=policy_env_memory_depth)
            
            # ----- Policy model - Step2: excute the actions -----
            next_policy_obs, rewards, dones, infos = policy_env_manager.step(
                actions, 
                store_full_action=True,
                use_previous_solutions=valid_length[0] > 0,
                previous_solutions=previous_solutions_nd
            )
            rewards_snapshot = rewards.tolist() if hasattr(rewards, "tolist") else rewards
            dones_snapshot = dones.tolist() if hasattr(dones, "tolist") else dones

            # ----- Policy model - Step3: update the success status -----
            # if <answer> tags are found in the actions, then the policy environment is done, enter the verifier session
            for i in range(policy_env_num):
                if policy_env_dones[i]:
                    continue
                if dones[i]:
                    policy_env_dones[i] = True
                    won = bool(infos[i].get("won", False))
                    policy_overall_success[i] = won
                    task_total_cnt["informal_math"] += 1
                    if won:
                        task_success_cnt["informal_math"] += 1

            # ----- Policy model - Step4: collect tool events -----
            tool_events = collect_tool_events(policy_env_manager.memory, infos, actions)
            correctness = compute_answer_correctness(actions[0] if actions else "", current_problem["ground_truth"])
            if correctness.get("policy_answer_correct"):
                policy_overall_success[0] = True

            step_outputs.append({
                "role": "policy",
                "step": step_idx,
                "observation": policy_obs_snapshot,"policy_actions": actions,
                "next_observation": copy.deepcopy(next_policy_obs),"rewards": rewards_snapshot,
                "dones": dones_snapshot,"infos": copy.deepcopy(infos),
                "verifier": None,"ground_truth": current_problem["ground_truth"],
                "gt_traj": current_problem["gt_traj"],"data_source": current_problem["data_source"],
                "policy_memory": policy_memory_snapshot,"verifier_memory": None,
                "tool_events": tool_events,"evolving_round": evolving_round,
                "policy_answer": correctness.get("policy_answer"),
                "policy_answer_correct": correctness.get("policy_answer_correct"),
            })
            # ----- Policy model - Step5: update the observation -----
            obs = next_policy_obs

            if all(policy_env_dones):
                log.info("===> All environments finished early!")
                break
        
        print("===> Policy model generated complete, start checking the answer format for each environment...")
        # ----- Policy model - Step6: repeat sampling for answer -----
        envs_has_answers = np.ones(policy_env_num, dtype=bool)
        for i in range(policy_env_num):
            if not has_tag_block(actions[i], "answer"):
                actions[i], success = policy_repeat_sampling(
                    policy_agent, policy_env_manager, policy_env_history_length, i,
                    previous_solutions=previous_solutions_nd
                )
                if not success:
                    envs_has_answers[i] = False
        
        # ! We now only have one single environment, so we can directly use the actions[0] to get the policy solution
        print(f"===> Policy model output: {actions[0]}")
        
        if verifier_configs.get("enabled"):
            verifier_env_num = runtime['verifier_configs']['verifier_env_num']
            verifier_env_max_steps = runtime['verifier_configs']['verifier_env_max_steps']
            verifier_env_history_length = runtime['verifier_configs']['verifier_env_history_length']
            verifier_memory_depth = verifier_env_history_length if verifier_env_history_length > 0 else 2
            verifier_dones = [False] * verifier_env_num
            
            print("===> Policy model's memory:", policy_env_manager.memory._data)
            policy_full_output = policy_env_manager.memory.fetch(history_length=1, obs_key="text_obs", action_key="action")[0][0]
            print("===> Policy model's full output:", policy_full_output)

            # ! extract the answer from the policy model's full output

            # print("===> Policy model's answer:", policy_answer)

            # ----- Policy model - Step7: summarization -----
            summarization_agent = Agent(runtime["verifier_configs"]["verifier_cfg"])
            policy_summary = summarization_agent.get_action_from_gpt(SUMMARIZER_TEMPLATE.format(content=policy_full_output))
            print("===> Summarization model's summary:", policy_summary)
            del summarization_agent
            current_problem["policy_solution"] = policy_summary
            
            # Broadcast problem to all verifier environments
            verifier_obs, _ = verifier_env_manager.reset(kwargs=[current_problem] * verifier_env_num, verifier=True)
        else:
            continue
        
        # ----- Verifier model - Step1: verify the policy model's answer -----
        final_verifier_actions = [""] * verifier_env_num
        for verifier_step_idx in range(verifier_env_max_steps):
            print(f"===> Verifier step {verifier_step_idx}")

            verifier_obs_snapshot = copy.deepcopy(verifier_obs)
            for i in range(verifier_env_num):
                if verifier_dones[i]:
                    print(f"===> Round {evolving_round}, Environment {i} is already done!")
                    continue
                print(f"===> Round {evolving_round}, Environment {i}, Verifier input: {verifier_obs['text'][i]}")
            # ----- Verifier model - Step1: generate actions (parallel) -----
            if verifier_max_workers > 0:
                verifier_actions = collect_actions_from_gpt_parallel(
                    verifier_agent, verifier_obs, verifier_dones, max_workers=verifier_max_workers
                )
            else:
                verifier_actions = collect_actions_from_gpt(verifier_agent, verifier_obs, verifier_dones)
            
            # Update final actions for active environments
            for i in range(verifier_env_num):
                if not verifier_dones[i]:
                    final_verifier_actions[i] = verifier_actions[i]

            # If it is the last step, we must ensure the action contains a report (parallel)
            if verifier_step_idx == verifier_env_max_steps - 1:
                need_retry_indices = [
                    i for i in range(verifier_env_num)
                    if not verifier_dones[i] and not has_tag_block(verifier_actions[i], "report")
                ]
                if need_retry_indices:
                    if verifier_max_workers > 0:
                        # Parallel retry
                        verifier_repeat_sampling_parallel(
                            verifier_agent,
                            verifier_env_manager,
                            current_problem["policy_solution"],
                            verifier_env_history_length,
                            need_retry_indices,
                            verifier_actions,  # Will be updated in-place
                            max_retries=5,
                            max_workers=verifier_max_workers
                        )
                        for i in need_retry_indices:
                            final_verifier_actions[i] = verifier_actions[i]
                    else:
                        # Sequential retry (original behavior)
                        for i in need_retry_indices:
                            verifier_actions[i], success = verifier_repeat_sampling(
                                verifier_agent, verifier_env_manager, 
                                current_problem["policy_solution"], verifier_env_history_length, i
                            )
                            if success:
                                final_verifier_actions[i] = verifier_actions[i]

            verifier_memory_snapshot = snapshot_memory(verifier_env_manager.memory, limit=verifier_memory_depth)
            
            # ----- Verifier model - Step2: excute the actions -----
            next_verifier_obs, rewards, dones, infos = verifier_env_manager.step(
                verifier_actions,
                verifier=True,
                store_full_action=True,
                env_dones=verifier_dones,
            ) # TODO: make store_full_action as configurable
            rewards_snapshot = rewards.tolist() if hasattr(rewards, "tolist") else rewards
            dones_snapshot = dones.tolist() if hasattr(dones, "tolist") else dones

            if verifier_env_num > 1:
                print(f"===> Verifier outputs ({verifier_env_num} environments):")
                for i, action in enumerate(verifier_actions):
                    judgment = extract_boxed_answer(action)
                    print(f"  Env {i}: judgment={judgment}, action={action[:100]}...")
            else:
                print(f"===> Verifier output: {verifier_actions[0]}")

            # ----- Verifier model - Step3: update the success status -----
            # if <answer> tags are found in the actions, then the policy environment is done, enter the verifier session
            for i in range(verifier_env_num):
                if verifier_dones[i]:
                    continue
                if dones[i]:
                    verifier_dones[i] = True

            # ----- Verifier model - Step4: collect tool events -----
            tool_events = collect_tool_events(verifier_env_manager.memory, infos, verifier_actions)

            step_outputs.append({
                "role": "verifier",
                "step": verifier_step_idx,
                "observation": verifier_obs_snapshot,
                "policy_actions": verifier_actions,
                "next_observation": copy.deepcopy(next_verifier_obs),
                "rewards": rewards_snapshot,
                "dones": dones_snapshot,
                "infos": copy.deepcopy(infos),
                "verifier": None,
                "ground_truth": current_problem["ground_truth"],
                "gt_traj": current_problem["gt_traj"],
                "data_source": current_problem["data_source"],
                "policy_memory": None,
                "verifier_memory": verifier_memory_snapshot,
                "tool_events": tool_events,
                "evolving_round": evolving_round,
            })
            # ----- Verifier model - Step5: update the observation -----
            verifier_obs = next_verifier_obs
            if all(verifier_dones):
                log.info("===> All environments finished early!")
                break
        
        # ----- Verifier model - Step6: repeat sampling for answer (parallel) -----
        envs_has_reports = np.zeros(verifier_env_num, dtype=bool)
        need_retry_indices = [
            i for i in range(verifier_env_num)
            if not has_tag_block(final_verifier_actions[i], "report")
        ]
        
        if need_retry_indices:
            if verifier_max_workers > 0:
                # Parallel retry
                results = verifier_repeat_sampling_parallel(
                    verifier_agent,
                    verifier_env_manager,
                    current_problem["policy_solution"],
                    verifier_env_history_length,
                    need_retry_indices,
                    final_verifier_actions,  # Will be updated in-place
                    max_retries=5,
                    max_workers=verifier_max_workers
                )
                for env_idx, action, success in results:
                    envs_has_reports[env_idx] = success
            else:
                # Sequential retry (original behavior)
                for i in need_retry_indices:
                    final_verifier_actions[i], success = verifier_repeat_sampling(
                        verifier_agent, verifier_env_manager,
                        current_problem["policy_solution"], verifier_env_history_length, i
                    )
                    envs_has_reports[i] = success
        else:
            envs_has_reports[:] = True

        # Aggregate final verifier judgments using majority voting and LLM-based report aggregation
        majority_judgment, representative_report, judgment_counts = aggregate_verifier_judgments(
            final_verifier_actions,
            aggregator_config=verifier_configs.get("verifier_cfg"),
            question=verifier_env_manager.tasks[0],
            policy_solution=current_problem["policy_solution"]
        )
        print(f"===> Verifier judgments: {judgment_counts}, Majority: {majority_judgment}")
        print(f"===> Verifier report (representative): {representative_report}")

        # Compute verifier judgment correctness based on aggregated majority judgment
        # correctness was computed earlier for policy answer; now add verifier judgment correctness
        policy_answer_correct = correctness.get("policy_answer_correct")
        # verifier_judgment_correct: True if verifier's majority judgment aligns with policy correctness
        # majority_judgment == 1 means verifier approved the answer
        # If policy is correct and verifier approved (1), or policy is wrong and verifier rejected (0), verifier is correct
        if policy_answer_correct is not None:
            verifier_judgment_correct = (majority_judgment == 1) == policy_answer_correct
        else:
            verifier_judgment_correct = None
        
        print(f"===> Policy correctness: {policy_answer_correct}, Verifier judgment correct: {verifier_judgment_correct}")

        step_outputs.append({
            "role": "verifier_aggregation",
            "step": verifier_env_max_steps,
            "evolving_round": evolving_round,
            "majority_judgment": majority_judgment,
            "judgment_counts": judgment_counts,
            "representative_report": representative_report,
            "final_verifier_actions": copy.deepcopy(final_verifier_actions),
            "policy_answer_correct": policy_answer_correct,
            "verifier_judgment": majority_judgment,
            "verifier_judgment_correct": verifier_judgment_correct,
        })

        verifier_report = representative_report
        if verifier_report:
            verifier_report = remove_tags_only(verifier_report,"report")
        else:
            verifier_report = remove_tags_only(final_verifier_actions[0],"report")
        solution_memory.store({
            "text_obs": [verifier_report],
            "action": [current_problem["policy_solution"]],
            "score": [majority_judgment],
        })

    elapsed = time.time() - start_time

    # Compute verifier correctness statistics from verifier_aggregation steps
    # This uses the aggregated majority judgment after all verifiers have voted
    verifier_correctness_stats = {"policy_correct_verifier_correct": 0, "policy_correct_verifier_wrong": 0, "policy_wrong_verifier_correct": 0, "policy_wrong_verifier_wrong": 0}
    for step in step_outputs:
        # Only count verifier_aggregation steps for accurate per-round statistics
        if step.get("role") == "verifier_aggregation":
            pc, vc = step.get("policy_answer_correct"), step.get("verifier_judgment_correct")
            if pc is not None and vc is not None:
                key = f"policy_{'correct' if pc else 'wrong'}_verifier_{'correct' if vc else 'wrong'}"
                verifier_correctness_stats[key] += 1
    
    # Use task counters to compute success rate; avoid being overwritten by later rounds
    total_trials = sum(task_total_cnt.values())
    success_trials = sum(task_success_cnt.values())
    success_rate = (success_trials / total_trials) if total_trials > 0 else 0.0
    task_success_cnt_dict = dict(task_success_cnt)
    task_total_cnt_dict = dict(task_total_cnt)
    
    problem_payload = {
        "problem_index": problem_idx,
        "question": current_problem.get("question"),
        "ground_truth": current_problem.get("ground_truth"),
        "gt_traj": current_problem.get("gt_traj"),
        "data_source": current_problem.get("data_source"),
        "full_config": runtime.get("full_config"),
        "env_config": {
            "policy_env_num": policy_env_num,
            "verifier_env_num": verifier_env_num,
            "policy_env_max_steps": policy_env_max_steps,
            "policy_env_history_length": policy_env_history_length,
            "verifier_max_steps": verifier_env_max_steps if verifier_configs.get("enabled") else 0,
            "verifier_history_length": verifier_env_history_length if verifier_configs.get("enabled") else 0,
        },
        "step_outputs": step_outputs,
        "summary": {
            "success_rate": success_rate,
            "elapsed": elapsed,
            "task_success_cnt": task_success_cnt_dict,
            "task_total_cnt": task_total_cnt_dict,
            "verifier_correctness_stats": verifier_correctness_stats,
        },
    }
    return {
        "problem_payload": problem_payload,
        "step_outputs": step_outputs,
        "success_rate": success_rate,
        "task_success_cnt": task_success_cnt_dict,
        "task_total_cnt": task_total_cnt_dict,
        "elapsed": elapsed,
    }
# ---------------------------------------------------------------


# ------ Helper functions for problem-level parallelization ---------
def create_runtime_for_problem(
    cfg_bundle: Dict,
    env_config,
    policy_agent,
    verifier_agent,
    full_config: Dict,
    verifier_max_workers: int = 0
) -> Dict:
    """
    Create an independent runtime for a single problem execution.
    Each problem gets its own env_managers to avoid state conflicts.
    Agents are shared (thread-safe).
    """
    policy_env_num = cfg_bundle["policy_env_num"]
    verifier_env_num = cfg_bundle["verifier_env_num"]
    env_name = cfg_bundle["env_name"]
    group_n = cfg_bundle["group_n"]
    policy_env_max_steps = cfg_bundle["policy_env_max_steps"]
    policy_env_history_length = cfg_bundle["policy_env_history_length"]
    verifier_env_max_steps = cfg_bundle["verifier_env_max_steps"]
    verifier_env_history_length = cfg_bundle["verifier_env_history_length"]
    scored_history_length = cfg_bundle["scored_history_length"]
    enable_verify = cfg_bundle["enable_verify"]
    verifier_cfg = cfg_bundle["verifier_cfg"]
    evolving_round = int(OmegaConf.select(env_config, "informal_math_evolving.evolving_round")) or 1
    
    # Create policy env_config
    policy_env_config = copy.deepcopy(env_config)
    policy_env_config.max_steps = policy_env_max_steps
    policy_env_config.history_length = policy_env_history_length
    policy_env_config.informal_math_evolving.memory_type = str(OmegaConf.select(env_config, "informal_math_evolving.policy_env.memory_type"))
    policy_env_config.informal_math_evolving.enable_python_code = bool(OmegaConf.select(env_config, "informal_math_evolving.policy_env.enable_python_code"))
    policy_env_config.informal_math_evolving.enable_local_rag = bool(OmegaConf.select(env_config, "informal_math_evolving.policy_env.enable_local_rag") or False)
    
    # Build policy env_manager for this problem
    policy_env_manager = build_env(
        env_name,
        env_num=policy_env_num,
        group_n=group_n,
        env_config=policy_env_config,
    )
    
    # Build verifier env_manager for this problem
    verifier_env_manager = None
    if enable_verify:
        verifier_env_config = copy.deepcopy(env_config)
        verifier_env_config.max_steps = verifier_env_max_steps
        verifier_env_config.history_length = verifier_env_history_length
        verifier_env_config.informal_math_evolving.memory_type = str(OmegaConf.select(env_config, "informal_math_evolving.verifier_env.memory_type"))
        verifier_env_config.informal_math_evolving.enable_python_code = bool(OmegaConf.select(env_config, "informal_math_evolving.verifier_env.enable_python_code"))
        verifier_env_config.informal_math_evolving.enable_local_rag = bool(OmegaConf.select(env_config, "informal_math_evolving.verifier_env.enable_local_rag") or False)
        
        verifier_env_manager = build_env(
            env_name,
            env_num=verifier_env_num,
            group_n=1,
            env_config=verifier_env_config,
        )
    
    return {
        "full_config": full_config,
        "policy_env_num": policy_env_num,
        "evolving_round": evolving_round,
        "policy_agent": policy_agent,  # Shared agent (thread-safe)
        "policy_env_manager": policy_env_manager,  # Independent per problem
        "policy_env_max_steps": policy_env_max_steps,
        "policy_env_history_length": policy_env_history_length,
        "scored_history_length": scored_history_length,
        "verifier_max_workers": verifier_max_workers,
        "verifier_configs": {
            "enabled": bool(enable_verify),
            "verifier_env_num": verifier_env_num,
            "verifier_env_manager": verifier_env_manager,  # Independent per problem
            "verifier_agent": verifier_agent,  # Shared agent (thread-safe)
            "verifier_cfg": verifier_cfg if enable_verify else None,
            "verifier_env_max_steps": verifier_env_max_steps if enable_verify else 0,
            "verifier_env_history_length": verifier_env_history_length if enable_verify else 0,
        },
    }


def run_single_problem_wrapper(args: Tuple) -> Dict:
    """
    Wrapper function for running a single problem in parallel.
    Unpacks arguments and calls run_problem.
    """
    problem_idx, current_problem, cfg_bundle, env_config, policy_agent, verifier_agent, full_config, verifier_max_workers = args
    
    try:
        # Create independent runtime for this problem
        runtime = create_runtime_for_problem(
            cfg_bundle, env_config, policy_agent, verifier_agent, full_config, verifier_max_workers
        )
        
        # Run the problem
        result = run_problem(problem_idx, current_problem, runtime)
        result["problem_idx"] = problem_idx
        result["success"] = True
        return result
    except Exception as e:
        log.error(f"Error running problem {problem_idx}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "problem_idx": problem_idx,
            "success": False,
            "error": str(e),
            "success_rate": 0.0,
            "task_success_cnt": {},
            "task_total_cnt": {},
            "elapsed": 0.0,
            "problem_payload": None,
        }


def run_problems_parallel(
    problems: List[Dict],
    cfg_bundle: Dict,
    env_config,
    policy_agent,
    verifier_agent,
    full_config: Dict,
    verifier_max_workers: int = 0,
    problem_max_workers: int = 4,
    test_idx: int = 0,
) -> List[Dict]:
    """
    Run multiple problems in parallel using ThreadPoolExecutor.
    
    Args:
        problems: List of problem dictionaries
        cfg_bundle: Configuration bundle
        env_config: Environment configuration
        policy_agent: Shared policy agent (thread-safe)
        verifier_agent: Shared verifier agent (thread-safe)
        full_config: Full configuration dictionary
        verifier_max_workers: Max workers for verifier parallelization within each problem
        problem_max_workers: Max workers for problem-level parallelization
        test_idx: Test iteration index
    
    Returns:
        List of result dictionaries for each problem
    """
    log.info(f"===> Running {len(problems)} problems in parallel with {problem_max_workers} workers")
    
    # Prepare arguments for each problem
    problem_args = [
        (idx, problem, cfg_bundle, env_config, policy_agent, verifier_agent, full_config, verifier_max_workers)
        for idx, problem in enumerate(problems)
    ]
    
    results = []
    with ThreadPoolExecutor(max_workers=problem_max_workers) as executor:
        futures = {executor.submit(run_single_problem_wrapper, args): args[0] for args in problem_args}
        
        for future in as_completed(futures):
            problem_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result["success"]:
                    log.info(f"===> Problem {problem_idx} completed: success_rate={result['success_rate']:.4f}, elapsed={result['elapsed']:.2f}s")
                else:
                    log.error(f"===> Problem {problem_idx} failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                log.error(f"===> Problem {problem_idx} raised exception: {e}")
                results.append({
                    "problem_idx": problem_idx,
                    "success": False,
                    "error": str(e),
                    "success_rate": 0.0,
                    "task_success_cnt": {},
                    "task_total_cnt": {},
                    "elapsed": 0.0,
                    "problem_payload": None,
                })
    
    # Sort results by problem index to maintain order
    results.sort(key=lambda x: x["problem_idx"])
    return results

# ---------------------------------------------------------------

# ------ Main function for running the evolving process ---------
def run(config: Optional[str] = None, DEBUG: bool = False, PROBLEM_IDX=None):
    log_level = logging.DEBUG if DEBUG else logging.INFO
    configure_color_logging(log_level)
    cfg_bundle = load_run_configuration(config)
    run_cfg = cfg_bundle["run_cfg"]
    env_config = cfg_bundle["env_config"]
    policy_model_cfg = cfg_bundle["policy_model_cfg"]
    verifier_cfg = cfg_bundle["verifier_cfg"]
    
    policy_env_num = cfg_bundle["policy_env_num"]
    verifier_env_num = cfg_bundle["verifier_env_num"]
    env_name = cfg_bundle["env_name"]
    group_n = cfg_bundle["group_n"]
    policy_env_max_steps = cfg_bundle["policy_env_max_steps"]
    policy_env_history_length = cfg_bundle["policy_env_history_length"]
    verifier_env_max_steps = cfg_bundle["verifier_env_max_steps"]
    verifier_env_history_length = cfg_bundle["verifier_env_history_length"]
    scored_history_length = cfg_bundle["scored_history_length"]
    
    # Evolving configuration
    evolving_round = int(OmegaConf.select(env_config, "informal_math_evolving.evolving_round")) or 1
    execution_mode = str(OmegaConf.select(env_config, "informal_math_evolving.execution_mode") or "agentic")
    assert execution_mode in ["agentic", "mechanical"], f"Invalid execution_mode: {execution_mode}"
    
    # Concurrency configuration
    verifier_max_workers = int(OmegaConf.select(env_config, "informal_math_evolving.concurrency.verifier_max_workers") or 0)
    problem_max_workers = int(OmegaConf.select(env_config, "informal_math_evolving.concurrency.problem_max_workers") or 0)
    log.info(f"===> Concurrency config: verifier_max_workers={verifier_max_workers}, problem_max_workers={problem_max_workers}")
    
    enable_verify = cfg_bundle["enable_verify"]
    test_times = cfg_bundle["test_times"]
    problems = load_problems(run_cfg, DEBUG, PROBLEM_IDX)
    log.info("Using environment config: %s", env_config)

    # Compose policy role env_config with per-role overrides
    policy_env_config = copy.deepcopy(env_config)
    policy_env_config.max_steps = policy_env_max_steps
    policy_env_config.history_length = policy_env_history_length
    policy_env_config.informal_math_evolving.memory_type = str(OmegaConf.select(env_config, "informal_math_evolving.policy_env.memory_type"))
    policy_env_config.informal_math_evolving.enable_python_code = bool(OmegaConf.select(env_config, "informal_math_evolving.policy_env.enable_python_code"))
    policy_env_config.informal_math_evolving.enable_local_rag = bool(OmegaConf.select(env_config, "informal_math_evolving.policy_env.enable_local_rag") or False)

    log.info("==> Building policy env manager...")
    log.info(f"==> policy_env_config: {policy_env_config}")
    policy_env_manager = build_env(
        env_name,
        env_num=policy_env_num,
        group_n=group_n,
        env_config=policy_env_config,
    )
    # NOTE: Here we define the policy agent
    agent = Agent(policy_model_cfg)
    
    verifier_env_manager = None
    verifier_agent = None
    if enable_verify:
        log.info("===> Verifier enabled with config: %s", verifier_cfg)
        log.info("===> Verifier orchestrator enabled.")
        # Compose verifier role env_config with per-role overrides
        verifier_env_config = copy.deepcopy(env_config)
        verifier_env_config.max_steps = verifier_env_max_steps
        verifier_env_config.history_length = verifier_env_history_length
        verifier_env_config.informal_math_evolving.memory_type = str(OmegaConf.select(env_config, "informal_math_evolving.verifier_env.memory_type"))
        verifier_env_config.informal_math_evolving.enable_python_code = bool(OmegaConf.select(env_config, "informal_math_evolving.verifier_env.enable_python_code"))
        verifier_env_config.informal_math_evolving.enable_local_rag = bool(OmegaConf.select(env_config, "informal_math_evolving.verifier_env.enable_local_rag") or False)

        log.info("==> Building verifier env manager...")
        log.info(f"==> verifier_env_config: {verifier_env_config}")
        verifier_env_manager = build_env(
            env_name,
            env_num=verifier_env_num,
            group_n=1,
            env_config=verifier_env_config,
        )
        verifier_agent = Agent(verifier_cfg)
    else:
        log.info("===> Verifier orchestration disabled.")

    full_config = {k: v for k, v in OmegaConf.to_container(cfg_bundle["cfg"], resolve=True).items() if not k.endswith("_config")}
    
    runtime = {
        "full_config": full_config,
        "policy_env_num": policy_env_num,
        "evolving_round": evolving_round,
        "policy_agent": agent,
        "policy_env_manager": policy_env_manager,
        "policy_env_max_steps": policy_env_max_steps,
        "policy_env_history_length": policy_env_history_length,
        "scored_history_length": scored_history_length,
        "verifier_max_workers": verifier_max_workers,
        "verifier_configs": {
            "enabled": bool(enable_verify),
            "verifier_env_num": verifier_env_num,
            "verifier_env_manager": verifier_env_manager if enable_verify else None,
            "verifier_agent": verifier_agent if enable_verify else None,
            "verifier_cfg": verifier_cfg if enable_verify else None,
            "verifier_env_max_steps": verifier_env_max_steps if enable_verify else 0,
            "verifier_env_history_length": verifier_env_history_length if enable_verify else 0,
        },
    }

    overall_success_rates = []
    task_success_history = defaultdict(list)

    for test_idx in range(test_times):
        # Check if problem-level parallelization is enabled
        if problem_max_workers > 0 and len(problems) > 1:
            # Parallel problem execution
            log.info(f"===> Using parallel problem execution with {problem_max_workers} workers")
            results = run_problems_parallel(
                problems=problems,
                cfg_bundle=cfg_bundle,
                env_config=env_config,
                policy_agent=agent,
                verifier_agent=verifier_agent,
                full_config=full_config,
                verifier_max_workers=verifier_max_workers,
                problem_max_workers=problem_max_workers,
                test_idx=test_idx,
            )
            
            # Process results and save outputs
            for result in results:
                problem_idx = result["problem_idx"]
                current_problem = problems[problem_idx]
                
                if result["success"] and result.get("problem_payload"):
                    output_path = save_problem_outputs(
                        run_cfg=run_cfg,
                        agent_model_name=agent.model_name,
                        test_idx=test_idx,
                        problem_idx=problem_idx,
                        policy_env_max_steps=policy_env_max_steps,
                        policy_env_num=policy_env_num,
                        verifier_env_num=verifier_env_num,
                        problem_payload=result["problem_payload"],
                    )
                    log.info("===> Step outputs saved to: %s", output_path)
                
                overall_success_rates.append(result["success_rate"])
                
                for task_name, total_cnt in result["task_total_cnt"].items():
                    if total_cnt == 0:
                        continue
                    success_cnt = result["task_success_cnt"].get(task_name, 0)
                    rate = success_cnt / total_cnt
                    task_success_history.setdefault(task_name, []).append(rate)
                
                log.info("===> Problem %s overall success: %.4f, elapsed: %.2fs", 
                         problem_idx, result['success_rate'], result['elapsed'])
        else:
            # Sequential problem execution (original behavior)
            for problem_idx, current_problem in enumerate(problems):
                
                if PROBLEM_IDX is not None:
                    problem_idx = PROBLEM_IDX

                log.info("===> Using problem %s: %s...", problem_idx, current_problem.get('question', ''))
                
                # --- Run the problem ---
                result = run_problem(problem_idx, current_problem, runtime)
                # --- Save the problem outputs ---
                output_path = save_problem_outputs(
                    run_cfg=run_cfg,
                    agent_model_name=agent.model_name,
                    test_idx=test_idx,
                    problem_idx=problem_idx,
                    policy_env_max_steps=policy_env_max_steps,
                    policy_env_num=policy_env_num,
                    verifier_env_num=verifier_env_num,
                    problem_payload=result["problem_payload"],
                )
                overall_success_rates.append(result["success_rate"])

                for task_name, total_cnt in result["task_total_cnt"].items():
                    if total_cnt == 0:
                        continue
                    success_cnt = result["task_success_cnt"].get(task_name, 0)
                    rate = success_cnt / total_cnt
                    task_success_history.setdefault(task_name, []).append(rate)
                log.info("===> Step outputs saved to: %s", output_path)
                log.info("===> You could visulize the step outputs by: python utils/trajectory_visualization.py %s", output_path)
                log.info("===> Problem %s overall success: %.4f", problem_idx, result['success_rate'])
                log.info("===> Problem %s time elapsed: %.2fs", problem_idx, result['elapsed'])
                verifier_stats = result.get("verifier_stats", {})
                total_calls = verifier_stats.get("total_calls", 0)
                if total_calls:
                    accuracy = verifier_stats.get("accuracy")
                    acc_display = f"{accuracy:.3f}" if accuracy is not None else "N/A"
                    log.info("===> Verifier tool accuracy: %s over %s calls", acc_display, total_calls)

    run_summary = {
        "avg_success_rate": float(np.mean(overall_success_rates)) if overall_success_rates else 0.0,
        "task_success_history": task_success_history,
        "problems_solved": len(overall_success_rates),
    }
    log.info(
        "===> Finished run: avg success %.4f, %s problems processed.",
        run_summary['avg_success_rate'],
        run_summary['problems_solved'],
    )

if __name__ == "__main__":
    fire.Fire(run)
