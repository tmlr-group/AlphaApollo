# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Generate responses given a dataset of prompts
"""

import os
import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.device import is_cuda_available


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(
            runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)  # used for multimodal LLM, could be none

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # Read original dataset for saving results (same as reference implementation)
    data_path = config.data.path
    if data_path.endswith(".parquet"):
        original_dataset = pd.read_parquet(data_path)
    elif data_path.endswith(".jsonl") or data_path.endswith(".json"):
        original_dataset = pd.read_json(data_path, lines=True)
    else:
        raise ValueError(f"Unsupported dataset format: {data_path}")

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # create environments if needed (for multi-turn generation)
    from agent_system.environments import make_envs
    config.data.train_batch_size = config.data.batch_size
    _, envs = make_envs(config)
    
    # create trajectory collector for multi-turn generation
    from agent_system.multi_turn_rollout import TrajectoryCollector
    traj_collector = TrajectoryCollector(config=config, tokenizer=tokenizer, processor=processor)
    
    # Create dataset and dataloader
    from verl.trainer.main_ppo import create_rl_dataset
    from verl.utils.dataset.rl_dataset import collate_fn
    from torch.utils.data import SequentialSampler
    from torchdata.stateful_dataloader import StatefulDataLoader
    
    
    rl_dataset = create_rl_dataset([config.data.path], config.data, tokenizer, processor)
    sampler = SequentialSampler(data_source=rl_dataset)
    
    dataloader = StatefulDataLoader(
        dataset=rl_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.get("dataloader_num_workers", 8),
        drop_last=False,
        collate_fn=collate_fn,
        sampler=sampler,
    )
    
    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init, device_name="cuda" if is_cuda_available else "npu")
    wg.init_model()

    num_batch = len(dataloader)
    output_lst = [[] for _ in range(config.data.n_samples)]
    total_prompt_lst = [[] for _ in range(config.data.n_samples)]
    history_lst = [[] for _ in range(config.data.n_samples)]  # Store step_str for each question
    rewards_lst = [[] for _ in range(config.data.n_samples)]  # Store rewards for each question

    for batch_idx, batch_dict in enumerate(dataloader):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        
        batch: DataProto = DataProto.from_single_dict(batch_dict)
        
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
        if "multi_modal_data" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("multi_modal_data")
        if "raw_prompt" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("raw_prompt")
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("tools_kwargs")
        if "env_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.append("env_kwargs")
        
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
        )
        
        # Save env_kwargs before multi_turn_loop (which will pop it)
        # This is needed when n_samples > 1, as multi_turn_loop pops env_kwargs from gen_batch
        saved_env_kwargs = None
        if "env_kwargs" in gen_batch.non_tensor_batch:
            saved_env_kwargs = gen_batch.non_tensor_batch["env_kwargs"].copy()
        
        # set meta_info for generation
        gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.rollout.get("do_sample", True),
            "validate": False,
        }

        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")
        for n_sample in range(config.data.n_samples):
            # Restore env_kwargs before each call to multi_turn_loop
            # because multi_turn_loop will pop it from gen_batch
            if saved_env_kwargs is not None:
                gen_batch.non_tensor_batch["env_kwargs"] = saved_env_kwargs.copy()
            
            # use multi_turn_loop instead of direct generate_sequences
            gen_batch_output = traj_collector.multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=wg,
                envs=envs,
                is_train=False,
            )
            
            output_texts = []
            total_prompts = []
            
            # Group steps by traj_uid (each traj_uid corresponds to one question)
            # Process gen_batch_output to extract step_str and rewards for each question
            question_steps_dict = {}  # {traj_uid: [step_str1, step_str2, ...]}
            question_rewards_dict = {}  # {traj_uid: [rewards1, rewards2, ...]}
            traj_uid_order = []  # Maintain the order of traj_uids as they first appear
            
            for i in range(len(gen_batch_output)):
                data_item = gen_batch_output[i]
                traj_uid = data_item.non_tensor_batch['traj_uid']
                rewards = data_item.non_tensor_batch.get('rewards', [])
                
                # Track traj_uid order (first appearance)
                if traj_uid not in question_steps_dict:
                    question_steps_dict[traj_uid] = []
                    question_rewards_dict[traj_uid] = []
                    traj_uid_order.append(traj_uid)
                
                # Get step_str from responses (the generated response for this step)
                if 'input_ids' in data_item.batch:
                    input_ids = data_item.batch['input_ids']
                    step_str = tokenizer.decode(input_ids, skip_special_tokens=True)
                else:
                    step_str = ""
                
                # Add current step to the question
                question_steps_dict[traj_uid].append(step_str)
                question_rewards_dict[traj_uid].append(rewards)
                
                
            
            # Append history and rewards for each question in order
            for traj_uid in traj_uid_order:
                history_lst[n_sample].append(question_steps_dict.get(traj_uid, []))
                rewards_lst[n_sample].append(question_rewards_dict.get(traj_uid, []))


            output_lst[n_sample].extend(output_texts)
            total_prompt_lst[n_sample].extend(total_prompts)

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()
    

    
    # convert history_lst from (n_samples, n_data) to (n_data, n_samples)
    # Each element is a list of lists: [step_str1, step_str2, ...] for each question
    arr = np.array(history_lst, dtype=object)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    arr = np.transpose(arr, axes=(1, 0))
    history_lst = [
        [x if isinstance(x, list) else [x] for x in row]
        for row in arr.tolist()
    ]


    # convert rewards_lst from (n_samples, n_data) to (n_data, n_samples)
    # Each element is a list of lists: [rewards1, rewards2, ...] for each question
    arr = np.array(rewards_lst, dtype=object)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
    arr = np.transpose(arr, axes=(1, 0))
    rewards_lst = [
        [x if isinstance(x, list) else [x] for x in row]
        for row in arr.tolist()
    ]

    # add to the data frame
    original_dataset["history"] = history_lst
    original_dataset["rewards"] = rewards_lst
    
    # Calculate avg@k and pass@k metrics
    # rewards_lst shape: (n_data, n_samples), each element is [rewards1, rewards2, ...] for a question
    # Each rewards_i is the reward for step i, which may be a scalar, list, or array
    n_questions = len(rewards_lst)
    n_samples = config.data.n_samples
    
    # Calculate total reward for each sample of each question
    # rewards_lst[i][j] is the rewards list for question i, sample j: [rewards1, rewards2, ...]
    # Sum all rewards in the list to get the total reward for that sample
    sample_rewards = []  # (n_questions, n_samples) - total reward for each sample
    for question_idx in range(n_questions):
        question_sample_rewards = []
        for sample_idx in range(n_samples):
            rewards_list = rewards_lst[question_idx][sample_idx]  # [rewards1, rewards2, ...]
            # Sum all rewards in the list
            total_reward = 0.0
            if isinstance(rewards_list, (list, np.ndarray)) and len(rewards_list) > 0:
                for step_reward in rewards_list:
                    # Each step_reward may be a scalar, list, or array
                    if isinstance(step_reward, (list, np.ndarray)):
                        # If step_reward is a list/array, sum all elements
                        step_total = sum(step_reward) if len(step_reward) > 0 else 0.0
                    elif isinstance(step_reward, (int, float, np.number)):
                        # If step_reward is a scalar, use it directly
                        step_total = float(step_reward)
                    else:
                        step_total = 0.0
                    total_reward += step_total
            question_sample_rewards.append(total_reward)
        sample_rewards.append(question_sample_rewards)
    
    # Convert to numpy array for easier computation
    sample_rewards = np.array(sample_rewards)  # (n_questions, n_samples)
    
    # Calculate correctness: reward > 0 means correct
    sample_correct = (sample_rewards > 0).astype(float)  # (n_questions, n_samples)
    
    # Calculate avg@k: average accuracy across all samples
    # For each k from 1 to n_samples, calculate average accuracy using top-k samples
    avg_at_k_results = {}
    pass_at_k_results = {}
    
    for k in range(1, n_samples + 1):
        # avg@k: average accuracy using top-k samples
        # For each question, take the first k samples and calculate their average accuracy
        top_k_correct = sample_correct[:, :k]  # (n_questions, k)
        avg_accuracy = np.mean(top_k_correct)  # Average across all questions and k samples
        avg_at_k_results[f'avg@{k}'] = avg_accuracy
        
        # pass@k: at least one correct in top-k samples
        # For each question, check if at least one of the top-k samples is correct
        question_correct = np.any(top_k_correct, axis=1)  # (n_questions,)
        pass_accuracy = np.mean(question_correct)  # Average across all questions
        pass_at_k_results[f'pass@{k}'] = pass_accuracy
    
    # Print results
    print("\n" + "="*80)
    print("Evaluation Metrics:")
    print("="*80)
    print("avg@k metrics:")
    for k in range(n_samples, n_samples + 1):
        print(f"  avg@{k}: {avg_at_k_results[f'avg@{k}']:.4f}")
    print("\npass@k metrics:")
    for k in range(n_samples, n_samples + 1):
        print(f"  pass@{k}: {pass_at_k_results[f'pass@{k}']:.4f}")
    print("="*80 + "\n")
    
    # write to output file
    output_dir = os.path.dirname(config.data.output_path)
    output_path = config.data.output_path
    makedirs(output_dir, exist_ok=True)
    
    if output_path.endswith(".parquet"):
        original_dataset.to_parquet(output_path)
    elif output_path.endswith(".jsonl") or output_path.endswith(".json"):
        original_dataset.to_json(output_path, orient="records", lines=True, force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format for path: {output_path}")
    
    # Check if save2json is true and save to JSON if so
    if config.data.get("save2json", False):
        # Ensure the directory for the JSON output path exists
        json_output_dir = os.path.dirname(config.data.json_output_path)
        makedirs(json_output_dir, exist_ok=True)
        # Save the dataset to the specified JSON output path
        original_dataset.to_json(config.data.json_output_path, orient="records", lines=True, force_ascii=False)


if __name__ == "__main__":
    main()
