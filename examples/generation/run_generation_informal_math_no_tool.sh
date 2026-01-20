#!/bin/bash
set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

export CUDA_VISIBLE_DEVICES=0


PYTHON_EXEC=$(which python)


# model & dataset config
model_path=Qwen/Qwen2.5-3B-Instruct
model_name="qwen2.5-3b-instruct"

env_name="informal_math_training"

data_source="math-ai/aime24"
python3 -m examples.data_preprocess.prepare_rl_validation_data \
    --data_source $data_source \
    --splits test
data_path=~/data/$data_source/test.parquet


# sampling config
n_samples=2
temperature=0.6
top_k=20
top_p=0.95

# save config
save2json=true
json_output_path=~/data/$data_source/${model_name}_${env_name}_no_tool.json
save_path=~/data/$data_source/${model_name}_${env_name}_no_tool.parquet


$PYTHON_EXEC -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1\
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=$n_samples \
    data.batch_size=32 \
    data.return_raw_chat=True \
    data.truncation='right' \
    data.output_path=$save_path \
    data.save2json=$save2json \
    data.json_output_path=$json_output_path \
    model.path=$model_path \
    rollout.temperature=$temperature \
    rollout.top_k=$top_k \
    rollout.top_p=$top_p \
    rollout.prompt_length=2048 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.90 \
    rollout.max_num_batched_tokens=16384 \
    rollout.name=vllm \
    env.env_name=$env_name \
    env.seed=0 \
    env.max_steps=1 \
    env.history_length=1 \
    env.resources_per_worker.num_cpus=0.1 \
    env.informal_math.memory_type=simple \
    env.informal_math.log_requests=false \
    env.informal_math.python_code_timeout=30 \
    env.informal_math.enable_python_code=false \
    env.informal_math.enable_local_rag=false
