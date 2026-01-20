set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=XFORMERS

export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

data_source='DigitalLearningGmbH/MATH-lighteval'
test_data_source='HuggingFaceH4/MATH-500'

train_batch_size=8
val_batch_size=128
group_size=8

export CUDA_VISIBLE_DEVICES=0,1
export HF_DATASETS_CACHE=$HOME/.cache/datasets

# NOTE: this is the model path in my environment, you need to change it to your own model path.
model_path=Qwen/Qwen2.5-3B-Instruct
project_name='verl_agent_informalmath'
experiment_name='grpo_qwen2.5_3b'

# We only use data preparation to indicate the modality and the data size.
python3 -m examples.data_preprocess.prepare_rl_training_data \
    --data_source $data_source 

python3 -m examples.data_preprocess.prepare_rl_validation_data \
    --data_source $test_data_source \
    --splits test

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/$data_source/train.parquet \
    data.val_files=$HOME/data/$test_data_source/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='right' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=informal_math_training \
    env.seed=0 \
    env.max_steps=4 \
    env.history_length=4 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    env.informal_math.memory_type=simple \
    env.informal_math.log_requests=false \
    env.informal_math.python_code_timeout=30 \
    env.informal_math.enable_python_code=true \
    env.informal_math.enable_local_rag=false \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False $@
