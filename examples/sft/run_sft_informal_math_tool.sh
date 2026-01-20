set -x

export CUDA_VISIBLE_DEVICES=0,1
nproc_per_node=2

data_source='AI-MO/NuminaMath-TIR'

python3 -m examples.data_preprocess.prepare_sft_tool \
    --data_source $data_source 

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$HOME/data/$data_source/train.parquet \
    data.val_files=$HOME/data/$data_source/test.parquet \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.max_length=2048 \
    data.truncation=right \
    optim.lr=1e-4 \
    data.micro_batch_size=4 \
    model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
    trainer.project_name=NuminaMath-sft \
    trainer.experiment_name=NuminaMath-sft-qwen-2.5-3b-instruct \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=1 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true
