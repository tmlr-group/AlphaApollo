export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1
export HF_ENDPOINT=https://hf-mirror.com

data_source="math-ai/aime24"

python3 -m examples.data_preprocess.prepare_evolving_data --data_source $data_source --local_dir ./data

python3 -m examples.evolving.evolving_main --config agent_system/environments/informal_math_evolving/configs/vllm_informal_math.yaml
