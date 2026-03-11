<h1 align="center">
<b>AlphaApollo: A System for Deep Agentic Reasoning</b>
</h1>

<p align="center">
  <a href="https://alphaapollo.org/"><img src="https://img.shields.io/badge/Project%20Website-AlphaApollo"></a>
  <a href="https://arxiv.org/abs/2510.06261"><img src="https://img.shields.io/badge/arXiv-2510.06261-b31b1b"></a>
</p>


AlphaApollo is an agentic reasoning framework that orchestrates multiple models and tools to enable iterative, verifiable, and self-evolving reasoning. It supports a broad range of paradigms, including tool-integrated reasoning, agentic post-training (e.g., multi-turn supervised fine-tuning and reinforcement learning), and agentic self-evolution. The framework offers extensible environments and toolsets for easy customization, extension, and scalable deployment of agentic reasoning workflows.


## News
- [2026.01] We are excited to release AlphaApollo, an agentic LLM reasoning system for advanced reasoning.
- [2025.10] Our technical report is released; see [here](https://arxiv.org/abs/2510.06261v1) for details.


## Installation

```bash
conda create -n alphaapollo python==3.12 -y
conda activate alphaapollo

git clone https://github.com/tmlr-group/AlphaApollo.git
cd AlphaApollo

bash installation.sh
```

## Supported features
### [Agentic reasoning](https://alphaapollo.org/multi-turn-agentic-reasoning)
- Tool-integrated reasoning rollout with seamless environment interaction
- Dynamic memory updates for multi-turn reasoning

### [Agentic learning](https://alphaapollo.org/multi-turn-agentic-learning)
- Multi-turn supervised fine-tuning (SFT)
- Reinforcement learning algorithms: GRPO, PPO, DAPO, and more

### [Agentic self-evolution](https://alphaapollo.org/multi-round-agentic-evolution)
- Multi-round, multi-model solution refinement with shared state
- Iterative improvement via feedback and executable checks

### [Built-in tools](docs/core-modules/tools.md)
- Python interpreter
- Retrieval-Augmented Generation (RAG)
  

## Quick-start recipes

Detailed quick-start commands (including script entrypoints) are documented in [quick-start.md](docs/getting-started/quick-start.md).

Note: Before using the local RAG module, please follow [RAG Service Setup](docs/core-modules/tools.md#rag-service-setup).

### Agentic reasoning
```bash
# no-tool reasoning
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=math-ai/aime24
```
```bash
# tool-integrated reasoning
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=math-ai/aime24 \
  --env.informal_math.enable_python_code=true \
  --env.informal_math.enable_local_rag=false \
  --env.max_steps=4
```

Single-question evaluation:
```bash
# Select specific dataset samples (e.g., the 0th AIME test question) and test
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.module=alphaapollo.data_preprocess.prepare_custom_data \
  --preprocess.data_source=math-ai/aime24 \
  --preprocess.splits=test \
  --preprocess.sample_indices=0 \
  --data.path=~/data/custom_data/test.parquet
```
```bash
# Directly evaluate a plain text question (not from a dataset)
python3 -m alphaapollo.workflows.test \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.module=alphaapollo.data_preprocess.prepare_single_question \
  --preprocess.question_text="What is the sum of integers from 1 to 1000?" \
  --preprocess.ground_truth="500500" \
  --data.path=~/data/single_question/test.parquet
```

### Agentic learning
```bash
# multi-turn SFT
python3 -m alphaapollo.workflows.sft \
  --model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=AI-MO/NuminaMath-TIR
```
```bash
# multi-turn RL
python3 -m alphaapollo.workflows.rl \
  --model.path=Qwen/Qwen2.5-3B-Instruct \
  --preprocess.data_source=HuggingFaceH4/MATH-500 \
  --algorithm.adv_estimator=grpo
```

### Agentic self-evolution
> Before running the self-evolution scripts, make sure to serve the corresponding number of models.
```python 
python alphaapollo/utils/ray_serve_llm.py --model_path Qwen/Qwen3-4B-Instruct-2507 --gpus "0,1" --port 8000 --model_id "qwen3_4b_inst"
```
```bash
# single-model evolution
python3 -m alphaapollo.workflows.evo \
  --preprocess.data_source=math-ai/aime24 \
  --run.dataset_name=aime24 \
  --policy_model_cfg.model_name=qwen3_4b_inst \
  --policy_model_cfg.base_url=http://localhost:8000/v1 \
  --verifier_cfg.model_name=qwen3_4b_inst \
  --verifier_cfg.base_url=http://localhost:8000/v1
```

## Code Structure

```text
+------------------------------------------------------------------+
| alphaapollo/data_preprocess                                      |
| (dataset preparation scripts)                                    |
+------------------------------------------------------------------+
                               |
                               V
+------------------------------------------------------------------+
| alphaapollo/core                                                 |
| (core code)                                                      |
|                                                                  |
|  +----------------------+              +----------------------+  |
|  | generation/          |              | tools/               |  |
|  |                      | <----------> | - python_code        |  |
|  |                      |              | - rag/               |  |
|  +----------------------+              +----------------------+  |
|              Λ                                                   |
|              |                                                   |
|              V                                                   |
|  +------------------------------------------------------------+  |
|  | environments/                                              |  |
|  | - informal_math_training/                                  |  |
|  | - informal_math_evolving/                                  |  |
|  | - memory/                                                  |  |
|  | - prompts/                                                 |  |
|  +------------------------------------------------------------+  |
+------------------------------------------------------------------+
```

### Informal Math Environment (Training): 
- Environment package in [alphaapollo/core/environments/informal_math_training/](alphaapollo/core/environments/informal_math_training/)
- Prompts in [alphaapollo/core/environments/prompts/informal_math_training.py](alphaapollo/core/environments/prompts/informal_math_training.py)

### Informal Math Environment (Evolving): 
- Environment package in [alphaapollo/core/environments/informal_math_evolving/](alphaapollo/core/environments/informal_math_evolving/)
- Prompts in [alphaapollo/core/environments/prompts/informal_math_evolving.py](alphaapollo/core/environments/prompts/informal_math_evolving.py)

### Tools (for reference)
- Python Code implementation: [alphaapollo/core/tools/python_code.py](alphaapollo/core/tools/python_code.py)
- RAG implementation: [alphaapollo/core/tools/rag/](alphaapollo/core/tools/rag/)


## Acknowledgement
AlphaApollo is built upon the open-source projects [verl](https://github.com/volcengine/verl), [verl-agent](https://github.com/langfengQ/verl-agent/tree/master), [vllm](https://github.com/vllm-project/vllm), and [sglang](https://github.com/sgl-project/sglang). We sincerely thank the contributors of these projects for their valuable work and support.

## Cite
If you find **AlphaApollo** useful in your research, please consider citing our work:

```
@article{zhou2025alphaapollo,
  title = {{AlphaApollo}: A System for Deep Agentic Reasoning},
  author = {Zhou, Zhanke and Cao, Chentao and Feng, Xiao and Li, Xuan and Li, Zongze and Lu, Xiangyu and Yao, Jiangchao and Huang, Weikai and Cheng, Tian and Zhang, Jianghangfan and Jiang, Tangyu and Xu, Linrui and Zheng, Yiming and Miranda, Brando and Liu, Tongliang and Koyejo, Sanmi and Sugiyama, Masashi and Han, Bo},
  journal = {arXiv preprint arXiv:2510.06261},
  year = {2025}
}
```

