<h1 align="center">
<b>AlphaApollo: A System for Deep Agentic Reasoning</b>
</h1>

<!-- <p align="center">
  <img src="src/AlphaApollo_logo.png" height="150">
</p>

<h1 align="center">
  AlphaApollo: A System for Deep Agentic Reasoning
</h1> -->



AlphaApollo is an agentic reasoning framework that integrates multiple models and tools to enable iterative, verifiable, and self-evolving reasoning.

It supports a wide range of agentic reasoning paradigms, including tool-integrated reasoning, agentic post-training (multi-turn SFT and reinforcement learning), and agentic self-evolution. AlphaApollo incorporates multiple post-training algorithms such as PPO, GRPO, and DAPO, and provides dataset-backed agentic evaluation pipelines.

AlphaApollo also offers flexible and extensible agentic environments and tool-set configurations, allowing users to easily customize, extend, and scale agentic reasoning workflows.


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
### Agentic reasoning
- Tool-integrated reasoning rollout with seamless environment interaction
- Dynamic memory updates for multi-turn reasoning

### Agentic learning
- Multi-turn supervised fine-tuning (SFT)
- Reinforcement learning algorithms: GRPO, PPO, DAPO, and more.

### Agentic self-evolution
- Multi-round, multi-model solution refinement with shared state
- Iterative improvement via feedback and executable checks

### Built-in tools
- Python interpreter
- Retrieval-Augmented Generation (RAG)
- Web search
  

## Quick-start recipes
### Agentic reasoning
```bash
bash examples/generation/run_generation_informal_math_no_tool.sh # no-tool reasoning
```
```bash
bash examples/generation/run_generation_informal_math_tool.sh # tool-integrated reasaoning
```

### Agentic learning
```bash
bash examples/sft/run_sft_informal_math_no_tool.sh # vallina SFT
```
```bash
bash examples/sft/run_sft_informal_math_tool.sh # multi-turn SFT
```
```bash
bash examples/grpo/run_grpo_informal_math_no_tool.sh # vallina GRPO
```
```bash
bash examples/grpo/run_grpo_informal_math_tool.sh # multi-turn GRPO
```

### Agentic self-evolution
> Before running the self-evolution scripts, make sure to serve the corresponding number of models.
```python 
python utils/ray_serve_llm.py --model_path <model_path> --gpus <gpus> --port <port> --model_id <model_id>
# python utils/ray_serve_llm.py --model_path Qwen/Qwen3-4B-Instruct-2507 --gpus "4,5" --port 9876 --model_id "qwen3_4b_inst"
```
```bash
bash examples/evolving/run_vllm_informalmath_evolving.sh # single-model evolution
```
```bash
bash examples/evolving/run_vllm_informalmath_evolving_multi_models.sh # multi-model evolution
```

## Code Structure

### Informal Math Environment (Training): 
- Environment package in `./agent_system/environments/informal_math_training`
- Prompts in `./agent_system/environments/prompts/informal_math_training.py`

### Informal Math Environment (Evolving): 
- Environment package in `./agent_system/environments/informal_math_evolving`
- Prompts in `./agent_system/environments/prompts/informal_math_evolving.py`

### Tools (for reference)
- Python Code implementation: `./tools/python_code.py`
- Local RAG implementation: `./tools/rag`

Note: Before using the local RAG module, please follow the instructions in `tools/rag/README.md` to set up the required environment.



## Acknowledgement
AlphaApollo is built upon the open-source projects [verl](https://github.com/volcengine/verl), [verl-agent](https://github.com/langfengQ/verl-agent/tree/master), [vllm](https://github.com/vllm-project/vllm), and [sglang](https://github.com/sgl-project/sglang). We sincerely thank the contributors of these projects for their valuable work and support.

## Cite
If you find **AlphaApollo** useful in your research, please consider citing our work:

```
@article{zhou2025alphaapollo,
  title={AlphaApollo: Orchestrating Foundation Models and Professional Tools into a Self-Evolving System for Deep Agentic Reasoning},
  author={Zhou, Zhanke and Cao, Chentao and Feng, Xiao and Li, Xuan and Li, Zongze and Lu, Xiangyu and Yao, Jiangchao and Huang, Weikai and Xu, Linrui and Cheng, Tian and others},
  journal={arXiv preprint arXiv:2510.06261},
  year={2025}
}
```

