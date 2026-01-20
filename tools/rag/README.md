# RAG System Guide

Local RAG is a locally deployed Retrieval-Augmented Generation system designed for querying GitHub repositories.

## Quick Start

### 1. Environment Setup

Install the DeepWiki dependencies:

```bash
pip install -r tools/rag/requirements.txt

```

### 2. Prepare DeepWiki Cache

```bash
mkdir -p ~/.adalflow
```

If you have pre-existing cache data, extract it to `~/.adalflow`:

deepwiki_data.zip link: https://drive.google.com/file/d/1QnORK_1Qcjstm4_DnrbIUX_rPd-AiGEx/view?usp=sharing

or use gdown to download the data:

```bash
cd ~/.adalflow
# pip install gdown
gdown https://drive.google.com/uc?id=1QnORK_1Qcjstm4_DnrbIUX_rPd-AiGEx
```

```bash
unzip deepwiki_data.zip -d ~/.adalflow
# Verify: Ensure you see 'databases' and 'wikicache' directories
ls ~/.adalflow

```

### 3. Edit Configuration

Edit `tools/rag/rag_config.yaml`. Key settings include:

1. Chat model path
2. Embedding model path
3. GPU device ID

```yaml
# Service Ports
ports:
  rag_api: 10086      # RAG API Port
  vllm_embed: 10088   # Embedding Model Port
  vllm_chat: 10089    # Chat Model Port

# Model Configuration
models:
  chat:
    path: "/path/to/Qwen3-8B"  # Chat Model Path
    max_model_len: 32768
    gpu_memory_utilization: 0.6
  embedding:
    path: "/path/to/Qwen3-Embedding-8B"  # Embedding Model Path

# GPU Configuration
gpu:
  cuda_visible_devices: "5"  # GPU Device ID

```

### 4. Start Services

```bash
bash tools/rag/start_rag_server.sh

```

The startup script launches three services in sequence:

1. vLLM Chat (Port 10089) - Query Rewriting and Summary Generation
2. vLLM Embed (Port 10088) - Embedding Generation
3. RAG API (Port 10086) - Document Retrieval Service

### 5. Test the RAG Tool

Run the test script:

```bash
python -m tools.rag.rag_test

```

### 6. Stop Services

```bash
bash tools/rag/stop_rag_server.sh

```

## Configuration Reference

Key configuration parameters:

| Parameter | Description | Default |
| --- | --- | --- |
| `ports.rag_api` | RAG API Port | 10086 |
| `ports.vllm_embed` | Embedding Service Port | 10088 |
| `ports.vllm_chat` | Chat Service Port | 10089 |
| `models.chat.path` | Chat Model Path | Qwen/Qwen3-8B |
| `models.embedding.path` | Embedding Model Path | Qwen/Qwen3-Embedding-8B |
| `gpu.cuda_visible_devices` | GPU Device ID | 5 |

## View Logs

```bash
# View service logs
tail -f tools/rag/logs/vllm_chat.log
tail -f tools/rag/logs/vllm_embed.log
tail -f tools/rag/logs/rag_api.log

```
