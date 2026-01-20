"""RAG System Configuration Loader - Unified Configuration Management"""
import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

_CONFIG_FILE = Path(__file__).parent / "rag_config.yaml"
_EMBEDDER_JSON = Path(__file__).parent / "deepwiki_server/deepwiki-open/api/config/embedder.json"
_config: Dict[str, Any] = {}


def load_config() -> Dict[str, Any]:
    """Load configuration file"""
    global _config
    if not _config:
        with open(_CONFIG_FILE, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def get(key: str, default: Any = None) -> Any:
    """Get config value, supports dot-separated path, e.g. 'ports.rag_api'"""
    cfg = load_config()
    for k in key.split("."):
        if isinstance(cfg, dict):
            cfg = cfg.get(k)
        else:
            return default
        if cfg is None:
            return default
    return cfg


def generate_embedder_json() -> None:
    """Generate embedder.json based on unified configuration"""
    cfg = load_config()
    embed_port = cfg["ports"]["vllm_embed"]
    embed_model = cfg["models"]["embedding"]["path"]
    api_key = cfg["embedder"]["api_key"]
    batch_size = cfg["embedder"]["batch_size"]
    retriever = cfg["retriever"]
    text_splitter = cfg["text_splitter"]

    embedder_config = {
        "embedder": {
            "client_class": "OpenAIClient",
            "initialize_kwargs": {
                "base_url": f"http://localhost:{embed_port}/v1/",
                "api_key": api_key,
            },
            "batch_size": batch_size,
            "model_kwargs": {"model": embed_model},
        },
        "retriever": retriever,
        "text_splitter": text_splitter,
    }

    _EMBEDDER_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(_EMBEDDER_JSON, "w") as f:
        json.dump(embedder_config, f, indent=2)


def export_env_vars() -> Dict[str, str]:
    """Export as environment variable format (for shell scripts)"""
    cfg = load_config()
    return {
        "RAG_API_PORT": str(cfg["ports"]["rag_api"]),
        "VLLM_EMBED_PORT": str(cfg["ports"]["vllm_embed"]),
        "VLLM_CHAT_PORT": str(cfg["ports"]["vllm_chat"]),
        "CHAT_MODEL": cfg["models"]["chat"]["path"],
        "EMBED_MODEL": cfg["models"]["embedding"]["path"],
        "MAX_MODEL_LEN": str(cfg["models"]["chat"]["max_model_len"]),
        "CHAT_GPU_MEMORY_UTILIZATION": str(cfg["models"]["chat"]["gpu_memory_utilization"]),
        "EMBEDDING_GPU_MEMORY_UTILIZATION": str(cfg["models"]["embedding"]["gpu_memory_utilization"]),
        "CUDA_VISIBLE_DEVICES": cfg["gpu"]["cuda_visible_devices"],
        "CHAT_READY_TIMEOUT": str(cfg["timeouts"]["chat_ready"]),
        "EMBED_READY_TIMEOUT": str(cfg["timeouts"]["embed_ready"]),
        "RAG_API_READY_TIMEOUT": str(cfg["timeouts"]["rag_api_ready"]),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "env":
        # Output environment variables for shell source
        for k, v in export_env_vars().items():
            print(f'export {k}="{v}"')
    elif len(sys.argv) > 1 and sys.argv[1] == "embedder":
        generate_embedder_json()
        print(f"Generated {_EMBEDDER_JSON}")
    else:
        print("Usage: python rag_config.py [env|embedder]")
