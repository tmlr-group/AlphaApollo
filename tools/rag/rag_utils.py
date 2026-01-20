"""RAG System Utilities (moved to tools/rag)
Full implementation copied from tools/rag_utils.py
"""

import os
import json
import requests
import time
import re
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =============================================================================
# Configuration Loading
# =============================================================================

_CONFIG_CACHE = None

def _load_config() -> Dict[str, Any]:
	"""Load configuration from rag_config.yaml if available."""
	global _CONFIG_CACHE
	if _CONFIG_CACHE is not None:
		return _CONFIG_CACHE
    
	# Try to find config file
	config_paths = [
		Path(__file__).parent / "rag_config.yaml",  # tools/rag/rag_config.yaml
		Path(__file__).parent.parent / "rag_config.yaml",   # tools/rag_config.yaml alternative
		Path.cwd() / "tools" / "rag_config.yaml",   # from repo root
	]
    
	for config_path in config_paths:
		if config_path.exists():
			try:
				import yaml
				with open(config_path, 'r') as f:
					_CONFIG_CACHE = yaml.safe_load(f)
					logger.debug(f"Loaded RAG config from: {config_path}")
					return _CONFIG_CACHE
			except Exception as e:
				logger.debug(f"Failed to load config from {config_path}: {e}")
    
	_CONFIG_CACHE = {}
	return _CONFIG_CACHE

def _get_config_value(keys: str, default: Any = None) -> Any:
	"""Get a nested config value using dot notation (e.g., 'ports.rag_api')."""
	config = _load_config()
	value = config
	for key in keys.split('.'):
		if isinstance(value, dict):
			value = value.get(key)
		else:
			return default
	return value if value is not None else default

# =============================================================================
# Default Configuration (from config file or fallback)
# =============================================================================

def _get_default_chat_base_url() -> str:
	port = _get_config_value('ports.vllm_chat', 10089)
	return os.environ.get("CHAT1_BASE_URL", f"http://127.0.0.1:{port}/v1")

def _get_default_chat_model() -> str:
	return os.environ.get("CHAT1_MODEL", _get_config_value('models.chat.path', 'Qwen/Qwen3-8B'))

def _get_default_chat_timeout() -> int:
	return int(os.environ.get("CHAT_TIMEOUT", _get_config_value('timeouts.chat_request', 60)))

def _get_default_rag_base_url() -> str:
	port = _get_config_value('ports.rag_api', 10086)
	return os.environ.get("RAG_RETRIEVE_BASE_URL", f"http://localhost:{port}")

# Legacy constants for backward compatibility
DEFAULT_CHAT_BASE_URL = None  # Will be resolved dynamically
DEFAULT_CHAT_MODEL = None     # Will be resolved dynamically
DEFAULT_CHAT_TIMEOUT = 60
DEFAULT_RAG_RETRIEVE_BASE_URL = None  # Will be resolved dynamically

# Debug settings
CHAT_SAVE = os.environ.get("CHAT_SAVE", "0").lower() in {"1", "true", "yes"}
CHAT_SAVE_DIR = os.environ.get("CHAT_SAVE_DIR", "tools/logs/rag_debug")

# Single-file debug accumulator (best-effort, per process)
_RAG_DEBUG_ACC = None


def openai_chat_completion(
	base_url: str,
	model: str,
	system_prompt: str,
	user_prompt: str,
	timeout: int = 60
) -> str:
	"""
	Call OpenAI-compatible chat completion API.
	"""
	url = f"{base_url.rstrip('/')}/chat/completions"
	headers = {"Content-Type": "application/json", "Authorization": "Bearer 1234"}
	payload = {
		"model": model,
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt}
		],
		"temperature": 0.2,
		"max_tokens": 16384,
		"chat_template_kwargs": {"enable_thinking": False}
	}
	resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
	resp.raise_for_status()
	data = resp.json()
	return data["choices"][0]["message"]["content"].strip()


def _clean_output(text: str) -> str:
	"""Remove thinking tags and clean up the output text."""
	if not isinstance(text, str):
		return ""
	t = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
	t = t.replace("<think>", "")
	t = t.replace("</think>", "")
	t = t.replace("\r\n", "\n")
	t = t.lstrip()
	return t.strip()


def rewrite_to_single_or_empty(
	original_query: str,
	chat_base_url: str = None,
	chat_model: str = None,
	chat_timeout: int = None
) -> str:
	"""
	Return one short question if needed; otherwise return empty string.
	"""
	base_url = chat_base_url or _get_default_chat_base_url()
	model = chat_model or _get_default_chat_model()
	timeout = chat_timeout or _get_default_chat_timeout()
    
	system_prompt = (
		"You decide whether to generalize the user's query. If it is already short, clear and general, return an empty line. "
		"Otherwise, rewrite it into ONE short, general question about the relevant functions/classes/APIs for this task type. "
		"The rewrite should: "
		"(1) Keep only: task category and the library/tool name if present. "
		"(2) Do NOT include any specific objects/structures, numbers, variable names, operators, or concrete equation forms. "
		"(3) Return a single concise question text (no preamble), or empty if the original is already short, clear, and general. "
		"Return ONLY the question text or empty."
	)
	user_prompt = f"Query:\n{original_query}"
	t0 = time.time()
    
	try:
		content = openai_chat_completion(base_url, model, system_prompt, user_prompt, timeout)
		content = _clean_output(content)
		t1 = time.time()
		ans = content.strip().strip('"').strip("'")
		ans = ans.splitlines()[0].strip() if ans else ""
        
		if CHAT_SAVE:
			global _RAG_DEBUG_ACC
			_RAG_DEBUG_ACC = {
				"params": {
					"chat1_system_prompt": system_prompt,
				},
				"raw_query": original_query,
				"chat1_output": ans,
				"timings": {
					"chat1_ms": int((t1 - t0) * 1000),
				},
				"_t_start": t0,
			}
        
		logger.debug(f"rewrite_to_single_or_empty: '{original_query[:50]}...' -> '{ans[:50]}...'")
		return ans
        
	except Exception as e:
		logger.warning(f"rewrite_to_single_or_empty failed: {e}")
		return ""


def summarize_or_empty(
	query: str,
	documents: str,
	repo_name: str = None,
	chat_base_url: str = None,
	chat_model: str = None,
	chat_timeout: int = None
) -> str:
	"""
	Return a concise summary if docs are unclear/off-topic; otherwise return empty string.
	"""
	base_url = chat_base_url or _get_default_chat_base_url()
	model = chat_model or _get_default_chat_model()
	timeout = chat_timeout or _get_default_chat_timeout()
    
	system_prompt = (
		"You decide whether to answer. Treat the provided docs as optional hints; they may be partial or off-topic. "
		"If the docs already clearly answer the query, return an empty line. "
		"Otherwise, write a concise, self-contained answer that: "
		"(1) includes fully-qualified function or class names when possible (e.g., sympy.core.function.diff); "
		"(2) includes brief, runnable usage examples; "
		"(3) NEVER mentions files, tests, sources, or 'documents', and avoids any meta commentary about what the docs do or do not contain; "
		"(4) may rely on general knowledge beyond the docs to help the user. "
		"Return ONLY the answer text (no preamble) or empty. "
	)
	user_prompt = f"Query:\n{query}\n\nDocs:\n{documents}"
	t0 = time.time()
    
	try:
		content = openai_chat_completion(base_url, model, system_prompt, user_prompt, timeout)
		content = _clean_output(content)
		t1 = time.time()
        
		if CHAT_SAVE:
			try:
				os.makedirs(CHAT_SAVE_DIR, exist_ok=True)
				ts = time.strftime("%Y%m%d_%H%M%S")
				path = os.path.join(CHAT_SAVE_DIR, f"rag_debug_{ts}.json")
                
				global _RAG_DEBUG_ACC
				acc = _RAG_DEBUG_ACC if isinstance(_RAG_DEBUG_ACC, dict) else {}
                
				params = acc.get("params", {})
				params["chat2_system_prompt"] = system_prompt
                
				timings = acc.get("timings", {})
				timings["chat2_ms"] = int((t1 - t0) * 1000)
				try:
					t_start = acc.get("_t_start", t0)
					timings["total_ms"] = int((t1 - t_start) * 1000)
				except Exception:
					pass
                
				payload = {
					"params": params,
					"repo_name": repo_name,
					"raw_query": acc.get("raw_query"),
					"chat1_output": acc.get("chat1_output"),
					"embedding_responses": documents if isinstance(documents, (list, dict)) else None,
					"context_text": None if isinstance(documents, (list, dict)) else documents,
					"chat2_output": content,
					"timings": timings,
				}
				with open(path, "w", encoding="utf-8") as f:
					json.dump(payload, f, ensure_ascii=False, indent=2)
			except Exception as e:
				logger.debug(f"Failed to save debug info: {e}")
			finally:
				_RAG_DEBUG_ACC = None
        
		logger.debug(f"summarize_or_empty: generated {len(content)} chars")
		return content
        
	except Exception as e:
		logger.warning(f"summarize_or_empty failed: {e}")
		return ""


def rag_retrieve(
	repo_url: str,
	query: str,
	top_k: int = 3,
	rag_base_url: str = None,
	timeout: int = 120
) -> Dict[str, Any]:
	"""
	Call the RAG retrieve API to get relevant documents.
	"""
	base_url = rag_base_url or _get_default_rag_base_url()
    
	payload = {
		"repo_url": repo_url,
		"query": query,
		"top_k": top_k
	}
    
	try:
		resp = requests.post(
			f"{base_url}/rag/retrieve",
			json=payload,
			timeout=timeout,
		)
		if resp.status_code != 200:
			return {"error": f"HTTP {resp.status_code}: {resp.text}", "context_text": ""}
		return resp.json()
	except Exception as e:
		return {"error": str(e), "context_text": ""}

