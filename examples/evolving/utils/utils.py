import copy
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import colorlog
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Ensure repository root is on PYTHONPATH so local packages can be imported
project_root = Path(__file__).parent.parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


MAX_SERIALIZED_FIELD_LEN = 800
MAX_SERIALIZED_ITEMS = 8
TOOL_RESPONSE_PATTERN = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
VERIFIER_REPORT_PATTERN = re.compile(r"<verifier_report>(.*?)</verifier_report>", re.DOTALL)
DEFAULT_CFG_PATH = Path(project_root) / "configs" / "vllm_informal_math.yaml"

LOG_COLORS_CONFIG = {
    "DEBUG": "cyan",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}


# Convert numpy types to native Python types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def configure_color_logging(log_level: int):
    """Configure root logging with colorized output."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.handlers.clear()
    if sys.stderr.isatty():
        handler = TqdmLoggingHandler()
    else:
        handler = logging.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        log_colors=LOG_COLORS_CONFIG,
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
    
    # Silence noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

def sanitize_for_serialization(value, max_len=MAX_SERIALIZED_FIELD_LEN):
    """
    Convert nested data into a JSON-friendly structure with bounded length.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if OmegaConf.is_config(value):
        return sanitize_for_serialization(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, str):
        if len(value) > max_len:
            return f"{value[:max_len]}...<{len(value) - max_len} chars truncated>"
        return value
    if isinstance(value, dict):
        sanitized = {}
        for idx, (key, val) in enumerate(value.items()):
            if idx >= MAX_SERIALIZED_ITEMS:
                sanitized["..."] = f"{len(value) - MAX_SERIALIZED_ITEMS} more entries"
                break
            sanitized[str(key)] = sanitize_for_serialization(val, max_len)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        limited = seq[:MAX_SERIALIZED_ITEMS]
        sanitized_seq = [sanitize_for_serialization(item, max_len) for item in limited]
        if len(seq) > MAX_SERIALIZED_ITEMS:
            sanitized_seq.append(f"... {len(seq) - MAX_SERIALIZED_ITEMS} more entries")
        return sanitized_seq
    if isinstance(value, np.ndarray):
        return sanitize_for_serialization(value.tolist(), max_len)
    if isinstance(value, defaultdict):
        return sanitize_for_serialization(dict(value), max_len)
    return str(value)

def snapshot_memory(memory, limit=3):
    """
    Capture a limited snapshot of the policy or verifier memory.
    """
    if memory is None or memory.batch_size <= 0:
        return []
    snapshot = []
    for env_idx in range(memory.batch_size):
        env_records = memory[env_idx]
        records = list(env_records)
        total_len = len(records)
        recent = records[-limit:] if limit > 0 else records
        formatted = []
        for rec in recent:
            if isinstance(rec, dict):
                formatted.append({k: sanitize_for_serialization(v) for k, v in rec.items()})
            else:
                formatted.append(sanitize_for_serialization(rec))
        snapshot.append({
            "env_index": env_idx,
            "size": total_len,
            "recent": formatted,
        })
    return snapshot


def parse_tool_response_payload(text: str):
    """
    Extract JSON payload embedded inside <tool_response> tags.
    """
    if not text:
        return None
    match = TOOL_RESPONSE_PATTERN.search(text)
    if not match:
        return None
    payload_text = match.group(1).strip()
    if not payload_text:
        return None
    return json.loads(payload_text)


def extract_memory_tool_payload(memory, env_idx: int, action_text: str):
    """
    Retrieve the stored observation for the specified env/action pair.
    """
    if memory is None or memory.batch_size <= env_idx:
        return None, None
    env_records = memory[env_idx]
    for rec in env_records:
        if not isinstance(rec, dict):
            continue
        if rec.get("action") != action_text:
            continue
        text_obs = rec.get("text_obs")
        payload = parse_tool_response_payload(text_obs)
        return payload, text_obs
    return None, None

def collect_tool_events(memory, infos, actions):
    """
    Collect per-step tool invocations along with parsed payloads.
    """
    events = []
    for env_idx, info in enumerate(infos):
        tool_infos = info.get("tool_infos") if isinstance(info, dict) else None
        if not tool_infos:
            continue
        for tool_entry in tool_infos:
            if not tool_entry:
                continue
            if isinstance(tool_entry, dict):
                if not tool_entry.get("tool_calling"):
                    continue
                tool_name = tool_entry["tool_name"]
                tool_input = tool_entry.get("tool_input")
            else:
                tool_name = str(tool_entry)
                tool_input = None
            payload, raw_obs = extract_memory_tool_payload(memory, env_idx, actions[env_idx])
            events.append({
                "env_index": env_idx,
                "tool_name": tool_name,
                "tool_input": tool_input,
                "raw_observation": raw_obs,
                "tool_payload": payload,
            })
    return events


def has_tag_block(text: str, tag: str) -> bool:
    return text and f"<{tag}>" in text and f"</{tag}>" in text


def update_verifier_stats(stats: dict, tool_event: dict):
    """
    Accumulate verifier tool metrics for later accuracy checks.
    """
    if tool_event.get("tool_name") != "informalmath_verify":
        return
    payload = tool_event.get("tool_payload", {})
    score = payload.get("score")
    if not isinstance(score, (int, float)):
        return
    stats["total_calls"] += 1
    stats["score_sum"] += float(score)
    stats["records"].append({
        "env_index": tool_event["env_index"],
        "score": score,
        "tool_input": tool_event.get("tool_input"),
        "raw_observation": tool_event.get("raw_observation"),
    })

# ---- Helper function for loading config ----

def _apply_base_config(cfg: DictConfig, config_path: str):
    base_ref = cfg.pop("base_config", None)
    if not base_ref:
        return cfg
    base_cfg = OmegaConf.load(str((Path(config_path).parent / base_ref).resolve())) if isinstance(base_ref, str) else OmegaConf.create(base_ref)
    return OmegaConf.merge(base_cfg, cfg)


def load_run_configuration(config_path: Optional[str]):
    config_path = config_path or str(DEFAULT_CFG_PATH)
    cfg = OmegaConf.load(config_path)
    cfg = _apply_base_config(cfg, config_path)

    run_cfg = cfg.get("run", {})
    env_cfg = cfg.get("env", {})

    env_config_raw = env_cfg["config"]
    env_config = OmegaConf.create(env_config_raw)

    # Single source of truth for verify
    enable_verify = bool(OmegaConf.select(env_config, "informal_math_evolving.enable_verify"))
    # Policy role parameters
    policy_env_max_steps = int(OmegaConf.select(env_config, "informal_math_evolving.policy_env.max_steps"))
    verifier_env_max_steps = int(OmegaConf.select(env_config, "informal_math_evolving.verifier_env.max_steps"))
    policy_env_history_length = int(OmegaConf.select(env_config, "informal_math_evolving.policy_env.history_length"))
    verifier_env_history_length = int(OmegaConf.select(env_config, "informal_math_evolving.verifier_env.history_length"))
    # BUG: Here will raise error if scored_history_length is not set in config
    scored_history_length = int(OmegaConf.select(env_config, "informal_math_evolving.nd_memory.scored_history_length"))

    # Model configs
    policy_model_cfg = copy.deepcopy(cfg["policy_model_cfg"])
    verifier_cfg = copy.deepcopy(cfg["verifier_cfg"])

    policy_env_num = int(run_cfg.get("policy_env_num", 1))
    verifier_env_num = int(run_cfg.get("verifier_env_num", 1))
    test_times = int(run_cfg.get("test_times", 1))
    env_name = env_cfg.get("name", "informal_math_evolving")
    group_n = int(env_cfg.get("group_n", 1))
    branches_cfg = cfg.get("branches", None)
    if branches_cfg is None:
        branches = []
    else:
        branches = OmegaConf.to_container(branches_cfg, resolve=True) or []

    return {
        "config_path": config_path,
        "cfg": cfg,
        "run_cfg": run_cfg,
        "env_cfg": env_cfg,
        "env_config": env_config,
        "policy_model_cfg": policy_model_cfg,
        "verifier_cfg": verifier_cfg,
        "policy_env_num": policy_env_num,
        "verifier_env_num": verifier_env_num,
        "test_times": test_times,
        "env_name": env_name,
        "group_n": group_n,
        "policy_env_max_steps": policy_env_max_steps,
        "verifier_env_max_steps": verifier_env_max_steps,
        "policy_env_history_length": policy_env_history_length,
        "verifier_env_history_length": verifier_env_history_length,
        "enable_verify": enable_verify,
        "scored_history_length": scored_history_length,
        "branches": branches,
    }


def assemble_memory_context(policy_context: str, verifier_context: str, session_history: List[str]):
    parts = []
    if policy_context and policy_context.strip():
        parts.append("Policy history:\n" + policy_context.strip())
    if verifier_context and verifier_context.strip():
        parts.append("Verifier history:\n" + verifier_context.strip())
    if session_history:
        parts.append("Current verifier session:\n" + "\n\n".join(session_history))
    return "\n\n".join(parts).strip()

def save_problem_outputs(run_cfg, agent_model_name, test_idx, problem_idx, policy_env_max_steps, policy_env_num, verifier_env_num, problem_payload):
    run_tag = run_cfg.get("tag", "default") or "default"
    outputs_root = Path(project_root) / "outputs" / run_cfg.get("dataset_name", "unknown") / str(run_tag) / agent_model_name.replace("/", "_") / f"test_{test_idx}"
    os.makedirs(outputs_root, exist_ok=True)
    output_path = outputs_root / f"problem{problem_idx}_steps{policy_env_max_steps}_policy_envs{policy_env_num}_verifier_envs{verifier_env_num}_output.json"
    serializable_outputs = convert_to_serializable(problem_payload)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_outputs, f, ensure_ascii=False, indent=2)
    return str(output_path)