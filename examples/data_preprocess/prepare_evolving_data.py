# Copyright 2026 TMLR Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare informal math data for training/evaluation.

This script downloads a math reasoning dataset (defaults to ``math-ai/aime24``)
from HuggingFace Hub, normalises each sample to the structure expected by
``load_informal_math_data`` and the informal math environment, and saves the
processed rows as parquet files.

This script is independent of the rollout process and serves as an independent 
data preprocessing process.

Example usage::

    python -m examples.data_preprocess.prepare_evolving_data \
        --data_source math-ai/aime24

    python -m examples.data_preprocess.prepare_evolving_data \
        --data_source math-ai/aime24 --splits train \
        --local_dir ./data/benchmark/informal_math/aime24

The script is intentionally defensive: it copes with multiple possible field
names for question/answer/solution that appear in open-source math datasets and
skips rows that miss required information.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    # print(f"===> solution_str: {solution_str}")
    try:
        return remove_boxed(last_boxed_only_string(solution_str))
    except:
        print(f"No boxed answer found, return the original solution string: {solution_str}")
        return str(solution_str)


# ---------------------------------------------------------------------------
# Defaults & logging
# ---------------------------------------------------------------------------

# SYSTEM_PROMPT = (
#     "You are a math expert. Please solve the problem carefully and report your "
#     "final answer inside \\boxed{} when appropriate."
# )
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}. You MUST use and depend on the `python_code` tool to solve the question use it to help your reasoning or help you verify your answer"

USER_PREFIX = ""
ABILITY = "math"

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

QUESTION_KEYS: List[str] = [
    "question",
    "problem",
    "prompt",
    "Problem",
    "instruction",
]
GROUND_TRUTH_KEYS: List[str] = [
    "answer",
    "solution",
    "ground_truth",
    "final_answer",
    "target",
    "boxed_answer",
    
]
SOLUTION_KEYS: List[str] = [
    "solution",
    "detailed_solution",
    "rationale",
    "chain_of_thought",
    "cot",
]


def _normalise_text(value: Any) -> str:
    """Convert a raw value to a clean string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        items = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return "\n".join(items)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _first_non_empty(example: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in example:
            value = _normalise_text(example[key])
            if value:
                return value
    return ""


def _build_prompt(question: str) -> List[Dict[str, str]]:
    content = f"{USER_PREFIX}{question}".strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _filter_metadata(example: Dict[str, Any], used_keys: Iterable[str]) -> Dict[str, Any]:
    used = set(used_keys)
    return {k: v for k, v in example.items() if k not in used}


def process_example(
    example: Dict[str, Any],
    *,
    idx: int,
    split: str,
    data_source: str,
) -> Optional[Dict[str, Any]]:
    """Convert a raw dataset example into verl's expected schema."""

    question = _first_non_empty(example, QUESTION_KEYS)
    if not question:
        LOG.warning("Skipping index %s (split=%s): missing question field", idx, split)
        return None

    ground_truth = _first_non_empty(example, GROUND_TRUTH_KEYS)
    if not ground_truth:
        LOG.warning(
            "Skipping index %s (split=%s): missing ground-truth/answer field", idx, split
        )
        return None

    ground_truth = extract_solution(ground_truth)

    gt_traj = _first_non_empty(example, SOLUTION_KEYS)

    prompt = _build_prompt(question)

    extra_info = {
        "split": split,
        "index": idx,
        "question": question,
        "ground_truth": ground_truth,
        "gt_traj": gt_traj,
        "data_source": data_source,
    }

    env_kwargs = {
        "question": question,
        "ground_truth": ground_truth,
        "gt_traj": gt_traj,
        "data_source": data_source,
    }

    metadata = _filter_metadata(
        example,
        used_keys=set(QUESTION_KEYS + GROUND_TRUTH_KEYS + SOLUTION_KEYS),
    )
    if not metadata:     # {} → None
        metadata = None

    record = {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": extra_info,
        "metadata": metadata,
        "env_kwargs": env_kwargs,
    }

    return record


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and normalise informal math datasets."
    )
    parser.add_argument(
        "--data_source",
        default="math-ai/aime24",
        help="HuggingFace dataset repository ID (default: math-ai/aime24).",
    )
    parser.add_argument(
        "--splits",
        default="train,test",
        help="Comma separated list of dataset splits to export (default: train,test).",
    )
    parser.add_argument(
        "--local_dir",
        default="~/data",
        help="Directory to store the processed parquet files.",
    )
    parser.add_argument(
        "--hdfs_dir",
        default=None,
        help="Optional HDFS directory (if provided, processed files are mirrored there).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def load_and_process_split(
    dataset_dict, split: str, *, args: argparse.Namespace, data_source: str
) -> Optional[pd.DataFrame]:
    if split not in dataset_dict:
        LOG.warning("Split '%s' not found in dataset %s", split, data_source)
        return None

    dataset = dataset_dict[split]

    records: List[Dict[str, Any]] = []
    for idx, example in enumerate(dataset):
        processed = process_example(
            example,
            idx=idx,
            split=split,
            data_source=data_source,
        )
        if processed is not None:
            records.append(processed)

    if not records:
        LOG.warning("No valid examples were processed for split '%s'", split)
        return None

    return pd.DataFrame.from_records(records)


def main():
    args = parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    hf_repo_id = args.data_source
    LOG.info(
        "Loading dataset '%s'",
        hf_repo_id,
    )

    print(f"===> hf_repo_id: {hf_repo_id}")

    if os.path.exists(hf_repo_id):
        print(f"Loading from local directory as parquet: {hf_repo_id}")
        dataset_dict = load_dataset("parquet", data_dir=hf_repo_id)
    else:
        dataset_dict = load_dataset(hf_repo_id)

    splits = [split.strip() for split in args.splits.split(",") if split.strip()]
    if not splits:
        splits = list(dataset_dict.keys())

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(os.path.join(local_dir, hf_repo_id), exist_ok=True)

    processed_files: List[str] = []
    for split in splits:
        df_split = load_and_process_split(dataset_dict, split, args=args, data_source=hf_repo_id)
        if df_split is None:
            continue

        output_file = os.path.join(local_dir, hf_repo_id, f"{split}.parquet")
        df_split.to_parquet(output_file, index=False)
        processed_files.append(output_file)
        print(f"Saved {len(df_split)} rows to {output_file}")

    if not processed_files:
        LOG.warning("No files were written. Please check dataset availability and filters.")
        return

    if args.hdfs_dir:
        try:
            makedirs(args.hdfs_dir)
            copy(src=local_dir, dst=args.hdfs_dir)
            LOG.info("Copied processed files to HDFS: %s", args.hdfs_dir)
        except Exception as exc:  # pragma: no cover
            LOG.error("Failed to copy files to HDFS: %s", exc)


if __name__ == "__main__":
    main()


