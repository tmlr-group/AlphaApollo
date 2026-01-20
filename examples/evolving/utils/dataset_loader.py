import os
import sys
import numpy as np
import time
import logging
import yaml
import fire
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
from collections import defaultdict
# Ensure repository root is on PYTHONPATH so local packages can be imported
project_root = Path(__file__).parent.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

from agent_system.environments import *
from openai import OpenAI
import json
import pandas as pd

def load_tomg_bench_data(data_path: str, benchmark_type: str, subtask: str = None) -> list:
    """
    Load TOMG_BENCH data from processed parquet files.

    Args:
        data_path: Path to the processed TOMG_BENCH data directory
        benchmark_type: Benchmark type (MolOpt, MolEdit, MolCustom)
        subtask: Specific subtask to load (optional, loads all if not specified)

    Returns:
        List of problem dictionaries with extra information
    """
    problems = []

    # Convert to Path object for easier handling
    data_dir = Path(data_path)

    # Check if directory exists
    if not data_dir.exists():
        logger.error(f"Error: Data directory does not exist: {data_path}")
        return problems

    # Look for the appropriate parquet file based on benchmark type and subtask
    benchmark_lower = benchmark_type.lower()
    available_files = list(data_dir.glob("*.parquet"))
    file_path = None

    if subtask:
        # For specific subtask, look for files that start with benchmark type and contain subtask
        subtask_lower = subtask.lower()
        for file in available_files:
            if file.name.startswith(f"{benchmark_lower}_") and subtask_lower in file.name:
                file_path = file
                break

        # If not found, try more flexible matching
        if not file_path:
            for file in available_files:
                if (benchmark_lower in file.name and
                    subtask_lower in file.name and
                    file.name.endswith("_test.parquet")):
                    file_path = file
                    break
    else:
        # For general benchmark, look for files that start with benchmark type
        for file in available_files:
            if file.name.startswith(f"{benchmark_lower}_") and file.name.endswith("_test.parquet"):
                file_path = file
                break

        # If no specific match found, try to find any file for this benchmark type
        if not file_path:
            for file in available_files:
                if benchmark_lower in file.name and file.name.endswith("_test.parquet"):
                    file_path = file
                    break

    if not file_path:
        logger.warning(f"Warning: No data file found for {benchmark_type} in {data_path}")
        logger.warning(f"Available parquet files: {[f.name for f in available_files]}")
        return problems

    logger.info(f"Selected file: {file_path.name}")

    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} examples from {file_path}")

    for _, row in df.iterrows():
        # Extract information from the processed data structure
        env_kwargs = row.get('env_kwargs', {})
        metadata = row.get('metadata', {})
        problem = {
            "problem": row.get('Instruction', ''),
            'target_property': env_kwargs.get('target_property', ''),
            'task_direction': env_kwargs.get('task_direction', ''),
            'reference_molecule': env_kwargs.get('reference_molecule', ''),
            'subtask': metadata.get('subtask', ''),
            'benchmark': metadata.get('benchmark', ''),
        }
        problems.append(problem)

    return problems


def load_informal_math_data(file_path) -> list:
    problems = []
    df = pd.read_parquet(file_path)
    logger.info(f"Loaded {len(df)} examples from {file_path}")

    for _, row in df.iterrows():
        info = row.get('extra_info', {})
        assert len(info) != 0, "extra_info is empty"
        problem = {
            "question": info.get('question', ''),
            "ground_truth": info.get('ground_truth', ''),
            "gt_traj": info.get('gt_traj', ''),
            "data_source": info.get('data_source', ''),
        }
        problems.append(problem)

    return problems
