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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ray Serve LLM Deployment Utility
================================================================================
ARCHIVED FOR RESEARCH GROUP USAGE
================================================================================

Description:
    This script deploys a high-throughput LLM inference service using Ray Serve 
    and vLLM. It is designed to maximize GPU utilization on multi-GPU nodes 
    (e.g., A6000, A100) by launching multiple independent replicas of the model.

Usage Examples:

    1. Basic usage (Interactive Mode):
       python utils/ray_serve_llm.py --model_path Qwen/Qwen3-4B-Instruct-2507 --gpus "4,5"

    2. Specify port and custom model ID:
       python utils/ray_serve_llm.py --model_path Qwen/Qwen3-4B-Instruct-2507 --gpus "4,5" --port 9876 --model_id "qwen3_4b_inst"

    3. View help menu:
       python utils/ray_serve_llm.py -h

Key Features:
    - Automatic GPU isolation (via CUDA_VISIBLE_DEVICES).
    - Auto-scaling configuration (1 Replica per GPU).
    - Integrated vLLM engine metrics.

Note: If you want to serve bigger model or use tensor parallel, change the `tensor_parallel_size=1` parameter.
"""

import os
import sys
import argparse
import warnings

# Suppress HuggingFace cache warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="🚀 Deploy a high-throughput LLM service with Ray Serve & vLLM.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example:\n  python ray_serve_llm.py --model_path ./model_dir --gpus '1,2'"
    )

    group = parser.add_argument_group("Required Configuration")
    group.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Absolute path to the local model directory (must contain config.json)."
    )
    group.add_argument(
        "--gpus", 
        type=str, 
        required=True,
        help="Comma-separated physical GPU IDs to use (e.g., '0' or '1,2').\n"
             "CRITICAL: Check 'nvidia-smi -L' to identify the correct IDs."
    )

    group = parser.add_argument_group("Optional Configuration")
    group.add_argument(
        "--model_id", 
        type=str, 
        default="default-model",
        help="The model name used by clients in API calls (default: 'default-model')."
    )
    group.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="The HTTP port to expose the OpenAI-compatible API (default: 8000)."
    )
    group.add_argument(
        "--context_len", 
        type=int, 
        default=32768,
        help="Maximum context length (tokens). Decrease if OOM occurs (default: 32768)."
    )

    return parser.parse_args()

def setup_environment(gpu_ids):
    """
    Sets up the environment variables BEFORE Ray initializes.
    This is crucial for preventing Ray from seeing unassigned GPUs (like A100s).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["VLLM_USE_V1"] = "1"
    # Ensure Ray uses the specified port
    os.environ["RAY_SERVE_HTTP_PORT"] = "8000" # Fallback if argument logic isn't used
    print(f"[*] Environment configured. Visible Devices: {os.environ['CUDA_VISIBLE_DEVICES']}")

def main():
    # 1. Parse Args
    args = parse_arguments()

    # 2. Setup Environment (Isolation)
    setup_environment(args.gpus)

    # 3. Import Ray (Must be done AFTER environment setup)
    try:
        import ray
        from ray import serve
        from ray.serve.llm import LLMConfig, build_openai_app
    except ImportError as e:
        print(f"[Error] Missing dependencies. {e} Please run:")
        print('       pip install "ray[serve]" vllm')
        sys.exit(1)

    # 4. Calculate Replica Count
    # We assume 1 Replica per GPU for maximum throughput via Data Parallelism
    num_replicas = len(args.gpus.split(","))
    
    print("=" * 60)
    print(f"🚀 Initializing Ray Serve Deployment")
    print(f"   - Model:    {args.model_id}")
    print(f"   - Path:     {args.model_path}")
    print(f"   - GPUs:     {args.gpus} (Count: {num_replicas})")
    print(f"   - Port:     {args.port}")
    print("=" * 60)

    # 5. Configure LLM
    llm_config = LLMConfig(
        model_loading_config=dict(
            model_id=args.model_id,
            model_source=args.model_path,
        ),
        deployment_config=dict(
            autoscaling_config=dict(
                # Fixed scaling: exactly one replica per GPU
                min_replicas=num_replicas,
                max_replicas=num_replicas,
            )
        ),
        # NOTE: "A100" is used here to satisfy Ray's strict schema validation for high-end GPUs.
        # Since we use CUDA_VISIBLE_DEVICES for physical isolation, this string label 
        # acts as a placeholder and does not affect actual hardware execution on A6000s.
        accelerator_type="A100", 
        
        # Enable metrics for Grafana/Dashboard
        log_engine_metrics=True,

        engine_kwargs=dict(
            tensor_parallel_size=1,     # 1 GPU per Replica
            max_model_len=args.context_len, 
            gpu_memory_utilization=0.9, # Reserve 90% VRAM for KV Cache
            trust_remote_code=True,
        ),
    )

    # 6. Build Application
    app = build_openai_app({"llm_configs": [llm_config]})

    # 7. Start Service
    # We use explicit port binding here
    serve.start(http_options={"port": args.port})
    
    print(f"[*] Service is starting... (Check http://localhost:{args.port}/v1/models)")
    print(f"[*] Ray Dashboard is available at http://localhost:8265")
    print(f"[*] Press Ctrl+C to stop the server.")
    
    serve.run(app, blocking=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Stopping Ray Serve...")
        import ray
        ray.shutdown()
        sys.exit(0)
