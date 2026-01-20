#!/bin/bash

# Standalone vLLM service startup script
# vLLM startup commands extracted from mcp_deepwiki_generation_with_wiki_start.sh

# Configuration parameters (can be modified as needed)
MODEL_NAME="Qwen/Qwen3-8B"
LLM_PORT=10089
EMBEDDING_PORT=10088
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
API_KEY="1234"
GPU_DEVICE_LLM=6
GPU_DEVICE_EMBEDDING=$GPU_DEVICE_LLM
GPU_MEMORY_UTILIZATION_LLM=0.65
GPU_MEMORY_UTILIZATION_EMBEDDING=0.2
ROPE_SCALING='{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}'
MAX_MODEL_LEN=65536
CHAT_TEMPLATE_PATH="scripts/config/chat_template/nothink_chat_template.j2"

# Log directory
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== Standalone vLLM Service Startup Script ==="
echo "Model: $MODEL_NAME"
echo "LLM Port: $LLM_PORT"
echo "Embedding Port: $EMBEDDING_PORT"
echo "Chat Template: $CHAT_TEMPLATE_PATH"
echo ""

# Check if chat template file exists
if [ ! -f "$CHAT_TEMPLATE_PATH" ]; then
    echo "Error: Chat template file does not exist: $CHAT_TEMPLATE_PATH"
    echo "Please ensure the file exists or modify the CHAT_TEMPLATE_PATH variable"
    exit 1
fi

# Check if ports are occupied
check_port() {
    local port=$1
    local service_name=$2
    if lsof -ti:$port >/dev/null 2>&1; then
        echo "Warning: Port $port is already occupied ($service_name)"
        # echo "Do you want to terminate the process occupying the port? (y/n)"
        # read -r response
        # if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Terminating process on port $port..."
            lsof -ti:$port | xargs kill -9
            sleep 2
        # else
        #     echo "Please manually free port $port and retry"
        #     exit 1
        # fi
    fi
}

# Check port occupancy
check_port $LLM_PORT "vLLM LLM"
check_port $EMBEDDING_PORT "vLLM Embedding"

echo "Starting vLLM services..."

# Start embedding service
echo "Starting vLLM embedding service..."
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE_EMBEDDING
python -m vllm.entrypoints.openai.api_server \
  --model $EMBEDDING_MODEL \
  --task embed \
  --port $EMBEDDING_PORT \
  --api-key $API_KEY \
  --trust-remote-code \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION_EMBEDDING > "$LOG_DIR/vllm_embedding_service.log" 2>&1 &

EMBEDDING_PID=$!
echo "Embedding service PID: $EMBEDDING_PID"

# Start LLM service
echo "Starting vLLM LLM service..."
export CUDA_VISIBLE_DEVICES=$GPU_DEVICE_LLM
vllm serve $MODEL_NAME \
  --rope-scaling "$ROPE_SCALING" \
  --max-model-len $MAX_MODEL_LEN \
  --port $LLM_PORT \
  --trust-remote-code \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION_LLM \
  --chat-template $CHAT_TEMPLATE_PATH > "$LOG_DIR/vllm_llm_service.log" 2>&1 &

LLM_PID=$!
echo "LLM service PID: $LLM_PID"

# Wait for services to start
echo "Waiting for services to start..."
sleep 120

# Check service status
echo "Checking service status..."

# Check LLM service
if curl -s http://localhost:$LLM_PORT/v1/models > /dev/null; then
    echo "✓ vLLM LLM service started successfully (port: $LLM_PORT)"
else
    echo "✗ vLLM LLM service failed to start"
    echo "Please check log file: $LOG_DIR/vllm_llm_service.log"
    exit 1
fi

# Check embedding service
if curl -s http://localhost:$EMBEDDING_PORT/v1/models > /dev/null; then
    echo "✓ vLLM embedding service started successfully (port: $EMBEDDING_PORT)"
else
    echo "✗ vLLM embedding service failed to start"
    echo "Please check log file: $LOG_DIR/vllm_embedding_service.log"
    exit 1
fi

echo ""
echo "=== vLLM Services Started Successfully ==="
echo "LLM Service: http://localhost:$LLM_PORT"
echo "Embedding Service: http://localhost:$EMBEDDING_PORT"
echo "LLM Log: $LOG_DIR/vllm_llm_service.log"
echo "Embedding Log: $LOG_DIR/vllm_embedding_service.log"
echo ""
echo "Test commands:"
echo "curl http://localhost:$LLM_PORT/v1/models"
echo "curl http://localhost:$EMBEDDING_PORT/v1/models"
echo ""
echo "Stop services:"
echo "kill $LLM_PID $EMBEDDING_PID"
echo ""

# Save PIDs to file for easy stopping later
echo "$LLM_PID $EMBEDDING_PID" > "$LOG_DIR/vllm_pids.txt"
echo "PIDs saved to: $LOG_DIR/vllm_pids.txt"