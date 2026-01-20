#!/bin/bash
# Start RAG services (moved to tools/rag)
set -e
[[ "${TRACE:-0}" == "1" ]] && set -x

# Use the parent script implementation copied here
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"
# Load configuration from YAML via Python helper
# This step is expected to export:
# RAG_API_PORT, VLLM_EMBED_PORT, VLLM_CHAT_PORT,
# CHAT_MODEL, EMBED_MODEL, MAX_MODEL_LEN, CHAT_GPU_MEMORY_UTILIZATION, EMBEDDING_GPU_MEMORY_UTILIZATION,
# CHAT_READY_TIMEOUT, EMBED_READY_TIMEOUT, RAG_API_READY_TIMEOUT,
# CUDA_VISIBLE_DEVICES
eval "$(python -m tools.rag.rag_config env)"
python -m tools.rag.rag_config embedder

export HYDRA_FULL_ERROR=1
ulimit -n 65535 2>/dev/null || true
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

log_info()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"; }
log_error() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2; }
log_warn()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" >&2; }

is_port_listening() {
    local port="$1"
    ss -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:\.]${port}$" && return 0
    lsof -iTCP -sTCP:LISTEN -P -n 2>/dev/null | grep -q ":${port} " && return 0
    return 1
}

wait_for_port() {
    local port="$1" timeout_sec="${2:-60}" name="${3:-service}" pid_check="${4:-}"
    local deadline=$(( $(date +%s) + timeout_sec ))

    log_info "Waiting up to ${timeout_sec}s for ${name} on port ${port}..."
    while true; do
        is_port_listening "${port}" && { log_info "${name} ready on port ${port}"; return 0; }
        [[ -n "${pid_check}" ]] && ! kill -0 "${pid_check}" 2>/dev/null && { log_error "${name} exited"; return 1; }
        [[ $(date +%s) -ge ${deadline} ]] && { log_error "${name} timeout on port ${port}"; return 1; }
        sleep 1
    done
}

cleanup_pid() {
    local pid="$1" name="$2"
    [[ -z "$pid" ]] && return
    kill -0 "$pid" 2>/dev/null || return

    log_info "Stopping $name (PID $pid)..."
    kill "$pid" 2>/dev/null || true

    for _ in {1..20}; do kill -0 "$pid" 2>/dev/null || { log_info "Stopped $name"; return; }; sleep 0.5; done
    kill -9 "$pid" 2>/dev/null || true
}

VLLM_CHAT_PID="" VLLM_EMBED_PID="" RAG_API_PID=""
cleanup() { cleanup_pid "${RAG_API_PID}" "rag_api"; cleanup_pid "${VLLM_EMBED_PID}" "vllm_embed"; cleanup_pid "${VLLM_CHAT_PID}" "vllm_chat"; }
trap 'cleanup; exit' INT TERM
trap cleanup EXIT

log_info "Checking service status..."

SKIP_VLLM_CHAT=false; SKIP_VLLM_EMBED=false; SKIP_RAG_API=false
is_port_listening "${VLLM_CHAT_PORT}" && { log_info "vLLM Chat already on ${VLLM_CHAT_PORT}"; SKIP_VLLM_CHAT=true; }
is_port_listening "${VLLM_EMBED_PORT}" && { log_info "vLLM Embed already on ${VLLM_EMBED_PORT}"; SKIP_VLLM_EMBED=true; }
is_port_listening "${RAG_API_PORT}" && { log_info "RAG API already on ${RAG_API_PORT}"; SKIP_RAG_API=true; }

if [[ "${SKIP_VLLM_CHAT}" == "false" ]]; then
    log_info "Starting vLLM Chat on port ${VLLM_CHAT_PORT}..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "${CHAT_MODEL}" --host 127.0.0.1 --port "${VLLM_CHAT_PORT}" \
        --max-model-len "${MAX_MODEL_LEN}" --trust-remote-code \
        --gpu-memory-utilization "${CHAT_GPU_MEMORY_UTILIZATION}" \
        > "${LOG_DIR}/vllm_chat.log" 2>&1 &
    VLLM_CHAT_PID=$!
    wait_for_port "${VLLM_CHAT_PORT}" "${CHAT_READY_TIMEOUT}" "vLLM Chat" "${VLLM_CHAT_PID}" || exit 1
fi

if [[ "${SKIP_VLLM_EMBED}" == "false" ]]; then
    log_info "Starting vLLM Embed on port ${VLLM_EMBED_PORT}..."
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "${EMBED_MODEL}" --task embed --host 127.0.0.1 --port "${VLLM_EMBED_PORT}" \
        --api-key 1234 --trust-remote-code \
        --gpu-memory-utilization "${EMBEDDING_GPU_MEMORY_UTILIZATION}" \
        > "${LOG_DIR}/vllm_embed.log" 2>&1 &
    VLLM_EMBED_PID=$!
    wait_for_port "${VLLM_EMBED_PORT}" "${EMBED_READY_TIMEOUT}" "vLLM Embed" "${VLLM_EMBED_PID}" || exit 1
fi

if [[ "${SKIP_RAG_API}" == "false" ]]; then
    log_info "Starting RAG API on port ${RAG_API_PORT}..."
    export OPENAI_API_KEY="${OPENAI_API_KEY:-1234}"
    export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_CHAT_PORT}/v1"
    nohup uvicorn tools.rag.deepwiki_server.rag_retrieve_api:app --host 0.0.0.0 --port "${RAG_API_PORT}" \
        > "${LOG_DIR}/rag_api.log" 2>&1 &
    RAG_API_PID=$!
    wait_for_port "${RAG_API_PORT}" "${RAG_API_READY_TIMEOUT}" "RAG API" "${RAG_API_PID}" || exit 1
fi

cat <<EOF

=============================================================================
RAG System Services Started!
=============================================================================
Services:
  vLLM Chat:  http://127.0.0.1:${VLLM_CHAT_PORT}  (PID: ${VLLM_CHAT_PID:-running})
  vLLM Embed: http://127.0.0.1:${VLLM_EMBED_PORT}  (PID: ${VLLM_EMBED_PID:-running})
  RAG API:    http://127.0.0.1:${RAG_API_PORT}  (PID: ${RAG_API_PID:-running})

Logs: ${LOG_DIR}/
Config: tools/rag/rag_config.yaml
Press Ctrl+C to stop...
=============================================================================

EOF

[[ "${SKIP_VLLM_CHAT}" == "true" && "${SKIP_VLLM_EMBED}" == "true" && "${SKIP_RAG_API}" == "true" ]] && exit 0
while true; do
    [[ -n "${VLLM_CHAT_PID}" ]] && ! kill -0 "${VLLM_CHAT_PID}" 2>/dev/null && { log_error "vLLM Chat exited"; exit 1; }
    [[ -n "${VLLM_EMBED_PID}" ]] && ! kill -0 "${VLLM_EMBED_PID}" 2>/dev/null && { log_error "vLLM Embed exited"; exit 1; }
    [[ -n "${RAG_API_PID}" ]] && ! kill -0 "${RAG_API_PID}" 2>/dev/null && { log_error "RAG API exited"; exit 1; }
    sleep 5
done