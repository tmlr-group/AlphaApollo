#!/bin/bash
# Stop RAG services (moved to tools/rag)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/rag_config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

if [[ -f "${CONFIG_FILE}" ]]; then
    RAG_API_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_FILE}')); print(c.get('ports',{}).get('rag_api', 10086))" 2>/dev/null || echo "10086")
    VLLM_EMBED_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_FILE}')); print(c.get('ports',{}).get('vllm_embed', 10088))" 2>/dev/null || echo "10088")
    VLLM_CHAT_PORT=$(python3 -c "import yaml; c=yaml.safe_load(open('${CONFIG_FILE}')); print(c.get('ports',{}).get('vllm_chat', 10089))" 2>/dev/null || echo "10089")
else
    RAG_API_PORT=${RAG_API_PORT:-10086}
    VLLM_EMBED_PORT=${VLLM_EMBED_PORT:-10088}
    VLLM_CHAT_PORT=${VLLM_CHAT_PORT:-10089}
fi

log_info() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"; }
log_warn() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" >&2; }

kill_by_port() {
    local port="$1"
    local name="$2"
    local pid=""
    if command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -ti TCP:${port} -sTCP:LISTEN 2>/dev/null || true)
    fi
    if [[ -z "$pid" ]] && command -v fuser >/dev/null 2>&1; then
        pid=$(fuser ${port}/tcp 2>/dev/null || true)
    fi
    if [[ -z "$pid" ]]; then
        log_info "No process found on port ${port} (${name})"
        return 0
    fi
    log_info "Stopping ${name} (PID: ${pid}) on port ${port}..."
    kill "$pid" 2>/dev/null || true
    for i in {1..10}; do
        if ! kill -0 "$pid" 2>/dev/null; then
            log_info "Stopped ${name}"
            return 0
        fi
        sleep 0.5
    done
    log_warn "Force killing ${name} (PID: ${pid})..."
    kill -9 "$pid" 2>/dev/null || true
    log_info "Force stopped ${name}"
}

log_info "Stopping RAG System Services..."
kill_by_port "${RAG_API_PORT}" "RAG API"
kill_by_port "${VLLM_EMBED_PORT}" "vLLM Embed"
kill_by_port "${VLLM_CHAT_PORT}" "vLLM Chat"
log_info "All RAG System services stopped"
