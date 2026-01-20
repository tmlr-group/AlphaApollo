#!/bin/bash

# DeepWiki Service Startup Script
#
# This script starts the DeepWiki API service with proper configuration
# and environment setup. It handles dependency installation, port checking,
# and service health verification.
#
# Features:
# - Automatic environment variable setup
# - Port conflict detection and resolution
# - Service health check verification
# - Dependency management
# - Configuration file validation
#
# Usage: ./tools/deepwiki_server/start_deepwiki.sh
#
# Prerequisites:
# - Python 3.x installed
# - Required API keys configured in .env file
# - DeepWiki dependencies available

echo "Starting DeepWiki service..."

echo "Inject environment variables..."
echo "Use local vLLM service..."
echo "OPENAI_BASE_URL=http://localhost:1234/v1/" >> tools/deepwiki_server/deepwiki-open/.env
echo "OPENAI_API_KEY=1234" >> tools/deepwiki_server/deepwiki-open/.env
echo "Overwrite generator.json..."
printf '%s\n' '{
  "default_provider": "openai",
  "providers": {
    "openai": {
      "default_model": "Qwen/Qwen3-4B",
      "supportsCustomModel": true,
      "models": {
        "Qwen/Qwen3-4B": {
          "temperature": 0.7,
          "top_p": 0.8
        }
      }
    }
  }
}' > ./tools/deepwiki_server/deepwiki-open/api/config/generator.json
echo "Overwrite embedder.json..."
printf '%s\n' '{
  "embedder": {
    "client_class": "OpenAIClient",
    "initialize_kwargs": {
      "base_url": "http://localhost:1235/v1/",
      "api_key": "1234"
    },
    "batch_size": 64,
    "model_kwargs": {
      "model": "Qwen/Qwen3-Embedding-0.6B"
    }
  },
  "retriever": {
    "top_k": 20
  },
  "text_splitter": {
    "split_by": "word",
    "chunk_size": 350,
    "chunk_overlap": 100
  }
}' > ./tools/deepwiki_server/deepwiki-open/api/config/embedder.json

export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

# Check if running in correct directory
if [ ! -f "tools/deepwiki_server/deepwiki-open/api/main.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Set environment variables
if [ -f "tools/deepwiki_server/deepwiki-open/.env" ]; then
    source tools/deepwiki_server/deepwiki-open/.env
fi
export PORT=11048
export DEEPWIKI_BASE_URL="http://localhost:${PORT}"

# Start DeepWiki API service
echo "Starting DeepWiki API service..."
cd tools/deepwiki_server/deepwiki-open

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found, please ensure necessary API keys are set"
    echo "Please create .env file and add the following content:"
    echo "PORT=11048 (optional, default: 11048)"
    echo "GOOGLE_API_KEY=your_google_api_key"
    echo "OPENAI_API_KEY=your_openai_api_key"
    echo "OPENROUTER_API_KEY=your_openrouter_api_key (optional)"
    echo "OLLAMA_HOST=your_ollama_host (optional)"
    echo "AZURE_OPENAI_API_KEY=your_azure_openai_api_key (optional)"
    echo "AZURE_OPENAI_ENDPOINT=your_azure_openai_endpoint (optional)"
    echo "AZURE_OPENAI_VERSION=your_azure_openai_version (optional)"
fi

# Install dependencies
# echo "Installing Python dependencies..."
# pip install -r api/requirements.txt

# Start API service
echo "Starting API service on port ${PORT}..."
# Check if port is occupied
if lsof -Pi :${PORT} -sTCP:LISTEN -t >/dev/null ; then
    echo "Service already running"
    echo "Killing existing process"
    kill -9 $(lsof -t -i:${PORT})
fi
python -m api.main > ../../../logs/deepwiki_api_service.log 2>&1 &

# Wait for service to start
sleep 15

# Check if service started successfully
if curl -s http://localhost:${PORT}/health > /dev/null; then
    echo "DeepWiki API service started successfully!"
    echo "API address: http://localhost:${PORT}"
    echo "Health check: http://localhost:${PORT}/health"
else
    echo "Warning: DeepWiki API service may not have started properly"
fi

echo "DeepWiki service startup completed!" 

echo "Log file: logs/deepwiki_api_service.log"

echo "Use tail -f logs/deepwiki_api_service.log to see the logs"