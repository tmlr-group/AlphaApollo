#!/bin/bash

PORT=${PORT:-11048}
export PYTHON_BACKEND_HOST="http://localhost:${PORT}"
export SERVER_BASE_URL="http://localhost:${PORT}"

if [ ! -f "tools/deepwiki_server/deepwiki-open/api/main.py" ]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

cd tools/deepwiki_server/deepwiki-open

npm run dev