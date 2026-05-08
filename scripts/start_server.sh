#!/bin/zsh
# Clean start for Academy API server
set -e

cd "$(dirname "$0")/.."

echo "Killing any existing uvicorn..."
pkill -f uvicorn 2>/dev/null || true
sleep 1

echo "Clearing Python cache..."
find app -name '*.pyc' -delete 2>/dev/null || true

echo "Starting server on port 8000..."
.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
