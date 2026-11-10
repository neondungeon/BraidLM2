#!/usr/bin/env bash
# BraidLM launcher for Linux / macOS
set -e
cd "$(dirname "$0")"
command -v python3 >/dev/null || { echo "Python 3.10+ is required."; exit 1; }
command -v ollama  >/dev/null || { echo "Ollama is required: https://ollama.com/download"; exit 1; }
if ! curl -s http://localhost:11434 >/dev/null 2>&1; then
  echo "Starting Ollama server in background..."
  (ollama serve >/dev/null 2>&1 &)
  sleep 3
fi
python3 -m pip install -r requirements.txt --quiet
exec python3 main.py
