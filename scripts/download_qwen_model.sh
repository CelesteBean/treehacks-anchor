#!/bin/bash
# Download Qwen2.5-0.5B-Instruct GGUF for on-device warning generation.
# Requires: wget or curl
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/models/qwen-0.5b"
MODEL_FILE="qwen2.5-0.5b-instruct-q4_k_m.gguf"
URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/${MODEL_FILE}"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

if [ -f "$MODEL_FILE" ]; then
    echo "Model already exists: $MODEL_DIR/$MODEL_FILE"
    echo "To re-download, remove the file first."
    exit 0
fi

echo "Downloading $MODEL_FILE (~350MB)..."
if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$MODEL_FILE" "$URL"
elif command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$MODEL_FILE" "$URL"
else
    echo "Error: wget or curl required"
    exit 1
fi

echo "Done. Model saved to $MODEL_DIR/$MODEL_FILE"
