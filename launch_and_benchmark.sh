#!/bin/bash

set -e

# Default values
DEFAULT_MODEL="Qwen/Qwen3-30B-A3B"
SERVER_PORT=30000
SERVER_HOST="127.0.0.1"

# Parse command line arguments
MODEL_PATH=${1:-$DEFAULT_MODEL}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Usage: $0 [MODEL_PATH] [PORT]"
    echo ""
    echo "Arguments:"
    echo "  MODEL_PATH    HuggingFace model ID or local path (default: $DEFAULT_MODEL)"
    echo "  PORT          Server port (default: $SERVER_PORT)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default model"
    echo "  $0 Qwen/Qwen2.5-8B-Instruct         # Benchmark 8B model"
    echo "  $0 meta-llama/Llama-3.1-8B-Instruct # Benchmark Llama"
    echo "  $0 Qwen/Qwen3-30B-A3B 30001         # Custom port"
    exit 0
fi

# Override port if provided
if [ ! -z "$2" ]; then
    SERVER_PORT="$2"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SGLang Model Benchmark Suite${NC}"
echo -e "${GREEN}========================================${NC}"


cleanup() {
    echo -e "\n${YELLOW}Cleaning up...${NC}"
    if [ ! -z "$SERVER_PID" ]; then
        echo "Stopping SGLang server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

trap cleanup EXIT

echo -e "${YELLOW}Checking model availability...${NC}"
python3 -c "
from transformers import AutoTokenizer
try:
    tokenizer = AutoTokenizer.from_pretrained('$MODEL_PATH')
    print('Model tokenizer accessible')
except Exception as e:
    print(f'Error accessing model: {e}')
    exit(1)
"

if [ $? -ne 0 ]; then
    echo -e "${RED}Model not accessible. Please ensure the model is available.${NC}"
    exit 1
fi

echo -e "${YELLOW}Starting SGLang server...${NC}"

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$SERVER_PORT" \
    --host "$SERVER_HOST" &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"

echo -e "${YELLOW}Waiting for server to be ready...${NC}"
max_attempts=60
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s "http://$SERVER_HOST:$SERVER_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}Server is ready!${NC}"
        break
    fi
    
    attempt=$((attempt + 1))
    echo "Attempt $attempt/$max_attempts - waiting for server..."
    sleep 5
done

if [ $attempt -eq $max_attempts ]; then
    echo -e "${RED}Server failed to start within expected time${NC}"
    exit 1
fi

echo -e "\n${GREEN}Starting benchmarks...${NC}"

python3 benchmark_qwen3_30b_a3b.py \
    --model-path "$MODEL_PATH" \
    --host "$SERVER_HOST" \
    --port "$SERVER_PORT" \
    --check-server

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Benchmark Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Results saved automatically with model-specific directory name"