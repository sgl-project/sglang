#!/bin/bash

# Streaming example runner
# Usage: ./run.sh [tokenizer_path] [endpoint]

# Set library path for Rust FFI library
# The library should be in ./lib directory (created by 'make lib')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIB_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)/lib"

# Check if lib directory exists
if [ ! -d "$LIB_DIR" ]; then
    echo "Error: Library directory not found at $LIB_DIR"
    echo "Please run 'make lib' first to build and export the library"
    exit 1
fi

# Get Python LDFLAGS (needed for Rust FFI that depends on Python)
PYTHON_LDFLAGS=$(python3-config --ldflags --embed 2>/dev/null || python3-config --ldflags 2>/dev/null || echo "")

# Set CGO_LDFLAGS to link with the Rust library
export CGO_LDFLAGS="-L${LIB_DIR} -lsglang_router_rs ${PYTHON_LDFLAGS} -ldl"

# macOS uses DYLD_LIBRARY_PATH, Linux uses LD_LIBRARY_PATH
if [[ "$OSTYPE" == "darwin"* ]]; then
    export DYLD_LIBRARY_PATH="${LIB_DIR}:${DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${LIB_DIR}:${LD_LIBRARY_PATH}"
fi

# Default configuration (can be overridden by environment variables or command line arguments)
# Tokenizer path: ../tokenizer (relative to this script)
DEFAULT_TOKENIZER_PATH="${SGL_TOKENIZER_PATH:-../tokenizer}"
DEFAULT_ENDPOINT="${SGL_GRPC_ENDPOINT:-grpc://localhost:20000}"

TOKENIZER_PATH="${1:-${DEFAULT_TOKENIZER_PATH}}"
ENDPOINT="${2:-${DEFAULT_ENDPOINT}}"

echo "Running streaming example..."
echo "Library path: ${LIB_DIR}"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Endpoint: $ENDPOINT"
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")"
SGL_TOKENIZER_PATH="$TOKENIZER_PATH" SGL_GRPC_ENDPOINT="$ENDPOINT" go run main.go
