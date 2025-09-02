#!/bin/bash
set -e

# Demo script for testing SGLang gRPC integration

echo "SGLang Router gRPC Demo"
echo "======================="

# Check if we're in the right directory
if [ ! -f "src/proto/sglang_scheduler.proto" ]; then
    echo "Error: Please run this script from the sgl-router root directory"
    echo "Usage: bash examples/run_demo.sh"
    exit 1
fi

# Check Python dependencies
echo "Checking Python dependencies..."
if ! python3 -c "import grpc" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip3 install -r examples/requirements.txt
fi

# Generate Python protobuf files
echo "Generating Python protobuf files..."
cd examples

# Find the Python version that has grpcio-tools installed
PYTHON_CMD=""
for py_cmd in python3.11 python3.12 python3.10 python3 python; do
    if command -v "$py_cmd" >/dev/null 2>&1; then
        if "$py_cmd" -c "import grpc_tools.protoc" 2>/dev/null; then
            PYTHON_CMD="$py_cmd"
            echo "Using Python command: $PYTHON_CMD"
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "Error: Could not find Python with grpc_tools installed"
    echo "Please install dependencies with: pip3 install grpcio grpcio-tools"
    exit 1
fi

$PYTHON_CMD -m grpc_tools.protoc \
    -I ../src/proto \
    --python_out=. \
    --grpc_python_out=. \
    ../src/proto/sglang_scheduler.proto

echo "Generated protobuf files:"
ls -la *pb2*.py

# Make the server executable
chmod +x demo_server.py

echo ""
echo "Setup complete! Now you can run the demo:"
echo ""
echo "1. Start the Python mock server (in one terminal):"
echo "   cd examples && $PYTHON_CMD demo_server.py"
echo ""
echo "2. Run the Rust client (in another terminal):"
echo "   cargo run --example client_demo --features grpc-client"
echo ""
echo "The server will run on localhost:30000 by default."
echo "Use --port to change the port: $PYTHON_CMD demo_server.py --port 8080"