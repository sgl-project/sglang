# SGLang Router gRPC Examples

This directory contains examples demonstrating the SGLang router gRPC client and server integration.

## Files

- `client_demo.rs` - Rust client example showing how to use the SglangSchedulerClient
- `demo_server.py` - Python mock server implementing the SGLang scheduler gRPC protocol
- `requirements.txt` - Python dependencies for the demo server
- `run_demo.sh` - Setup script to generate protobuf files and provide usage instructions
- `README.md` - This file

## Quick Start

### Option 1: Automated Setup

From the sgl-router root directory:

```bash
bash examples/run_demo.sh
```

This script will:
1. Install Python dependencies
2. Generate Python protobuf files
3. Provide instructions for running the demo

### Option 2: Manual Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r examples/requirements.txt
   ```

2. **Generate Python protobuf files:**
   ```bash
   cd examples
   python -m grpc_tools.protoc -I ../src/proto --python_out=. --grpc_python_out=. ../src/proto/sglang_scheduler.proto
   ```

3. **Run the demo server:**
   ```bash
   python demo_server.py
   ```
   
   Options:
   - `--port 8080` - Change port (default: 50051)
   - `--verbose` - Enable verbose logging

4. **Run the Rust client (in another terminal):**
   ```bash
   cargo run --example client_demo --features grpc-client
   ```

## What the Demo Shows

### Rust Client (`client_demo.rs`)

The Rust client demonstrates:

1. **Connection Management**
   - Connecting to the gRPC server
   - Client initialization with version negotiation
   - Health checks and server capability discovery

2. **Text Generation**
   - Simple text completion with streaming responses
   - Structured JSON generation with schema constraints
   - Real-time token streaming and metrics

3. **Server Management**
   - Cache flush operations
   - Request abort functionality
   - Performance metrics collection

4. **Error Handling**
   - Connection failures
   - Invalid endpoints
   - Stream interruptions

### Python Mock Server (`demo_server.py`)

The Python server provides:

1. **Protocol Implementation**
   - Full SGLang scheduler gRPC protocol
   - Initialize, Generate, Embed, HealthCheck, Abort, FlushCache endpoints
   - Streaming response generation

2. **Realistic Behavior**
   - Mock model information (Llama-3-8B-Instruct)
   - Server capabilities and hardware specs
   - Token-by-token streaming with realistic timing
   - Performance metrics simulation

3. **Structured Generation**
   - JSON schema constraint recognition
   - Structured output generation
   - Different response patterns based on input

4. **Monitoring and Metrics**
   - Request counting
   - Performance simulation
   - Resource usage mocking
   - Cache statistics

## Usage Examples

### Basic Connection Test

Start the server:
```bash
python examples/demo_server.py
```

The client will automatically connect and perform a health check.

### Custom Server Port

Server:
```bash
python examples/demo_server.py --port 8080
```

Client (modify endpoint in `client_demo.rs`):
```rust
let endpoint = "http://127.0.0.1:8080";
```

### Verbose Logging

Enable detailed server logs:
```bash
python examples/demo_server.py --verbose
```

Enable Rust client logs:
```bash
RUST_LOG=debug cargo run --example client_demo --features grpc-client
```

## Integration with Real SGLang

To use the Rust client with a real SGLang scheduler:

1. **Start SGLang server** with gRPC enabled (see SGLang documentation)

2. **Update client endpoint** in `client_demo.rs`:
   ```rust
   let endpoint = "http://your-sglang-server:50051";
   ```

3. **Remove mock-specific code** like embedding demo that might not be available

4. **Handle real model responses** which will have different content than the mock

## Troubleshooting

### Python Import Errors

If you get import errors for the protobuf files:
```bash
# Regenerate the files
cd examples
python -m grpc_tools.protoc -I ../src/proto --python_out=. --grpc_python_out=. ../src/proto/sglang_scheduler.proto
```

### Rust Compilation Errors

Ensure you're using the grpc-client feature:
```bash
cargo run --example client_demo --features grpc-client
```

### Connection Refused

1. Check that the server is running
2. Verify the port matches between client and server
3. Check firewall settings if running across machines

### Protobuf Version Mismatch

If you get protobuf compatibility errors:
```bash
pip install --upgrade grpcio grpcio-tools
```

## Extending the Examples

### Adding New RPCs

1. **Update the .proto file** with new RPC methods
2. **Regenerate protobuf files** using the run_demo.sh script
3. **Implement server methods** in demo_server.py
4. **Add client calls** in client_demo.rs
5. **Update tests** in the main codebase

### Custom Message Types

1. **Define new messages** in sglang_scheduler.proto
2. **Implement serialization/deserialization** in both Rust and Python
3. **Add validation and error handling**
4. **Update documentation**

## Performance Notes

The mock server simulates realistic performance characteristics:

- **Token generation delay**: 50ms ± 20ms per token (temperature dependent)
- **Queue time simulation**: 5-10ms baseline
- **TTFT (Time to First Token)**: ~50ms
- **Throughput simulation**: 80-120 tokens/sec
- **Memory usage**: Realistic GPU memory simulation

These values can be adjusted in `demo_server.py` for testing different scenarios.