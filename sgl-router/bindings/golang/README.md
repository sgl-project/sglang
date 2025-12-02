# SGLang Go gRPC SDK

A high-level Go SDK for interacting with SGLang gRPC API, designed with an OpenAI-style API for familiarity and ease of use.

**Location**: `sgl-router/bindings/golang/`

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Basic Usage](#basic-usage)
  - [Streaming Usage](#streaming-usage)
- [Examples](#examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Testing](#testing)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
  - [Benchmarks](#benchmarks)
- [Documentation](#documentation)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **OpenAI-style API**: Familiar interface similar to OpenAI Go SDK
- **Streaming Support**: Real-time streaming chat completions
- **Non-streaming Support**: Simple request/response API
- **Tool Calling**: Support for function calling and tool use
- **Type-safe**: Full Go type definitions for requests and responses
- **Comprehensive Testing**: 18+ unit and integration tests
- **Thread-safe**: All public methods are safe for concurrent use
- **Well-documented**: Full API documentation with examples

## Installation

```bash
go get github.com/sglang/sglang-go-grpc-sdk
```

### Build Requirements

- Go 1.21 or later
- Rust toolchain (for building the FFI library)
- Python 3.x (for Python bindings in Rust FFI)
- Tokio runtime for async operations

## Quick Start

### Basic Usage (Non-streaming)

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/sglang/sglang-go-grpc-sdk"
)

func main() {
    // Create client
    client, err := sglang.NewClient(sglang.ClientConfig{
        Endpoint:      "grpc://localhost:20000",
        TokenizerPath: "/path/to/tokenizer",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // Create completion
    resp, err := client.CreateChatCompletion(context.Background(), sglang.ChatCompletionRequest{
        Model: "default",
        Messages: []sglang.ChatMessage{
            {Role: "user", Content: "Hello!"},
        },
        Stream: false,
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(resp.Choices[0].Message.Content)
    fmt.Printf("Usage: Prompt=%d, Completion=%d, Total=%d\n",
        resp.Usage.PromptTokens,
        resp.Usage.CompletionTokens,
        resp.Usage.TotalTokens)
}
```

### Streaming Usage

```go
package main

import (
    "context"
    "fmt"
    "io"
    "log"

    "github.com/sglang/sglang-go-grpc-sdk"
)

func main() {
    // Create client
    client, err := sglang.NewClient(sglang.ClientConfig{
        Endpoint:      "grpc://localhost:20000",
        TokenizerPath: "/path/to/tokenizer",
    })
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    // Create streaming completion
    ctx := context.Background()
    stream, err := client.CreateChatCompletionStream(ctx, sglang.ChatCompletionRequest{
        Model: "default",
        Messages: []sglang.ChatMessage{
            {Role: "user", Content: "Tell me a story"},
        },
        Stream:              true,
        MaxCompletionTokens: intPtr(500),
    })
    if err != nil {
        log.Fatal(err)
    }
    defer stream.Close()

    // Read streaming response
    for {
        chunk, err := stream.Recv()
        if err == io.EOF {
            break
        }
        if err != nil {
            log.Fatal(err)
        }

        for _, choice := range chunk.Choices {
            if choice.Delta.Content != "" {
                fmt.Print(choice.Delta.Content)
            }
        }
    }
    fmt.Println() // newline
}

// Helper functions for optional pointer fields
func intPtr(i int) *int {
    return &i
}

func float32Ptr(f float32) *float32 {
    return &f
}
```

## Examples

The SDK includes several examples in the `examples/` directory:

- **simple**: Basic non-streaming chat completion example
- **streaming**: Real-time streaming with performance metrics

### Running Examples

```bash
# Run simple example
cd bindings/golang/examples/simple
bash run.sh

# Run streaming example
cd bindings/golang/examples/streaming
bash run.sh

# Or use Makefile from bindings/golang directory
cd bindings/golang
make run-simple
make run-streaming
```

Examples automatically detect the server endpoint and tokenizer path via environment variables or defaults.

## Configuration

### Environment Variables

- `SGL_GRPC_ENDPOINT`: gRPC server endpoint (default: `grpc://localhost:20000`)
- `SGL_TOKENIZER_PATH`: Path to tokenizer directory (required)
- `CARGO_BUILD_DIR`: Rust build output directory (auto-detected if not set)

### ClientConfig

```go
type ClientConfig struct {
    // Endpoint is the gRPC endpoint URL (e.g., "grpc://localhost:20000")
    // Required field. Must include the scheme (grpc://) and port number.
    Endpoint string

    // TokenizerPath is the path to the tokenizer directory containing
    // tokenizer configuration files (e.g., tokenizer.json, vocab.json)
    // Required field.
    TokenizerPath string
}
```

## API Reference

### Client Methods

```go
type Client struct {
    // Thread-safe client for SGLang gRPC API
}

// Creates a new client with the given configuration
func NewClient(config ClientConfig) (*Client, error)

// Closes the client and releases all resources
func (c *Client) Close() error

// Creates a non-streaming chat completion
func (c *Client) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error)

// Creates a streaming chat completion
func (c *Client) CreateChatCompletionStream(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionStream, error)
```

### Request Types

- `ChatCompletionRequest`: Main request type for chat completions
  - Model, Messages, Stream, Temperature, TopP, MaxCompletionTokens, Tools, etc.
- `ChatMessage`: Individual message in a conversation
  - Role, Content
- `Tool`: Tool/function definition for function calling
  - Type, Function (name, description, parameters)

### Response Types

- `ChatCompletionResponse`: Non-streaming response
  - ID, Model, Created, Choices, Usage
- `ChatCompletionStreamResponse`: Streaming response chunk
  - Same structure as above but for incremental updates
- `Message`: Complete message with content and tool calls
- `ToolCall`: Tool call information with function and arguments
- `Usage`: Token usage statistics
  - PromptTokens, CompletionTokens, TotalTokens

## Testing

The SDK includes comprehensive testing infrastructure with both unit and integration tests.

### Unit Tests

Unit tests are located in `client_test.go` and test individual components without requiring a server.

#### Running Unit Tests

```bash
# Run all unit tests
go test ./...

# Run with verbose output
go test -v ./...

# Run specific test
go test -run TestClientConfig

# Run tests with race detector (detects concurrency issues)
go test -race ./...

# Run with coverage analysis
go test -cover ./...

# Generate detailed coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

#### Unit Test Coverage

- **Configuration validation** (`TestClientConfig`) - Validates ClientConfig requirements
- **Type structures** - Verifying all struct types work correctly
- **Response handling** - Testing response parsing and validation
- **Concurrent operations** (`TestConcurrentClientOperations`) - Thread-safety verification
- **Benchmarks** (`BenchmarkChatCompletionRequest`) - Performance measurement

**Test Files**:
- `client_test.go` - 10 unit tests covering core functionality
- Tests cover: config validation, message types, request validation, close operations, response types, streaming, tools, concurrency, and context cancellation

### Integration Tests

Integration tests require a running SGLang server and test the full client-server interaction.

#### Prerequisites

1. Start an SGLang server:

```bash
# Using Python (requires sglang package installed)
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-hf

# Or using pre-built Docker image
docker run -p 20000:20000 lmsys/sglang:latest

# Or build your own
sglang launch_server --model-path <model_path>
```

2. Set required environment variables:

```bash
# Set the gRPC endpoint (default: grpc://localhost:20000)
export SGL_GRPC_ENDPOINT=grpc://localhost:20000

# Set the tokenizer path (required)
export SGL_TOKENIZER_PATH=/path/to/tokenizer
```

#### Running Integration Tests

```bash
# Run all integration tests
go test -tags=integration ./...

# Run specific integration test
go test -tags=integration -run TestIntegrationNonStreamingCompletion

# Run with verbose output
go test -tags=integration -v ./...

# Run with race detector
go test -tags=integration -race ./...
```

#### Integration Test Coverage

**Test File**: `integration_test.go` - 4 integration tests

- `TestIntegrationNonStreamingCompletion` - Basic non-streaming request/response
- `TestIntegrationStreamingCompletion` - Streaming response handling
- `TestIntegrationConcurrentRequests` - Multiple simultaneous requests
- `TestIntegrationContextCancellation` - Context timeout and cancellation

### Benchmarks

Measure performance of SDK operations:

```bash
# Run all benchmarks
go test -bench=. -benchmem ./...

# Run specific benchmark
go test -bench=BenchmarkChatCompletionRequest -benchmem

# Run for longer duration
go test -bench=. -benchtime=10s ./...
```

Current benchmarks:
- `BenchmarkChatCompletionRequest` - Measures request creation performance

### CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
- name: Run Go tests
  run: |
    go test -race -cover ./...

- name: Run integration tests (on main branch)
  if: github.ref == 'refs/heads/main'
  env:
    SGL_GRPC_ENDPOINT: grpc://localhost:20000
    SGL_TOKENIZER_PATH: /path/to/tokenizer
  run: go test -tags=integration ./...
```

## Documentation

### Code Documentation

All public types and functions include comprehensive documentation:

1. **Package-level documentation** in `client.go` with usage examples
2. **Type documentation** for all structs with field descriptions
3. **Function documentation** with:
   - Purpose and behavior description
   - Parameter documentation with types and constraints
   - Return value documentation
   - Error cases and handling
   - Safety notes (for FFI functions)
   - Usage examples

### Key Documented Components

- `Client` - Main client with thread-safety notes
- `ClientConfig` - Configuration requirements and validation rules
- `ChatCompletionRequest` - Request structure with field descriptions
- `ChatCompletionResponse` - Response structure and usage
- `ChatCompletionStreamResponse` - Streaming response format
- `Usage` - Token usage information structure
- `Tool`, `Function`, `ToolCall` - Tool call structures

### Viewing Documentation

Generate and view HTML documentation:

```bash
# Install godoc (if not already installed)
go install golang.org/x/tools/cmd/godoc@latest

# Generate and serve documentation
godoc -http=:6060

# Visit: http://localhost:6060/pkg/github.com/sglang/sglang-go-grpc-sdk/
```

## Development

### Building

```bash
cd bindings/golang

# Build the Go bindings (compiles Rust FFI library)
make build

# Clean build
make clean && make build
```

### Code Quality

Ensure code quality before committing:

```bash
# Run Go vet (check for potential bugs)
go vet ./...

# Format code
go fmt ./...

# Run all tests with race detection
go test -race ./...
```

### Project Structure

```
bindings/golang/
├── client.go                 # Main client implementation
├── client_test.go            # Unit tests
├── integration_test.go       # Integration tests
├── README.md                 # This file
├── Makefile                  # Build automation
├── Cargo.toml               # Rust FFI dependencies
├── examples/                # Example programs
│   ├── simple/             # Non-streaming example
│   └── streaming/          # Streaming example
├── src/                    # Rust FFI source
│   ├── client.rs          # Client FFI
│   ├── stream.rs          # Stream handling
│   ├── grpc_converter.rs  # Response conversion
│   └── ...
└── internal/               # Internal packages
    └── ffi/               # FFI bindings
```

## Troubleshooting

### Connection Errors

**Error**: `connection refused` or `failed to dial`

**Solution**:
1. Ensure SGLang server is running: `python -m sglang.launch_server`
2. Check endpoint: `echo $SGL_GRPC_ENDPOINT`
3. Verify port is not blocked: `nc -zv localhost 20000`

### Tokenizer Not Found

**Error**: `tokenizer path not found` or `tokenizer configuration missing`

**Solution**:
1. Set `SGL_TOKENIZER_PATH` environment variable
2. Verify path contains required files: `ls $SGL_TOKENIZER_PATH`
3. Files should include: `tokenizer.json`, `vocab.json`, `config.json`

### Build Failures

**Error**: `library 'sglang_router_rs' not found`

**Solution**:
1. Rebuild Rust library: `cd sgl-router/bindings/golang && make build`
2. Or manually with cargo: `cd sgl-router/bindings/golang && cargo build --release`
3. Set `CARGO_BUILD_DIR` if using non-standard build location
4. Ensure Rust toolchain is installed: `rustup toolchain list`

### Tests Hanging

**Error**: Tests seem to hang indefinitely

**Solution**:
1. Use timeout for hanging tests: `timeout 30s go test ./...`
2. Run with verbose output to see which test hangs: `go test -v ./...`
3. Ensure server is responsive: `grpcurl -plaintext localhost:20000 list`

### Memory Issues

**Error**: Out of memory during tests

**Solution**:
```bash
# Run with memory limit for long-running tests
GODEBUG=madvdontneed=1 go test -timeout 5m ./...

# Monitor memory during tests
watch -n1 'ps aux | grep test'
```

## Contributing

When adding new features:

1. Add comprehensive documentation to public types/functions
2. Include usage examples for complex APIs
3. Add unit tests covering happy path and error cases
4. Add integration tests if server interaction required
5. Ensure code passes `go vet` and `go test -race`
6. Update this README if adding new features

## License

See LICENSE file for details.

---

**Need Help?**
- Check examples in `examples/` directory
- Run tests to see working code: `go test -v ./...`
- Review function documentation: `godoc` or inline comments
- Check troubleshooting section above
