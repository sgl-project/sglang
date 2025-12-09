# SGLang Go gRPC SDK

A high-level Go SDK for interacting with SGLang gRPC API, designed with an OpenAI-style API for familiarity and ease of use.

**Location**: `sgl-model-gateway/bindings/golang/`

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

### Sync Dependencies

```bash
cd sgl-model-gateway/bindings/golang
go mod tidy
```

### Build Requirements

- Go 1.21+, Rust toolchain, Python 3.x

## Quick Start

### Benchmark

Run the OpenAI-compatible server and benchmark:

```bash
# Set environment variables
export SGL_TOKENIZER_PATH="/Users/yangyanbo/tokenizer"
export SGL_GRPC_ENDPOINT="grpc://10.109.185.20:8001"

# Run server
cd examples/oai_server
bash run.sh

# Run E2E benchmark
cd ../..
make e2e E2E_MODEL=/work/models/qwencoder-3b E2E_TOKENIZER=/Users/yangyanbo/tokenizer E2E_INPUT_LEN=1024 E2E_OUTPUT_LEN=512
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

- Configuration validation, type structures, response handling, concurrent operations, and benchmarks
- `client_test.go` - 10 unit tests covering core functionality

### Integration Tests

Integration tests require a running SGLang server and test the full client-server interaction.

#### Prerequisites

1. Start SGLang server: `python -m sglang.launch_server --model-path <model_path>`
2. Set environment variables:
   ```bash
   export SGL_GRPC_ENDPOINT=grpc://localhost:20000
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

```bash
go test -bench=. -benchmem ./...
```

## Documentation

All public types and functions include comprehensive documentation with usage examples.

### Key Documented Components

- `Client` - Main client with thread-safety notes
- `ClientConfig` - Configuration requirements and validation rules
- `ChatCompletionRequest` - Request structure with field descriptions
- `ChatCompletionResponse` - Response structure and usage
- `ChatCompletionStreamResponse` - Streaming response format
- `Usage` - Token usage information structure
- `Tool`, `Function`, `ToolCall` - Tool call structures

### Viewing Documentation

```bash
godoc -http=:6060
# Visit: http://localhost:6060/pkg/github.com/sglang/sglang-go-grpc-sdk/
```

## Development

```bash
cd bindings/golang
make build          # Build Go bindings
go vet ./...        # Check code quality
go fmt ./...        # Format code
go test -race ./... # Run tests
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

### Missing Dependencies

Run `go mod tidy` to sync dependencies.

### Connection Errors

Ensure SGLang server is running and check `SGL_GRPC_ENDPOINT`.

### Tokenizer Not Found

Set `SGL_TOKENIZER_PATH` environment variable.
2. Verify path contains required files: `ls $SGL_TOKENIZER_PATH`
3. Files should include: `tokenizer.json`, `vocab.json`, `config.json`

### Build Failures

**Error**: `library 'sgl_model_gateway_go' not found`

**Solution**:
1. Rebuild Rust library: `cd sgl-model-gateway/bindings/golang && make build`
2. Or manually with cargo: `cd sgl-model-gateway/bindings/golang && cargo build --release`
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
