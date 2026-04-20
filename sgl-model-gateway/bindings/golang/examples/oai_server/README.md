# Go SGLang Router - OpenAI Compatible API Server

Go SGLang Router is a high-performance OpenAI-compatible API server that communicates with the SGLang backend via gRPC and performs efficient preprocessing and postprocessing through Rust FFI.

## Features

- ✅ **OpenAI API Compatible**: Fully compatible with OpenAI Chat Completions API
- ✅ **High Performance**: Low latency and high throughput using gRPC and Rust FFI
- ✅ **Streaming Support**: Server-Sent Events (SSE) streaming responses
- ✅ **Thread-Safe**: Pre-created tokenizer handle, lock-free concurrency
- ✅ **Graceful Shutdown**: Context cancellation mechanism to avoid resource leaks and panics
- ✅ **Configurable**: Supports configuring channel buffer sizes and timeout durations

## Architecture Overview

**Important Note**: gRPC mode **still calls FFI**, which is used for:
- **Preprocessing**: chat_template and tokenization (request phase)
- **Postprocessing**: token decoding and tool parsing (response phase)

gRPC is only used for communication with the SGLang backend, while input/output processing completely relies on Rust FFI.

```
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP Client                               │
│                    (OpenAI API Format)                           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastHTTP Server                               │
│              handlers/chat.go:HandleChatCompletion               │
│              - Parse request JSON                                │
│              - SetBodyStreamWriter (SSE)                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              SGLang Client (client.go)                           │
│         CreateChatCompletionStream(ctx, req)                      │
│         - Wraps gRPC client                                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│          gRPC Client (internal/grpc/client_grpc.go)              │
│         CreateChatCompletionStream(ctx, reqJSON)                 │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 1: FFI Preprocess (Rust FFI)                       │  │
│  │  - ffi.PreprocessChatRequestWithTokenizer()              │  │
│  │  - chat_template application                              │  │
│  │  - tokenization                                           │  │
│  │  - tool constraints generation                            │  │
│  │  Returns: PromptText, TokenIDs, ToolConstraintsJSON,     │  │
│  │           PromptTokens                                   │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 2: Build gRPC Request                              │  │
│  │  - Parse request JSON (model, temperature, etc.)        │  │
│  │  - Create proto.GenerateRequest                         │  │
│  │  - Set TokenizedInput (PromptText, TokenIDs)            │  │
│  │  - Set SamplingParams (temperature, top_p, top_k, etc.)  │  │
│  │  - Set Constraints (from ToolConstraintsJSON)            │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 3: Create gRPC Stream                              │  │
│  │  - client.Generate(generateReq) → gRPC stream            │  │
│  │  - Connects to SGLang Backend (Rust)                      │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 4: Create Converter & BatchPostprocessor          │  │
│  │  - ffi.CreateGrpcResponseConverterWithTokenizer()       │  │
│  │  - Uses preprocessed.PromptTokens for initial count      │  │
│  │  - ffi.NewBatchPostprocessor(batchSize=1, immediate)     │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Step 5: Start readLoop (Background Goroutine)           │  │
│  │  - go grpcStream.readLoop()                               │  │
│  │  - Returns GrpcChatCompletionStream immediately          │  │
│  └────────────────────┬─────────────────────────────────────┘  │
└───────────────────────┼────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         GrpcChatCompletionStream.readLoop()                     │
│         (Background Goroutine)                                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Recv() Goroutine (Dedicated)                            │  │
│  │  - Continuously calls stream.Recv()                      │  │
│  │  - Sends results to recvChan (buffered, 2000)          │  │
│  │  - Exits on ctx.Done() or error                          │  │
│  │  - Calls stream.CloseSend() on ctx.Done()               │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                          │
│                       ▼                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Main Loop                                                │  │
│  │  - Reads from recvChan                                    │  │
│  │  - For each proto.GenerateResponse:                      │  │
│  │    → go processAndSendResponse() (async)                 │  │
│  │      - protoToJSON() converts proto to JSON string        │  │
│  │      - batchPostprocessor.AddChunk(protoJSON)            │  │
│  │        → FFI postprocessing (token decoding, tool parsing)│  │
│  │        → Returns OpenAI-format JSON strings               │  │
│  │      - Sends JSON to resultJSONChan (buffered, 10000)     │  │
│  │      - All operations check ctx.Done() for cancellation  │  │
│  │  - On EOF: flush batch, send remaining results, return  │  │
│  │  - On error: send to errChan (buffered, 100)            │  │
│  │  - defer: cancel ctx, wait goroutines, close channels     │  │
│  └────────────────────┬─────────────────────────────────────┘  │
└───────────────────────┼────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│         resultJSONChan (Buffered Channel, 10000)                 │
│         - Contains OpenAI-format JSON strings                    │
│         - Ready for consumption                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         ChatCompletionStream.RecvJSON()                          │
│         (client.go:410)                                          │
│         - Direct wrapper: return grpcStream.RecvJSON()           │
│         - No intermediate processing                             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         FastHTTP SetBodyStreamWriter                             │
│         (handlers/chat.go:159)                                   │
│         - Loop: stream.RecvJSON() → format SSE → flush         │
│         - Format: "data: {json}\n\n"                           │
│         - Final: "data: [DONE]\n\n"                             │
│         - Immediate flush after each chunk                      │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        HTTP Client                               │
│                    (SSE Stream)                                  │
│                    Receives: data: {...}\n\n                    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Start Server

```bash
./run.sh
```

The server will start on port `:8080`.

### Usage Example

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Key Design

### 1. Thread-Safe Tokenizer
- Pre-create `TokenizerHandle` at startup
- Rust side uses `Arc<dyn TokenizerTrait>`, thread-safe
- Lock-free concurrency, eliminating lock contention

### 2. Context Cancellation Mechanism (Graceful Shutdown)
- Use `context.Context` cancellation mechanism
- In `readLoop`'s `defer`: cancel context first, then wait for all goroutines to complete, finally close channels
- `processAndSendResponse` checks `ctx.Done()` at function start, all `select` statements include `case <-s.ctx.Done()`
- Avoids "send on closed channel" panic

### 3. Cancellable Recv()
- Use dedicated goroutine to execute `Recv()`
- Pass results through `recvChan`
- Call `CloseSend()` when context is cancelled to make `Recv()` return error

### 4. Simplified Channel Design
- `resultJSONChan`: Main data channel (gRPC layer)
- `errChan`: Error channel (gRPC layer)
- `recvChan`: Internal communication channel (gRPC layer)
- Removed redundant channels and duplicate reads

## Configuration

### Channel Buffer Sizes

```go
type ChannelBufferSizes struct {
    ResultJSONChan int // Default: 10000
    ErrChan        int // Default: 100
    RecvChan       int // Default: 2000
}
```

### Timeout Configuration

```go
type Timeouts struct {
    KeepaliveTime    time.Duration // Default: 300s
    KeepaliveTimeout time.Duration // Default: 20s
    CloseTimeout     time.Duration // Default: 5s
}
```

## Performance Optimizations

1. **Pre-create Tokenizer**: Created at startup to avoid first request latency
2. **Lock-Free Concurrency**: Tokenizer is thread-safe, no locks needed
3. **Lazy Parsing**: JSON parsing deferred until needed
4. **Direct JSON Passing**: `RecvJSON()` avoids parse/serialize overhead
5. **Immediate Batching**: batchSize=1, no delay
6. **Async Processing**: `readLoop` processes in background, doesn't block request handling
7. **Configurable Buffers**: Adjust channel sizes based on concurrency needs

## File Structure

```
sgl-model-gateway/bindings/golang/
├── client.go                          # High-level client API
├── internal/
│   ├── grpc/
│   │   └── client_grpc.go            # gRPC client implementation
│   ├── ffi/                          # FFI bindings (Rust)
│   └── proto/                        # Protobuf definitions
└── examples/
    └── oai_server/
        ├── handlers/
        │   └── chat.go               # HTTP request handling
        ├── models/
        │   └── chat.go               # Request/response models
        └── service/
            └── sglang_service.go      # Service layer
```

## Error Handling

### Context Cancellation Mechanism
1. **Client disconnects** → `SetBodyStreamWriter` detects flush error
2. **Cancel streamCtx** → `readLoop` detects `ctx.Done()`
3. **Call stream.CloseSend()** → `Recv()` goroutine returns error
4. **readLoop defer executes**:
   - Set `closed` flag
   - Cancel context (if not already cancelled)
   - Wait for all `processAndSendResponse` goroutines to complete (`processWg.Wait()`)
   - Close all channels (`resultJSONChan`, `errChan`, `readLoopDone`)
5. **Clean up resources and exit**

### Channel Blocking and Race Condition Prevention
- **Context cancellation mechanism**: All channel sends use `select` statements with `case <-s.ctx.Done()`
- **Graceful exit**: When context is cancelled, all blocking send operations can return immediately
- **WaitGroup synchronization**: `readLoop`'s `defer` uses `processWg.Wait()` to ensure all goroutines complete before closing channels
- **Avoid panic**: Through context cancellation and WaitGroup synchronization, avoids "send on closed channel" panic

## Key Functions

### CreateChatCompletionStream
**Location**: `internal/grpc/client_grpc.go:108`
- Preprocess request (FFI)
- Build gRPC request
- Create converter and batch processor
- Start `readLoop`

### readLoop
**Location**: `internal/grpc/client_grpc.go:290`
- Start Recv() goroutine (continuously calls `stream.Recv()`)
- Process proto responses
- Asynchronously call `processAndSendResponse` (tracked with `processWg`)
- **Graceful shutdown in defer**:
  - Set `closed` flag
  - Cancel context (if not already cancelled)
  - Wait for all `processAndSendResponse` goroutines to complete (`processWg.Wait()`)
  - Close all channels (`resultJSONChan`, `errChan`, `readLoopDone`)

### processAndSendResponse
**Location**: `internal/grpc/client_grpc.go:379`
- Check `ctx.Done()` at function start, return immediately if cancelled
- Convert proto to JSON
- Call FFI batch processor
- All `select` statements include `case <-s.ctx.Done()` for graceful shutdown handling
- Send JSON to channel

### RecvJSON
**Location**:
- `internal/grpc/client_grpc.go:412`: gRPC layer implementation
- `client.go:410`: Client wrapper layer
- Read from `resultJSONChan`
- Directly return JSON string, no parsing needed
