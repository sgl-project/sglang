# Go SGLang Router - OpenAI 兼容 API 服务器

Go SGLang Router 是一个高性能的 OpenAI 兼容 API 服务器，使用 gRPC 与 SGLang 后端通信，并通过 Rust FFI 进行高效的预处理和后处理。

## 特性

- ✅ **OpenAI API 兼容**: 完全兼容 OpenAI Chat Completions API
- ✅ **高性能**: 使用 gRPC 和 Rust FFI 实现低延迟和高吞吐
- ✅ **流式传输**: 支持 Server-Sent Events (SSE) 流式响应
- ✅ **线程安全**: 预创建的 tokenizer handle，无锁并发
- ✅ **优雅关闭**: 使用 context 取消机制，避免资源泄漏和 panic
- ✅ **可配置**: 支持配置 channel 缓冲区大小和超时时间

## 架构概览

**重要说明**：gRPC 模式**仍然会调用 FFI**，FFI 用于：
- **预处理**：chat_template 和 tokenization（请求阶段）
- **后处理**：token decoding 和 tool parsing（响应阶段）

gRPC 仅用于与 SGLang 后端通信，输入输出的处理完全依赖 Rust FFI。

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

## 快速开始

### 启动服务器

```bash
./run.sh
```

服务器将在 `:8080` 端口启动。

### 使用示例

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## 关键设计

### 1. 线程安全的 Tokenizer
- 启动时预创建 `TokenizerHandle`
- Rust 端使用 `Arc<dyn TokenizerTrait>`，线程安全
- 无锁并发，消除锁竞争

### 2. Context 取消机制（优雅关闭）
- 使用 `context.Context` 的取消机制
- `readLoop` 的 `defer` 中：先取消 context，然后等待所有 goroutine 完成，最后关闭 channel
- `processAndSendResponse` 在函数开始时检查 `ctx.Done()`，所有 `select` 语句都包含 `case <-s.ctx.Done()`
- 避免了 "send on closed channel" panic

### 3. 可取消的 Recv()
- 使用专门的 goroutine 执行 `Recv()`
- 通过 `recvChan` 传递结果
- Context 取消时调用 `CloseSend()` 使 `Recv()` 返回错误

### 4. 简化的 Channel 设计
- `resultJSONChan`: 主要数据通道（gRPC 层）
- `errChan`: 错误通道（gRPC 层）
- `recvChan`: 内部通信通道（gRPC 层）
- 移除了冗余的 channel 和重复读取

## 配置

### Channel 缓冲区大小

```go
type ChannelBufferSizes struct {
    ResultJSONChan int // 默认: 10000
    ErrChan        int // 默认: 100
    RecvChan       int // 默认: 2000
}
```

### 超时配置

```go
type Timeouts struct {
    KeepaliveTime    time.Duration // 默认: 300s
    KeepaliveTimeout time.Duration // 默认: 20s
    CloseTimeout     time.Duration // 默认: 5s
}
```

## 性能优化

1. **预创建 Tokenizer**: 启动时创建，避免首次请求延迟
2. **无锁并发**: Tokenizer 线程安全，无需锁
3. **Lazy Parsing**: JSON 解析延迟到需要时
4. **直接 JSON 传递**: `RecvJSON()` 避免解析/序列化开销
5. **立即批处理**: batchSize=1，无延迟
6. **异步处理**: `readLoop` 在后台处理，不阻塞请求处理
7. **可配置的缓冲区**: 根据并发需求调整 channel 大小

## 文件结构

```
sgl-model-gateway/bindings/golang/
├── client.go                          # 高级客户端 API
├── internal/
│   ├── grpc/
│   │   └── client_grpc.go            # gRPC 客户端实现
│   ├── ffi/                          # FFI 绑定（Rust）
│   └── proto/                        # Protobuf 定义
└── examples/
    └── oai_server/
        ├── handlers/
        │   └── chat.go               # HTTP 请求处理
        ├── models/
        │   └── chat.go               # 请求/响应模型
        └── service/
            └── sglang_service.go      # 服务层
```

## 错误处理

### Context 取消机制
1. **客户端断开连接** → `SetBodyStreamWriter` 检测到 flush 错误
2. **取消 streamCtx** → `readLoop` 检测到 `ctx.Done()`
3. **调用 stream.CloseSend()** → `Recv()` goroutine 返回错误
4. **readLoop defer 执行**：
   - 设置 `closed` 标志
   - 取消 context（如果还没有被取消）
   - 等待所有 `processAndSendResponse` goroutine 完成（`processWg.Wait()`）
   - 关闭所有 channel（`resultJSONChan`, `errChan`, `readLoopDone`）
5. **清理资源并退出**

### Channel 阻塞和竞态条件防护
- **Context 取消机制**：所有 channel 发送都使用 `select` 语句，包含 `case <-s.ctx.Done()`
- **优雅退出**：当 context 被取消时，所有阻塞的发送操作都能立即返回
- **WaitGroup 同步**：`readLoop` 的 `defer` 中使用 `processWg.Wait()` 确保所有 goroutine 完成后再关闭 channel
- **避免 panic**：通过 context 取消和 WaitGroup 同步，避免了 "send on closed channel" panic

## 关键函数

### CreateChatCompletionStream
**位置**: `internal/grpc/client_grpc.go:108`
- 预处理请求（FFI）
- 构建 gRPC 请求
- 创建 converter 和 batch processor
- 启动 `readLoop`

### readLoop
**位置**: `internal/grpc/client_grpc.go:290`
- 启动 Recv() goroutine（持续调用 `stream.Recv()`）
- 处理 proto 响应
- 异步调用 `processAndSendResponse`（使用 `processWg` 跟踪）
- **defer 中的优雅关闭**：
  - 设置 `closed` 标志
  - 取消 context（如果还没有被取消）
  - 等待所有 `processAndSendResponse` goroutine 完成（`processWg.Wait()`）
  - 关闭所有 channel（`resultJSONChan`, `errChan`, `readLoopDone`）

### processAndSendResponse
**位置**: `internal/grpc/client_grpc.go:379`
- 在函数开始时检查 `ctx.Done()`，如果已取消则立即返回
- 转换 proto 到 JSON
- 调用 FFI batch processor
- 所有 `select` 语句都包含 `case <-s.ctx.Done()` 来优雅处理关闭
- 发送 JSON 到 channel

### RecvJSON
**位置**: 
- `internal/grpc/client_grpc.go:412`: gRPC 层实现
- `client.go:410`: 客户端包装层
- 从 `resultJSONChan` 读取
- 直接返回 JSON 字符串，无需解析


