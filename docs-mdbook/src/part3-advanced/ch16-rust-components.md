# 第 16 章 Rust 组件：sglang-server、sglang-grpc 与 sglang-mm

## 16.1 为什么会有 Rust

Python 的瓶颈在于解释器开销：每个 HTTP 请求、每个 token 的序列化/反序列化、tokenizer 调用，都会产生可观 CPU 成本。当服务规模大到"CPU 先于 GPU 饱和"时，把高频路径搬到 Rust 就是自然的工程选择。

SGLang 的 Rust 代码在 `rust/` 下三个 crate：

| crate | 职责 |
| --- | --- |
| `sglang-server` | Rust 版 HTTP/gRPC 服务端：API 层 + tokenizer manager + detokenizer |
| `sglang-grpc` | 原生 gRPC 服务（`--grpc-port`），以及桥接到 Python 侧的入口 |
| `sglang-mm` | 多模态处理：图像 resize/transform、Inkling 引擎 |

## 16.2 sglang-server：Rust 里的"HTTP + TokenizerManager"

`rust/sglang-server/src/` 的结构几乎复刻了 Python 侧的分层：

```text
api_server/          # HTTP API：native_api.rs、openai.rs、pd_bootstrap.rs
tokenizer_manager/   # ingress（请求进来）/ egress（结果出去）
detokenizer.rs       # 增量 detokenize
message/             # io_struct.rs、sampling.rs、request.rs（Rust 版协议对象）
runtime/             # 线程池与运行时：threads.rs、runnable.rs
fsm.rs               # 有限状态机（结构化输出）
ring.rs              # Rust↔Python 边界的进程内队列（flume channel）
tokenizer.rs         # tokenizer 封装
```

它与 Python 的 Scheduler/ModelRunner 的关系：Rust 服务端承担**网络与文本处理**，GPU 侧推理仍由 Python Scheduler 执行——Rust 通过内部通道把 `BatchTokenizedGenerateReqInput` 交给调度器。`ring.rs` 就是这条边界的实现（`Ingress`：Rust tokenizer manager → Python scheduler；`Egress` 反向），在嵌入式模式下是进程内 flume 队列，消息体用 msgpack/列式 int64 传输。这是"各取所长"的分工。

## 16.3 sglang-grpc：下一代接口

`proto/sglang/runtime/v1/sglang.proto` 定义了服务契约；`rust/sglang-grpc/src/server.rs` 是 gRPC server 实现，`bridge.rs` 提供与 Python 侧的桥接（Python 侧对应 `entrypoints/grpc_bridge.py`）。`--grpc-port` 让 HTTP 服务旁边同时起 gRPC 端口，便于训练框架（verl 等）低开销接入。

## 16.4 sglang-mm：多模态的 Rust 加速

`rust/sglang-mm/src/common/` 提供图像 resize、transform；`inkling/` 是 Inkling 视觉引擎的 Rust 实现。它由 Python 侧 `srt/multimodal/inkling/` 与 `processors/inkling.py` 调用，证明"多模态预处理也可下沉到 Rust"。

## 16.5 工程启示

- **渐进式替换**：Rust 不是推翻重写，而是沿着"高频、纯计算、易出错"的路径逐步替换（HTTP 解析 → tokenize → 结构化输出 → gRPC）。
- **协议先行**：`proto/` 与 `message/io_struct.rs` 保证 Rust/Python 两侧消息结构一致。
- **测试**：`rust/sglang-grpc/src/server/tests.rs`、`bridge/tests.rs` 等表明 Rust 侧也有完善的单元/集成测试（`cargo test`）。

## 16.6 本章小结

- Rust 组件是"Python 运行时的高速带"，覆盖 API、tokenizer、detokenizer、gRPC、多模态预处理。
- 推理主体仍在 Python Scheduler，Rust 负责不涉及 GPU 计算的高频路径。
- 读代码时遇到"同名文件出现在 rust/ 和 python/sglang/srt/"，多半是同一职责的两种实现。
- 下一章把视角拉远：多实例、负载均衡与 KV-aware 路由。
