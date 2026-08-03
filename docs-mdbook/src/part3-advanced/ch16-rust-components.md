# 第 16 章 Rust 组件：架构决策与演进逻辑

## 16.1 为什么 SGLang 需要 Rust

一句话：**当 CPU 比 GPU 先饱和时，Python 解释器就成了瓶颈。**

推理集群里 CPU 的活并不少：HTTP 解析、JSON 序列化、tokenize、detokenize、请求调度元数据处理。单个 token 的 Python 处理开销可能是几微秒到几十微秒，而 decode 每步 GPU 才几十微秒——**CPU 侧多花的时间会直接变成 TPOT 的一部分**。

Rust 的价值不是"快一点点"，而是把高频路径的开销从"微秒级 Python"降到"纳秒级原生代码"，让 CPU 不再是瓶颈。

## 16.2 替换的边界：只搬高频、易错、纯计算的活

`rust/` 下三个 crate 的职责划分非常清晰：

| crate | 职责 | 替换了 Python 的谁 |
| --- | --- | --- |
| `sglang-server` | HTTP 服务端 + tokenizer manager + detokenizer | `entrypoints/http_server.py` + `managers/tokenizer_manager.py` 的网络/文本部分 |
| `sglang-grpc` | 原生 gRPC 服务（`--grpc-port`）+ Python 桥 | `entrypoints/grpc_bridge.py` 等 |
| `sglang-mm` | 图像 resize/transform、Inkling 视觉引擎 | 多模态预处理 |

注意边界：**GPU 推理（Scheduler/ModelRunner）没有被 Rust 替换**。原因很实在：

1. 调度/执行的核心在 PyTorch/CUDA，Python 只是胶水，替换收益低、风险高；
2. Rust 侧替换的都是"高频、纯计算、容易出序列化 bug"的路径——每替换一块，CPU 瓶颈就后退一步。

这是"渐进式替换"的范本：不搞大爆炸重写，而是沿着性价比最高的路径逐步蚕食。

## 16.3 Rust 与 Python 怎么对话：ring.rs 的通道

`rust/sglang-server/src/ring.rs` 定义了两个边界队列：

```text
Ingress：Rust tokenizer manager → Python scheduler（请求进来）
Egress ：Python scheduler → Rust（结果出去）
```

工程细节（文件注释里写得很清楚）：

- 嵌入式模式下是**进程内 flume channel**，不做序列化以外的拷贝；
- 消息体用 msgpack，大数组（`input_ids`）用**列式 int64 原始字节**直传，绕开 msgpack 的大对象开销；
- Python 侧只调非阻塞的 `drain`/`try_push`，不持有 GIL 等 Rust 线程，避免 GIL 死锁。

`message/io_struct.rs` 与 Python 侧 `managers/io_struct.py` 保持同构，保证两侧对"请求长什么样"的理解一致。

## 16.4 演进方向：从代码里读出来的路线图

从仓库现状能推断出 SGLang 的 Rust 化路线：

1. **HTTP 层**（已完成大半）：`api_server/` 有 `openai.rs`、`native_api.rs`，`--rust-server` 相关参数可切换；
2. **协议层**：`proto/sglang/runtime/v1/sglang.proto` 定义了 gRPC 契约，`sglang-grpc` 是原生实现——训练框架（verl 等）接入的低开销通道；
3. **边界**：`sglang-mm` 把多模态预处理也搬了过来，说明"能搬的都搬"是明确方向。

对读者的启示：如果你要评估"某个 Python 路径该不该 Rust 化"，参照这个标准——**频率高不高、开销占比大不大、纯计算还是涉 GPU、有没有现成协议**。

## 16.5 测试与质量

Rust 侧不是二等公民：`rust/sglang-grpc/src/server/tests.rs`、`bridge/tests.rs` 都是完整测试；`.github/workflows/pr-test-rust.yml` 说明 Rust 改动有专门的 CI。改 Rust 代码的标准是"和 Python 侧同等严格"。

## 16.6 本章小结

- Rust 化的动机是 CPU 先于 GPU 饱和，收益在 TPOT/吞吐。
- 替换边界清晰：高频、纯计算、易错的路径；GPU 推理不动。
- 两侧通过 ring.rs 的进程内通道通信，消息结构在 Rust/Python 双端同构。
- 演进方向是 HTTP → gRPC/协议 → 多模态预处理，逐步蚕食。

> 下一章：单机之外——路由、集群与容量规划。
