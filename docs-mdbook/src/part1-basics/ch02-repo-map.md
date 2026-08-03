# 第 2 章 代码仓地图：从顶层目录到核心模块

## 2.1 顶层目录

```text
sglang/
├── python/            # Python 包主体（前端 + SRT 运行时 + 内核）
├── rust/              # Rust 组件（HTTP/gRPC 服务骨架、多模态）
├── proto/             # gRPC 协议定义
├── benchmark/         # 端到端 benchmark 脚本
├── examples/          # 各类用法示例
├── experimental/      # 实验性组件（sgl-router 等）
├── docs_new/          # 官方用户文档（Mintlify 格式）
├── scripts/           # CI / 开发脚本
├── docker/            # Dockerfile
├── 3rdparty/          # 第三方子模块
└── test/              # Python 测试（pytest）
```

其中 `python/sglang/` 下又分几个平行世界：

```text
python/sglang/
├── srt/               # SGLang Runtime：真正的 serving 引擎
├── lang/              # 前端：LLM / Engine / 编程原语
├── kernels/           # Triton / CUDA 自定义算子
├── benchmark/         # serving 基准（与顶层 benchmark/ 部分重叠）
├── eval/              # 评测脚本
├── multimodal_gen/    # 扩散模型（图像/视频生成）运行时
├── cli/               # sglang 命令行（serve/generate/version）
├── launch_server.py   # 旧版启动入口（python -m sglang.launch_server）
└── __init__.py        # 顶层公共 API 导出（Runtime/Engine/原语）
```

## 2.2 SRT 内部：服务的五脏六腑

`python/sglang/srt/` 是理解这个仓库的钥匙，核心子目录如下：

| 目录 | 职责 |
| --- | --- |
| `entrypoints/` | 进程入口：`http_server.py`（FastAPI）、`engine.py`（Engine/多进程拉起）、`openai/`、`anthropic/`、`ollama/` 协议适配 |
| `managers/` | 各管理进程：`tokenizer_manager.py`、`scheduler.py`、`detokenizer_manager.py`、`data_parallel_controller.py` |
| `model_executor/` | GPU 侧执行：`model_runner.py`、CUDA graph runner、ForwardBatch 定义 |
| `models/` | 各种模型的 PyTorch 实现（llama、qwen、deepseek、glm…） |
| `mem_cache/` | 显存管理：`radix_cache.py`（前缀缓存）、`memory_pool.py`（KV 池）、`allocator/` |
| `layers/` | 可复用层：attention（radix attention）、linear、moe、rotary、sampler、logits_processor |
| `distributed/` | 并行通信：`parallel_state.py`、`communication_op.py`、bootstrap |
| `disaggregation/` | prefill/decode 分离：传输、KV 事件、gRPC |
| `speculative/` | 投机解码：EAGLE、MTP、DFlash 等 |
| `lora/` | 多 LoRA 管理：`lora_manager.py`、内存池、重叠加载 |
| `hardware_backend/` | 各厂商硬件适配（CUDA/ROCm/XPU/NPU/TPU…） |
| `constrained/` | 结构化输出：xgrammar / outlines / llguidance 后端 |
| `multimodal/` | 多模态输入处理：图像/视频/音频 |
| `observability/` | metrics、trace、监控 |
| `compilation/` | torch.compile 相关编译管线 |
| `server_args.py` | 全部命令行参数的定义（约 9000 行，字段即功能） |

## 2.3 进程模型：先建立一个全局图景

SGLang 服务是**多进程架构**。单卡部署时典型布局如下：

```text
┌───────────────────────────────────────────────────┐
│  主进程 (launch_server)                          │
│  ├── HTTP Server (uvicorn/FastAPI)               │
│  ├── TokenizerManager（也可独立进程/线程）        │
│  └── 拉起子进程：                                 │
│      ├── Scheduler 进程 (每 TP 组一个)            │
│      │   └── 内含 ModelRunner (GPU worker)       │
│      ├── Detokenizer 进程                        │
│      └── (可选) DP Controller / Router 等        │
└───────────────────────────────────────────────────┘
```

进程间通过 **ZeroMQ (ZMQ)** 传递消息（`managers/communicator.py`、`entrypoints/engine.py` 中的 `zmq`），用 pickle/msgspec 序列化。scheduler 与 HTTP 层解耦带来的好处是：CPU 调度与 GPU 计算可以并行推进（`event_loop_overlap`），这会在第 7 章展开。

## 2.4 Rust 与协议

- `rust/sglang-server/`：Rust 实现的 HTTP/gRPC 服务骨架，可以替代 Python 侧的 FastAPI 层（`--rust-server` 相关参数），内部也有自己的 tokenizer manager 实现。
- `rust/sglang-grpc/`：原生 gRPC 服务与 Python 侧的桥接（`entrypoints/grpc_bridge.py`）。
- `rust/sglang-mm/`：多模态相关的 Rust 处理（如图像去重、inkling 引擎）。
- `proto/sglang/runtime/v1/sglang.proto`：gRPC 协议定义，是未来服务接口演进的方向。

## 2.5 官方文档目录的阅读价值

`docs_new/docs/` 按 `get-started`、`basic_usage`、`advanced_features`、`developer_guide`、`references` 组织，是"怎么用"的权威来源；本 mdbook 侧重"怎么实现的"。两者配合使用效果最好。比如 `advanced_features/pd_disaggregation.mdx` 讲用法，第 13 章讲 `python/sglang/srt/disaggregation/` 的代码实现。

## 2.6 本章小结

- 仓库 = Python 运行时（主）+ Rust 组件（辅助）+ 内核 + 测试 + 文档。
- SRT 是核心，`entrypoints`（入口）→ `managers`（调度/分词）→ `model_executor`（执行）→ `mem_cache`（显存）构成主干。
- 服务是多进程 + ZMQ 消息传递的架构，这是理解请求链路的前提。
- 下一步：把它跑起来，亲手发一个请求，再对照第 6 章的链路图看日志。
