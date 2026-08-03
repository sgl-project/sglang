# 附录 B：关键文件索引

按主题列出本仓库最重要文件的路径与作用，方便快速跳转。

## 入口与启动

| 文件 | 作用 |
| --- | --- |
| `python/sglang/cli/serve.py` | `sglang serve` 命令实现（含模型类型探测） |
| `python/sglang/cli/main.py` | CLI 子命令分发 |
| `python/sglang/launch_server.py` | 旧版启动入口（`python -m sglang.launch_server`） |
| `python/sglang/srt/entrypoints/http_server.py` | FastAPI 入口，全部 HTTP 端点 |
| `python/sglang/srt/entrypoints/engine.py` | 进程内 Engine 与多进程拉起（`launch_server` 主流程） |
| `python/sglang/srt/entrypoints/EngineBase.py` | Engine 接口定义 |
| `python/sglang/srt/server_args.py` | 全部命令行参数（唯一真相） |

## 请求链路

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/managers/io_struct.py` | `GenerateReqInput` 等全部消息对象 |
| `python/sglang/srt/managers/tokenizer_manager.py` | TokenizerManager：tokenize、校验、ZMQ 投递 |
| `python/sglang/srt/managers/detokenizer_manager.py` | Detokenizer 进程 |
| `python/sglang/srt/managers/communicator.py` | ZMQ 通信封装 |
| `python/sglang/srt/entrypoints/openai/serving_chat.py` | OpenAI chat 协议 → GenerateReqInput |
| `python/sglang/srt/entrypoints/openai/protocol.py` | OpenAI 协议对象定义 |

## 调度

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/managers/scheduler.py` | Scheduler 类、事件循环、组批 |
| `python/sglang/srt/managers/schedule_policy.py` | SchedulePolicy、PrefillAdder/DecodeAdder |
| `python/sglang/srt/managers/schedule_batch.py` | `Req`、`ScheduleBatch` 定义 |
| `python/sglang/srt/managers/data_parallel_controller.py` | DP Controller |
| `python/sglang/srt/managers/scheduler_pp_mixin.py` | 流水线并行 mixin |

## 显存与缓存

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/mem_cache/radix_cache.py` | RadixCache（前缀树） |
| `python/sglang/srt/mem_cache/memory_pool.py` | ReqToTokenPool、KVCache 各实现 |
| `python/sglang/srt/mem_cache/allocator/` | 页/token 粒度分配器 |
| `python/sglang/srt/mem_cache/evict_policy.py` | 淘汰策略 |
| `python/sglang/srt/mem_cache/cpp_radix_tree/` | C++ radix 树 |
| `python/sglang/srt/mem_cache/flush_cache.py` | `/flush_cache` 实现 |

## 模型执行

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/model_executor/model_runner.py` | ModelRunner、forward 主路径 |
| `python/sglang/srt/model_executor/forward_batch_info.py` | ForwardBatch |
| `python/sglang/srt/model_executor/runner_backend/` | CUDA graph 各后端 |
| `python/sglang/srt/layers/attention/` | 注意力后端（flashinfer/triton/mla…） |
| `python/sglang/srt/layers/sampler.py` | 采样器 |
| `python/sglang/srt/layers/logits_processor.py` | logits 后处理 |
| `python/sglang/srt/compilation/` | torch.compile 集成 |
| `python/sglang/srt/models/` | 各模型实现 |

## 分布式

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/distributed/parallel_state.py` | 并行拓扑 |
| `python/sglang/srt/distributed/communication_op.py` | 通信原语 |
| `python/sglang/srt/distributed/bootstrap.py` | 多进程/多机初始化 |
| `python/sglang/srt/hardware_backend/` | 硬件适配 |
| `python/sglang/srt/elastic_ep/` | 弹性专家并行 |
| `python/sglang/srt/eplb/` | 专家负载均衡 |

## 高级特性

| 文件 | 作用 |
| --- | --- |
| `python/sglang/srt/disaggregation/` | PD 分离（prefill/decode/encode、传输后端） |
| `python/sglang/srt/speculative/` | 投机解码（EAGLE/MTP/DFlash…） |
| `python/sglang/srt/lora/` | LoRA 管理器、内存池、重叠加载 |
| `python/sglang/srt/constrained/` | 结构化输出后端（xgrammar/outlines/llguidance） |
| `python/sglang/srt/multimodal/processors/` | 各模型多模态处理器 |
| `python/sglang/srt/observability/` | metrics / trace / 统计 |

## 前端与 Rust

| 文件 | 作用 |
| --- | --- |
| `python/sglang/__init__.py` | 顶层公共 API 导出（Runtime/Engine/原语） |
| `python/sglang/lang/api.py` | 编程原语（gen/select/image/video 等） |
| `python/sglang/lang/backend/runtime_endpoint.py` | RuntimeEndpoint 后端 |
| `rust/sglang-server/src/` | Rust 服务端 |
| `rust/sglang-grpc/src/server.rs` | gRPC server |
| `rust/sglang-mm/src/` | Rust 多模态处理 |
| `proto/sglang/runtime/v1/sglang.proto` | gRPC 协议 |
| `experimental/sgl-router/src/policies/` | 路由策略 |

## 测试与示例

| 文件 | 作用 |
| --- | --- |
| `test/srt/` | 运行时 pytest |
| `test/run_suite.py` | 测试套件入口 |
| `python/sglang/test/mock_model/` | 模拟模型 |
| `examples/runtime/engine/` | 引擎用法示例 |
| `python/sglang/benchmark/serving.py` | 服务 benchmark |
| `benchmark/` | 各类专项 benchmark |
