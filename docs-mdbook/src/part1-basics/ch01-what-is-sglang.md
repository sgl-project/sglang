# 第 1 章 认识 SGLang：一个 LLM Serving 框架的定位

## 1.1 一句话概括

SGLang 是一个面向大语言模型（LLM）和多模态模型的高性能推理服务框架，定位是"从单张 GPU 到大型分布式集群"都能高效运行的 serving 层。它的核心思路是：**把"调度"和"执行"解耦，让 CPU 上的调度器与 GPU 上的模型执行各自做到极致**。

项目首页（`README.md`）官方列出四大特点：

- **快速运行时**：RadixAttention 前缀缓存、零开销 CPU 调度器、prefill/decode 分离、投机解码、连续批处理、paged attention、张量/流水线/专家/数据并行、结构化输出、chunked prefill、量化、多 LoRA 批处理。
- **广泛的模型支持**：Llama、Qwen、DeepSeek、GLM、Gemma 等文本模型，e5-mistral 等 embedding 模型，Skywork 等 reward 模型，以及 WAN、Qwen-Image 等扩散模型。
- **广泛的硬件支持**：NVIDIA（GB200/B300/H100/A100…）、AMD（MI355/MI300…）、Intel CPU、Google TPU、华为 Ascend NPU 等。
- **RL/后训练骨干**：被 AReaL、Miles、slime、Tunix、verl 等训练框架用作 rollout 后端。

## 1.2 它解决什么问题

把一个大模型部署成"服务"，表面上是 `model.generate(...)`，但工业级 serving 要解决一批很实际的问题：

1. **并发与吞吐**：多个用户同时请求，怎么在同一个 GPU 上把请求拼成 batch，又不让长请求饿死短请求？
2. **显存**：KV Cache 随序列长度线性增长，怎么分页、复用、淘汰，让更多请求共享有限显存？
3. **首 token 延迟 (TTFT) 与 token 间延迟 (TPOT)**：用户体感取决于这两个指标，分别由 prefill 和 decode 阶段决定，优化手段完全不同。
4. **协议**：客户端的 OpenAI SDK、工具调用、流式输出、结构化 JSON 输出，怎么在服务端统一承接？
5. **规模**：模型大到一张卡放不下（如 DeepSeek 671B），怎么在多卡、多机之间切分并保持通信高效？

SGLang 对上述每个问题都有成体系的答案，这也是它适合作为源码研究对象的原因。

## 1.3 从仓库里能看到什么

这个仓库不只是"一个服务"，而是一个完整的工程生态，主要分几块：

| 部分 | 位置 | 作用 |
| --- | --- | --- |
| 前端语言 | `python/sglang/lang/` | 可编程的 `LLM`/`Engine` 接口，`gen()`、`chat()` 等原语 |
| SRT 运行时 | `python/sglang/srt/` | 核心：HTTP 入口、TokenizerManager、Scheduler、ModelRunner、KV Cache |
| 内核与算子 | `python/sglang/kernels/` | Triton/CUDA 算子（JIT 与 AOT 两种形态） |
| Rust 组件 | `rust/` | 高性能 gRPC/HTTP 服务骨架、多模态处理 |
| 路由 | `experimental/sgl-router/` | KV-aware 的 OpenAI 兼容负载均衡路由 |
| 基准测试 | `benchmark/` | 各类端到端 benchmark 与论文复现脚本 |
| 官方文档 | `docs_new/` | 面向用户的官方站点源码（Mintlify 格式） |

## 1.4 核心概念的第一次接触

后面所有章节都会反复用到这几个词，先建立一个直觉：

- **Scheduler（调度器）**：一个不断循环的进程，决定"下一批该跑哪些请求、跑 prefill 还是 decode、显存够不够"。
- **TokenizerManager（分词管理）**：位于 HTTP 层与调度器之间，负责把请求文本转成 token ids、把输出 token 转回文本，并维护每个请求的状态。
- **KV Cache（键值缓存）**：自回归解码必须保存的历史注意力键值，是显存占用的主要来源，也是前缀缓存（RadixAttention）的作用对象。
- **RadixAttention**：SGLang 的标志性优化。把 KV Cache 组织成 radix 树，请求之间的公共前缀可以共享，多轮对话、few-shot 场景能省掉大量重复 prefill。
- **Continuous Batching（连续批处理）**：请求完成就离开 batch，新请求随时可以加入，而不是等整个 batch 一起结束。
- **Prefill / Decode（预填充 / 解码）**：prefill 处理整段 prompt 并写入 KV Cache，是计算密集的；decode 逐 token 生成，是访存（memory-bound）密集的。

## 1.5 值得了解的历史与社区

仓库 `README.md` 的 News 列表本身就是一份发展史：

- 2024 年初发布，凭 RadixAttention 获得约 5x 推理加速；
- 2024 年用压缩 FSM 实现 3x 更快的 JSON 解码；
- 2024 年 12 月 v0.4 发布"零开销批调度器"与 KV-aware 负载均衡；
- 2025 年在 GB200/96×H100 等集群上支撑 DeepSeek 等超大模型，成为 RL 训练的主流 rollout 引擎；
- 2026 年扩展到了扩散模型、音频、VLA，以及新一代投机解码 DFlash / Spec V2。

了解这些演进脉络很有价值：`experimental/`、`rust/` 这些目录里既有历史包袱也有前沿尝试，读代码时不必认为所有模块都是同一代设计。

## 1.6 本章小结

- SGLang = 高性能 LLM serving 框架 + 可编程前端 + RL rollout 引擎。
- 它的核心矛盾是：GPU 计算资源宝贵，而请求形态（长度、优先级、模态）千差万别，所以要有一层聪明的调度。
- 本仓库是一套完整工程：Python 运行时、Rust 组件、内核算子、基准测试、路由。
- 下一章我们带着这张地图进入代码仓，把每个目录的职责搞清楚。
