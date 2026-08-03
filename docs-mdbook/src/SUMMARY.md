# 总结

[前言：写给谁、怎么读](intro.md)

---

# 第一部分：入门 —— 先跑起来，再读代码

- [第 1 章 认识 SGLang：一个 LLM Serving 框架的定位](part1-basics/ch01-what-is-sglang.md)
- [第 2 章 代码仓地图：从顶层目录到核心模块](part1-basics/ch02-repo-map.md)
- [第 3 章 快速开始：安装、启动与第一个请求](part1-basics/ch03-quickstart.md)
- [第 4 章 前端 sglang.lang：编程范式与 Backend 抽象](part1-basics/ch04-frontend.md)
- [第 5 章 HTTP 服务层：OpenAI 兼容 API 与协议适配](part1-basics/ch05-http-serving.md)
- [第 6 章 一次请求的完整生命周期](part1-basics/ch06-request-lifecycle.md)

# 第二部分：进阶 —— 核心机制深入

- [第 7 章 调度器 Scheduler：连续批处理与事件循环](part2-intermediate/ch07-scheduler.md)
- [第 8 章 内存与 KV Cache：RadixAttention 与层级化缓存](part2-intermediate/ch08-memory-kv-cache.md)
- [第 9 章 模型执行：ModelRunner、Attention Backend 与 CUDA Graph](part2-intermediate/ch09-model-execution.md)
- [第 10 章 并行策略：TP / EP / DP / PP](part2-intermediate/ch10-parallelism.md)
- [第 11 章 采样与结构化输出](part2-intermediate/ch11-sampling-structured-output.md)
- [第 12 章 多模态支持：图像、视频、音频的接入方式](part2-intermediate/ch12-multimodal.md)

# 第三部分：进阶级 —— 分布式与性能专家之路

- [第 13 章 Prefill/Decode 分离 (PD Disaggregation)](part3-advanced/ch13-pd-disaggregation.md)
- [第 14 章 投机解码：EAGLE、MTP 与 DFlash](part3-advanced/ch14-speculative-decoding.md)
- [第 15 章 多 LoRA：参数高效微调的批量服务](part3-advanced/ch15-lora.md)
- [第 16 章 Rust 组件：sglang-server、sglang-grpc 与 sglang-mm](part3-advanced/ch16-rust-components.md)
- [第 17 章 路由与集群扩展：sgl-router 与多实例部署](part3-advanced/ch17-router-and-scaling.md)
- [第 18 章 性能调优：从 server_args 到工程实践](part3-advanced/ch18-performance-tuning.md)
- [第 19 章 可观测性与调试：Metrics、Trace、Profiling 与 Benchmark](part3-advanced/ch19-observability-debugging.md)
- [第 20 章 RL 与后训练：SGLang 作为 Rollout 引擎](part3-advanced/ch20-rl-and-posttraining.md)
- [第 21 章 参与贡献：测试、开发流程与学习路线](part3-advanced/ch21-contributing.md)

# 附录

- [附录 A：术语表](appendix/appx-terms.md)
- [附录 B：关键文件索引](appendix/appx-file-index.md)
