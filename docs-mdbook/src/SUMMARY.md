# 总结

[前言：这份文档怎么读](intro.md)

---

# 第一部分：入门 —— 先建立直觉，再学会使用

> 目标读者：刚接触 LLM 服务，不知道 KV Cache 是什么。读完能讲清请求怎么被服务、能独立完成启动与调用。

- [第 1 章 从零理解：LLM 推理服务在干什么](part1-basics/ch01-what-is-sglang.md)
- [第 2 章 第一次启动：把服务跑起来](part1-basics/ch02-first-run.md)
- [第 3 章 服务端是一座分工明确的餐厅（心智模型）](part1-basics/ch03-mental-model.md)
- [第 4 章 用 Python 调用服务：Engine、Runtime 和编程原语](part1-basics/ch04-python-client.md)
- [第 5 章 对外 API 入门：OpenAI 兼容接口怎么用](part1-basics/ch05-http-api-basics.md)
- [第 6 章 一次请求的完整旅程（入门版）](part1-basics/ch06-request-lifecycle.md)

# 第二部分：进阶 —— 读懂核心代码

> 目标读者：已经跑通 SGLang。读完能对着源码讲出调度器、缓存、执行器、并行、采样、多模态的工作原理。

- [第 7 章 调度器代码走读：下一批跑什么、为什么](part2-intermediate/ch07-scheduler.md)
- [第 8 章 KV Cache 与 RadixAttention 代码走读](part2-intermediate/ch08-memory-kv-cache.md)
- [第 9 章 模型执行代码走读：ForwardBatch、Attention Backend 与 CUDA Graph](part2-intermediate/ch09-model-execution.md)
- [第 10 章 并行策略代码走读：TP / EP / DP / PP](part2-intermediate/ch10-parallelism.md)
- [第 11 章 采样与结构化输出代码走读](part2-intermediate/ch11-sampling-structured-output.md)
- [第 12 章 多模态实现走读：图像如何变成 token](part2-intermediate/ch12-multimodal.md)

# 第三部分：进阶级 —— 理解权衡，面向生产

> 目标读者：做生产部署、性能调优、二次开发。读完能解释每个设计为什么存在、怎么失效、怎么排查。

- [第 13 章 Prefill/Decode 分离：为什么、怎么传、什么坑](part3-advanced/ch13-pd-disaggregation.md)
- [第 14 章 投机解码：原理、收益、工程代价](part3-advanced/ch14-speculative-decoding.md)
- [第 15 章 多 LoRA：显存账、动态加载与踩坑](part3-advanced/ch15-lora.md)
- [第 16 章 Rust 组件：架构决策与演进逻辑](part3-advanced/ch16-rust-components.md)
- [第 17 章 集群路由与容量规划：从单机到多实例](part3-advanced/ch17-router-and-scaling.md)
- [第 18 章 性能调优实战：从指标到案例](part3-advanced/ch18-performance-tuning.md)
- [第 19 章 可观测性与故障排查：从指标到定位](part3-advanced/ch19-observability-debugging.md)
- [第 20 章 RL 与后训练：推理引擎的第二战场](part3-advanced/ch20-rl-and-posttraining.md)
- [第 21 章 二次开发与贡献：从改文档到改内核](part3-advanced/ch21-contributing.md)

# 附录

- [附录 A：术语表](appendix/appx-terms.md)
- [附录 B：关键文件索引](appendix/appx-file-index.md)
- [附录 C：SM89/L40S 生产部署实录](appendix/appx-sm89-l40s-deployment.md)
