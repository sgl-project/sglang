# 附录 A：术语表

| 术语 | 全称/原文 | 一句话解释 | 相关章节 |
| --- | --- | --- | --- |
| TTFT | Time To First Token | 从发请求到收到第一个 token 的时间，主要受 prefill 影响 | 18 |
| TPOT/ITL | Time Per Output Token / Inter-Token Latency | 相邻输出 token 的时间间隔，主要受 decode 影响 | 18 |
| SRT | SGLang Runtime | 推理运行时，仓库中 `python/sglang/srt/` | 2 |
| KV Cache | Key-Value Cache | 自回归解码保存的历史注意力键值，显存大头 | 8 |
| RadixAttention | — | 用 radix 树组织 KV 页，共享公共前缀 | 8 |
| Continuous Batching | 连续批处理 | 请求完成即离队、新请求随时入队的批处理 | 7 |
| Prefill | 预填充 | 处理整段 prompt、写入 KV 的阶段（计算密集） | 6, 7 |
| Decode | 解码 | 逐 token 生成阶段（访存密集） | 6, 7 |
| Chunked Prefill | 分块预填充 | 把超长 prefill 切成小块，避免独占 GPU | 7 |
| Scheduler | 调度器 | 决定"下一批跑谁、跑什么阶段"的进程 | 7 |
| TokenizerManager | 分词管理 | HTTP 层与调度器之间做 tokenize/detokenize 的枢纽 | 5, 6 |
| ModelRunner | 模型执行器 | GPU 侧的模型前向封装 | 9 |
| ForwardBatch | 前向批 | 一次 GPU 前向的完整输入描述 | 9 |
| CUDA Graph | CUDA 图 | 录制并回放 kernel 序列，压低 CPU 启动开销 | 9 |
| ZMQ | ZeroMQ | 进程间消息传递库，SGLang 多进程通信的基础 | 2, 6 |
| TP / EP / DP / PP | Tensor/Expert/Data/Pipeline Parallel | 四种并行切分方式 | 10 |
| PD Disaggregation | Prefill/Decode 分离 | prefill 与 decode 由不同实例承担并传输 KV | 13 |
| Speculative Decoding | 投机解码 | 小模型草稿 + 大模型验证，多出正确 token | 14 |
| EAGLE / MTP / DFlash | — | SGLang 支持的几种投机算法 | 14 |
| LoRA | Low-Rank Adaptation | 低秩增量微调，多适配器可共享基座模型 | 15 |
| Grammar / FSM | 文法 / 有限状态机 | 约束解码时"当前合法 token 集合"的状态机 | 11 |
| Jump-Forward | 跳步 | 文法确定前缀时一次跳过多个 token | 11 |
| sgl-router | — | KV-aware 的 OpenAI 兼容路由层 | 17 |
| Rollout | 展开/采样 | RL 中策略模型生成样本的过程 | 20 |
| Detokenizer | 反分词 | token 流还原成文本的进程 | 6 |
| Memory Pool | 显存池 | 预分配的 KV 张量与分配器 | 8 |
| Page | 页 | KV 分配的最小连续单元（`page_size`） | 8 |
| Eviction | 淘汰 | 缓存超限时按策略移除节点 | 8 |
| extra_key | — | RadixKey 的命名空间，隔离不同 LoRA/会话的前缀 | 8, 15 |
| GrammarManager | 文法管理 | 调度器侧 grammar 编译与队列化 | 11 |
| DP Controller | 数据并行控制器 | 引擎内部分发请求给多个 Scheduler 副本 | 10, 17 |
| Hierarchical Cache | 层级缓存 | HiCache：KV 可存到更大容量/异构存储 | 8, 13 |
| HiCache / Mooncake / Nixl | — | PD 分离的 KV 传输后端 | 13 |
| Load Format | 权重加载格式 | 模型权重如何加载（auto/safetensors/presharded 等） | 18 |
| Quantization | 量化 | 权重/KV 用低精度表示，省显存提吞吐 | 18 |
| Deterministic Inference | 确定性推理 | 保证多次推理结果一致（RL 训练用） | 20 |
| Watchdog | 看门狗 | 监控子进程健康并重启的机制 | 20 |
| Mock Model | 模拟模型 | 不加载真实权重的测试用模型 | 21 |
