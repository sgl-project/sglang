# 第 19 章 可观测性与故障排查：从指标到定位

## 19.1 三件套：指标、Trace、日志

生产排障需要回答三个问题：**现在发生了什么**（指标）、**这次请求经历了什么**（trace）、**为什么**（日志）。SGLang 对应三件套：

| 能力 | 开启方式 | 回答什么 |
| --- | --- | --- |
| Prometheus 指标 | `--enable-metrics` | 吞吐、延迟分布、队列、显存、缓存命中 |
| 请求级 trace | `--enable-tracing` | 一次请求各阶段耗时 |
| 分级日志 | `--log-level debug` | 内部流转细节 |

指标实现在 `srt/observability/metrics_collector.py`，trace 在 `trace.py`。

## 19.2 指标怎么读：先看这五组

| 指标组 | 具体指标 | 异常信号 |
| --- | --- | --- |
| 吞吐 | prefill/decode token 速率 | 与 GPU 利用率不匹配 |
| 延迟 | TTFT/TPOT 直方图 | P99 与 P50 差距大 = 抖动 |
| 队列 | waiting queue 长度、running batch 大小 | 队列持续增长 = 消化不了 |
| 显存 | KV 池使用率、可用页 | 使用率贴顶 = 抢占风险 |
| 缓存 | 命中 token 数、命中率 | 命中率低 = 前缀复用失效 |

**P99 与 P50 的差距比平均值重要**：调度类系统的均值好看，尾部可能惨不忍睹。看直方图而不是平均值。

## 19.3 请求级 trace：慢请求归因

一个请求的耗时可以被拆成几段（`observability/req_time_stats.py` 记录）：

```text
总耗时 = 网络传输 + 排队等待 + tokenize + prefill 耗时 + Σdecode 耗时 + detokenize + 网络返回
```

排查"这个请求为什么慢"的决策树：

```text
排队占比高   → 实例负载高 / 调度不过来
prefill 占比高 → prompt 太长 / 缓存没命中
decode 占比高 → 输出长 / batch 太大互相拖累 / 访存瓶颈
tokenize/detokenize 占比高 → CPU 瓶颈，检查 Rust 化是否生效
```

## 19.4 Benchmark 工具矩阵

| 工具 | 测什么 | 用法 |
| --- | --- | --- |
| `python -m sglang.benchmark.serving` | OpenAI 协议端到端（吞吐/TTFT/TPOT） | 压测首选 |
| `benchmark/bench_serving.py` | 同上有真实 prompt 分布 | 更接近生产 |
| `python/sglang/benchmark/offline_throughput.py` | 纯引擎吞吐（无 HTTP） | 隔离网络开销 |
| `python/sglang/benchmark/one_batch.py` | 单 batch 延迟 | 参数对比 |
| `benchmark/json_schema/`、`lora/`、`speculative/` | 专项能力 | 特性回归 |
| `benchmark/scheduler/` | 调度器决策开销 | 调度优化 |

跑 benchmark 的纪律：固定 GPU 型号、固定并发/速率、固定 prompt 分布，**一次只改一个变量**，记录复现环境。

## 19.5 故障排查流程：开关二分法

SGLang 的复杂特性很多，排查"行为异常"时用开关二分：

```text
现象：输出错误 / 变慢 / OOM
  ├─ 关 CUDA graph（--disable-cuda-graph）
  ├─ 关前缀缓存（--disable-radix-cache）
  ├─ 关投机（不传 --speculative-algorithm）
  ├─ 关 overlap（去掉 --enable-overlap-schedule）
  ├─ 换注意力后端（--attention-backend torch_native）
  └─ 最小复现：单请求、短 prompt、无并发
```

哪个开关关闭后现象消失，瓶颈/嫌疑就在那个特性里，再深入该模块的代码。

## 19.6 经典故障与处置速查

| 故障 | 快速判断 | 处置 |
| --- | --- | --- |
| CUDA OOM | 日志堆栈在 alloc | 降 mem_fraction_static / 量化 / 减并发 |
| 输出乱码/重复 | 无 | 查 KV 正确性：跑 `kv_canary` 测试（`srt/kv_canary/`）、关 radix cache 对比 |
| 慢但利用率低 | 队列空、GPU 闲 | CPU 瓶颈：overlap、CUDA graph 覆盖、Rust 化 |
| 随机性结果不一致 | 多次请求同 prompt 输出不同 | 采样参数；`--deterministic-inference`（RL 用） |
| 流式卡顿 | TPOT 抖动 | 查 decode batch 波动、抢占、共享 GPU 的其他任务 |
| 启动即崩 | 堆栈在初始化 | `--log-level debug` 复现；检查硬件/驱动/依赖版本 |

## 19.7 生产监控建议

1. 必配告警：waiting queue 突增、KV 使用率 > 90%、TTFT P99 超阈值、GPU 利用率 < 30% 且队列非空；
2. 缓存命中率进 dashboard：它是最容易被忽视的"免费性能"；
3. 版本化记录每个实例的 server_args：**没有参数记录的压测结果没有价值**；
4. 崩溃时保留日志与 `nsys profile`（`examples/profiler/nsys_profile_tools/`）。

## 19.8 本章小结

- 可观测性三件套回答"发生了什么 / 经历了什么 / 为什么"。
- 看直方图别看均值；P99-P50 的差距是抖动信号。
- 慢请求按"排队/prefill/decode/CPU"四段归因。
- 排障用开关二分 + 最小复现；正确性疑云先跑 kv_canary。
- 生产必备：队列、显存、TTFT、命中率四类告警。

> 下一章：RL 与后训练场景——SGLang 的另一张面孔。
