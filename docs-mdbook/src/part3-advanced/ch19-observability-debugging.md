# 第 19 章 可观测性与调试：Metrics、Trace、Profiling 与 Benchmark

## 19.1 可观测性三件套

生产环境必须能回答"现在发生了什么、之前发生了什么、为什么慢了"。SGLang 在 `srt/observability/` 提供了三件套：

| 能力 | 代码位置 | 说明 |
| --- | --- | --- |
| Metrics | `observability/metrics_collector.py` | Prometheus 指标，`--enable-metrics` 开启 |
| Tracing | `observability/trace.py` | 请求级 trace（配合 `--enable-tracing`） |
| Request 统计 | `observability/req_time_stats.py` | 每请求各阶段耗时 |

`entrypoints/http_server.py` 在 `--enable-metrics` 时挂载 Prometheus middleware，导出 `/metrics`；`observability/request_metrics_exporter.py` 负责把请求级指标导出。

## 19.2 核心指标

`metrics_collector.py`（约 1000 行）收集的指标可分为：

- **吞吐**：每步生成 token 数、prefill/decode token 数；
- **延迟**：TTFT、TPOT、请求总时长的直方图；
- **队列**：waiting queue 长度、running batch 大小；
- **显存**：KV 池使用率、可用页数；
- **缓存**：前缀命中 token 数、命中率；
- **健康**：请求失败/超时数、abort 数。

配合 Grafana 可以画出"缓存命中率 × 吞吐"这类关联图，是调优时的第一手数据。

## 19.3 请求级 Trace

`observability/trace.py` 支持把一次请求的完整路径（tokenize → 调度 → prefill → decode → detokenize）串成 trace。生产上可用它做慢请求归因：到底是排队久、prefill 慢还是 decode 慢。

## 19.4 Benchmark：把优化变成数字

仓库提供多层次的 benchmark：

### 服务层

```bash
python -m sglang.benchmark.serving \
  --model-path <model> --num-prompts 100 --request-rate 10
```

输出 OpenAI 协议下的吞吐、TTFT、TPOT 分布。`benchmark/bench_serving.py` 是更完整的版本，支持 sharegpt 等真实 prompt 分布。

### 离线吞吐

`python/sglang/benchmark/offline_throughput.py`（顶层 `python/sglang/bench_offline_throughput.py` 亦可用）测纯引擎吞吐，不带 HTTP 开销。

### 单批延迟

`python/sglang/benchmark/one_batch.py` / `one_batch_server.py`：测单 batch 的端到端延迟，适合对比"参数改动对延迟的影响"。

### 专项

`benchmark/` 下还有 json_schema、lora、speculative、deepseek_v3、kernels、scheduler 等专项，`benchmark/scheduler/` 直接测调度器决策开销。`examples/profiler/nsys_profile_tools/` 提供 Nsight Systems 的 GPU profile 辅助脚本。

## 19.5 调试技巧与工具

- **日志分级**：`--log-level debug` 能看到请求进入/离开各阶段的日志；
- **开关诊断**：`--disable-cuda-graph`、`--disable-radix-cache` 等把复杂特性逐个关闭，做二分定位；
- **内存检查**：`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY`（`environ.py` 中定义的环境变量）开启严格检查；
- **调试工具**：`srt/debug_utils/` 有 tensor dump、文本对比、CUDA core dump 等，`dumper.py` 可在前向时 dump 张量用于比对；
- **注入测试**：`srt/kv_canary/` 是 KV 正确性金丝雀（canary）测试，检测缓存/注意力实现是否污染；
- **CI 测试**：`test/srt/` 下按模块组织 pytest，本地改代码后先跑相关子目录。

## 19.6 本章小结

- 可观测性 = metrics + trace + request stats，均在 `srt/observability/`。
- Benchmark 分服务层、离线层、专项层，先测基线再谈优化。
- 调试靠"开关二分 + debug 日志 + dump 工具"三件套。
- 下一章看 SGLang 的另一张面孔：作为 RL 训练引擎。
