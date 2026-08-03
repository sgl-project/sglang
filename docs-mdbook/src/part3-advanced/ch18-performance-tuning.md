# 第 18 章 性能调优：从 server_args 到工程实践

## 18.1 调优的目标与指标

先定义目标：

- **TTFT**（Time To First Token）：用户体验首 token 延迟，主要被 prefill 影响；
- **TPOT / ITL**（Time Per Output Token / Inter-Token Latency）：连贯性感知，主要被 decode 影响；
- **Throughput**（tokens/s）：系统吞吐，batch 越大越高，但与延迟有 trade-off；
- **显存占用**：决定能同时跑多少请求。

任何调优都要先明确"优化哪个指标"，因为多数参数是权衡。仓库 benchmark（第 19 章）输出这些指标，先测基线再动手。

## 18.2 server_args 关键参数地图

`python/sglang/srt/server_args.py` 是参数的唯一真相（约 9000 行，含参数组 `NS(...)` 命名空间）。按维度分组：

### 显存与批大小

| 参数 | 作用 | 常见调整方向 |
| --- | --- | --- |
| `--mem-fraction-static` | KV 可占显存比例 | 小值保权重/激活，大值提并发 |
| `--max-running-requests` | 运行请求上限 | 调大提升吞吐，调小保延迟 |
| `--max-prefill-tokens` | 单批 prefill token 上限 | 防止长 prefill 卡住 batch |
| `--chunked-prefill-size` | 长请求分块大小 | 小→延迟稳，大→吞吐高 |

### 调度

| 参数 | 作用 |
| --- | --- |
| `--schedule-policy` | lpm / fcfs / lof 等（第 7 章） |
| `--enable-priority-scheduling` | 请求带 `priority` 字段时按优先级 |
| `--enable-mixed-chunk` | prefill/decode 混合进同一 batch |
| `--enable-overlap-schedule` | CPU 调度与 GPU 前向重叠 |

### 执行加速

| 参数 | 作用 |
| --- | --- |
| `--attention-backend` / `--decode-attention-backend` / `--prefill-attention-backend` | 指定注意力内核 |
| `--cuda-graph-bs` / `--cuda-graph-max-bs` | CUDA graph 覆盖的 batch 序列 |
| `--cuda-graph-backend-decode` | full / breakable / tc_piecewise |
| `--disable-cuda-graph` | 诊断用 |
| `--torch-compile-*` 系列 | torch.compile 开关与后端 |

### 高级特性开关

`--enable-dp-attention`、`--enable-speculative-*`、`--enable-lora-*`、`--disaggregation-*`、`--quantization`、`--kv-cache-dtype`（fp8/int4 量化 KV）等，均在对应章节讲过。

## 18.3 调优方法论（对照代码）

### 第一步：定位瓶颈

- 用 `--log-level info` + metrics 看 batch 大小、pool 使用率（`mem_cache/allocator` 的统计）；
- GPU 利用率低但 batch 大：看 CPU 调度是否成为瓶颈 → 开 `--enable-overlap-schedule`；
- GPU 利用率高但单请求慢：看 batch 是否过大、是否有长 prefill 混入 → 调 chunked prefill；
- TTFT 高：检查 `schedule_policy` 与 prefill 并发限制；
- 显存 OOM：看 `mem_fraction_static`、量化 KV、LoRA 缓存。

### 第二步：缓存效率

前缀缓存命中率是 SGLang 调优的独有维度：

- 多轮/共享 prompt 场景确保 `--schedule-policy lpm`；
- 用 `--enable-cache-report` 观察每个请求命中 token 数；
- `/flush_cache` 在测试不同 prompt 分布时用于隔离；
- 会话场景用 `session_id` + session radix cache（`srt/session/`）跨请求复用。

### 第三步：显存效率

- 量化 KV：`--kv-cache-dtype fp8_e5m2` 等（`mem_cache/kv_cache_dtype.py`）；
- 权重量化：`--quantization fp8` / awq / gptq 等（`srt/layers/quantization/`、`model_loader/`）；
- 内存 saver：`--enable-memory-saver`（TorchMemorySaverAdapter）。

## 18.4 实战案例：从参数反推场景

| 场景 | 推荐组合 |
| --- | --- |
| 在线低延迟（小并发） | 小 batch、禁用 chunk、优先 decode 延迟、CUDA graph 全覆盖 |
| 离线高吞吐（批处理） | 大 batch、`max-running-requests` 拉满、混合 chunk |
| 长上下文 | chunked prefill、PD 分离、注意力后端选长序列优化版本 |
| 多租户 | Multi-LoRA + `enable-lora-overlap-loading` + 优先级调度 |
| DeepSeek 级 MoE | TP+EP 组合、MLA 后端、speculative（MTP/DFlash） |

## 18.5 本章小结

- 调优 = 明确指标 → 测基线 → 定位瓶颈 → 改参数 → 复测。
- 参数集中在 `server_args.py`，分为显存、调度、执行加速三大类。
- SGLang 特有的调优维度是前缀缓存命中率与 CUDA graph 覆盖。
- 下一章讲怎么观察：metrics、trace、profiling 与 benchmark。
