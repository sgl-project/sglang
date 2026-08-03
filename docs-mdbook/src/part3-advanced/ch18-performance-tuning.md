# 第 18 章 性能调优实战：从指标到案例

> 本章不列参数表（那是 `server_args.py` 和官方文档的事），而是讲"怎么系统地找到瓶颈并验证修复"。

## 18.1 先定义指标

| 指标 | 定义 | 谁在乎 |
| --- | --- | --- |
| TTFT | 首 token 延迟 | 交互体验 |
| TPOT / ITL | 相邻输出 token 间隔 | 流式体验 |
| Throughput | tokens/s（prefill + decode 分开看） | 成本 |
| 显存占用 | KV 预算、激活峰值 | 能开多大并发 |
| 缓存命中率 | 命中 token / 总 token | 前缀复用效率 |

**调优第一步永远是测基线**：`python -m sglang.benchmark.serving`（OpenAI 协议、可配请求速率）或 `benchmark/bench_serving.py`（更完整）。没有基线，所有"优化"都无从谈起。

## 18.2 瓶颈定位三张表

### 症状 → 瓶颈 → 对策

| 症状 | 瓶颈 | 对策 |
| --- | --- | --- |
| GPU 利用率低 + batch 小 | CPU 调度跟不上 | `--enable-overlap-schedule`；检查 CUDA graph 覆盖；CPU 核数/主频 |
| GPU 利用率高 + TTFT 高 | prefill 计算量大 | chunked prefill；加 prefill 实例（PD）；检查缓存命中率 |
| GPU 利用率高 + TPOT 高 | decode 访存受限 | 投机解码；量化 KV/权重；减小 batch 波动 |
| GPU 利用率高 + 请求完成慢 | 队列排队 | 加大并发/实例；调 `max_running_requests`；检查抢占是否频繁 |
| 显存 OOM | 预算分配不当 | 降 `mem_fraction_static`；量化；开 memory saver |

### 缓存维度

| 现象 | 原因 | 对策 |
| --- | --- | --- |
| `cached_tokens` 一直是 0 | 请求间无公共前缀 / radix 被禁用 | 检查 prompt 分布；`--schedule-policy lpm` |
| 命中率波动大 | 路由不亲和 / 缓存被淘汰 | router cache-aware；加大 KV 预算 |
| 命中但速度没提升 | 命中段不在页边界 | 检查 `page_size` 与共享粒度 |

### 显存维度

| 手段 | 收益 | 代价 |
| --- | --- | --- |
| `--kv-cache-dtype fp8_e5m2` 等 | KV 减半 | 精度下降，需验证 |
| 权重量化（fp8/awq/gptq） | 权重减半 + 访存减半 | 精度/兼容性 |
| 降低 `mem_fraction_static` | 更稳 | 并发上限下降 |
| LoRA buffer 管理 | 多租户容量 | 动态加载复杂度 |

## 18.3 案例一：在线服务 TTFT 突然劣化

现象：压测时 TTFT 从 200ms 涨到 2s，TPOT 正常。

排查顺序（对照代码）：

1. **看等待队列**：`/metrics` 的 waiting queue 长度是否增长 → 如果是，调度器来不及消化（CPU 瓶颈），开 overlap；
2. **看 prefill batch**：是否出现超大 prefill 把 GPU 占住 → chunked prefill 参数是否生效；
3. **看缓存命中率**：压测用的 prompt 是否随机导致命中率近 0 → 换有前缀分布的负载测试，或确认 LPM；
4. **看抢占**：`mem_fraction_static` 是否过低导致频繁抢占 → 抢占会让请求"先算一半再等"，TTFT 剧增。

结论通常是 2+4 的组合：长 prefill 挤占 + 抢占回退。对策：chunked prefill + 调大 KV 预算。

## 18.4 案例二：离线批处理吞吐上不去

现象：GPU 利用率 60%，batch 上不去，显存还有富余。

排查：

1. `max_running_requests` 是否卡住 batch 上限 → 调大；
2. CUDA graph 是否覆盖目标 batch → `--cuda-graph-bs` 补上；
3. CPU 侧是否成为瓶颈 → 开 overlap，检查 `schedule_policy` 的排序开销（LPM 在大队列会退化 FCFS）；
4. 显存其实够，但 allocator 的页碎片/保留导致可用量低 → 调 `page_size` 或看 pool 统计。

## 18.5 案例三：长上下文（100k+）

长上下文会把三类资源同时逼到极限：

1. **KV 显存**：线性增长，100k token × 多层 × 多请求很快爆掉 → 量化 KV、PD 分离（P 侧专门吃长 prompt）、HiCache 存储级缓存；
2. **prefill 计算**：首 token 延迟和 prompt 长度成正比 → chunked prefill 让其他请求能插队；
3. **attention 复杂度**：全量 attention 是平方级 → 稀疏注意力模型（DSA 等）+ 对应 backend。

这也是为什么官方把 GB300 长上下文单独写 blog：**长上下文不是一个参数能解决的，需要架构级组合**。

## 18.6 一个被低估的参数：new_token_ratio

调度器里有个"新 token 比例"的估算（`new_token_ratio_tracker`），它影响 decode 请求预留多少 KV 空间。如果你的负载是"输出特别长"（如 Agent 写长文），这个比例会被持续低估/高估，导致 prefill 挤占 decode 的显存预算。观察 pool 使用率与预估值是否偏差大，是调优时容易漏的一环。

## 18.7 调优方法论总结

```text
1. 明确优化指标（TTFT / TPOT / 吞吐 / 成本）
2. 测基线，记录指标与 GPU/CPU/显存利用率
3. 用 18.2 的表定位瓶颈（先看利用率，再看队列，再看缓存）
4. 一次只改一个参数，复测
5. 回归验证：精度（量化）、正确性（投机/PD）不能牺牲
```

## 18.8 本章小结

- 调优是"定位瓶颈"的游戏，不是"堆参数"。
- 三张表：利用率表（CPU/GPU/显存）、缓存表、显存表。
- 三个案例覆盖在线、离线、长上下文三类典型负载。
- 每次只改一个变量，用指标验证，别信感觉。

> 下一章：怎么观察和排查——metrics、trace、benchmark 与故障流程。
