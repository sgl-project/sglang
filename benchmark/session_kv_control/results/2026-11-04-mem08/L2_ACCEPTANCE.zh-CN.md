# mem_fraction_static=0.8 的 L2 保留压力测试

两种口径共 732 次真实生成调用，全部完成并通过缓存、输出与控制校验。热点平均 TTFT 降低约 84%–87%；整个混合工作负载的吞吐约持平。不能把热点改善扩展为总服务吞吐的大幅提升。

执行 Engine 源码为 `f65f0393c4c916184c6c9017fafb9fa389ec2f4a`。本报告对比同一实现上的按需恢复与 L2 PIN，属于策略动作对照。后续 native 流水与排队 demand 复用改动由另行运行的 L3 矩阵检验。

## 配置与实际压力

- Qwen3-8B BF16，单机双 TP=1 Engine；A 为 A100 80GB PCIe，B 为 A100-SXM4-80GB。配对测试均在 B 执行。
- 两个 Engine 均 `--mem-fraction-static 0.8`，没有显式 `max-total-tokens` 上限，`max-running-requests=32`。B 的 L1 为 346,944 tokens，L2 为 693,952 tokens；A 的 L1 为 346,688 tokens。自动分配差异约 0.074%，配置参数保持相同。
- HiCache ratio=2、direct/page_first_direct、write_through、persistent host；Mooncake v0.3.14 TCP，两个 Engine 各挂载 128GiB segment，共享 Master。未使用 RDMA。
- page=64，chunked prefill=4096，context=32768，seed=0，triton attention/pytorch sampling，关闭 overlap 和 decode/prefill CUDA graph。
- 每组 4 个独立 16K 热点：真实计算、确认完整 L3 发布、DEMOTE 到 host；候选组对全部 65,536 tokens 获取有界 L2 eviction-exclusion PIN。随后生成 53 个独立 16K 冷前缀冲刷，冷数据量 868,352 tokens，约为 L2 的 1.25 倍。压力并发 8，热点恢复输出 32 tokens，3 轮配对随机顺序。
- 每种口径 366 次生成：24 次热点预热、318 次冷压力、24 次恢复。每个动作组含 12 条热点恢复与 183 次全部生成调用。两组使用独立 salt，防止后一组读取前一组刚产生的 L3 内容。

`natural` 保留冷请求的 GPU KV，只 COPY 到 host，让原生 LRU 处理容量。它的 GPU KV 分配/预留采样峰值为 344,320/346,944（约 99.24%），host 峰值为 692,224/693,952（约 99.75%）。`demote` 对每条冷压力请求显式回收 GPU，GPU KV 峰值仅 65,792 tokens，用作分层动作成本对照。

这里的 GPU KV 占用来自启动容量减 `kv_available_tokens`，包含缓存及分配中的工作。`token_usage` / `kv_used_tokens` 仅表示活动请求锁定的槽位。资源每 2 秒采样，峰值为观测值；`nvidia-smi` 常驻分配也不等于 KV 内容占用。原始监控在压缩 JSONL 中保留。

## 结果

TTFT/p95 只统计热点恢复；工作时长与请求/s 包含预热、发布、全部冷压力、恢复及控制清理，仅扣除每组起始的实验 `/flush_cache` RPC。capability refresh 继续计入。L2 workload 没有增加工具等待来制造收益。

| 口径 | 动作 | 热点平均 TTFT ms | 热点 p95 ms | 工作时长 s | 全部完成请求/s |
| --- | --- | ---: | ---: | ---: | ---: |
| natural | 按需恢复 | 4552.012 | 7880.944 | 453.716 | 0.4033 |
| natural | PIN | 586.192 | 767.835 | 451.510 | 0.4053 |
| demote | 按需恢复 | 3630.534 | 6154.894 | 421.662 | 0.4340 |
| demote | PIN | 573.820 | 871.750 | 422.218 | 0.4334 |

每种口径中，按需组热点共 196,608 tokens 实际来自 L3；PIN 组同一数量实际来自 L2，没有 GPU 命中。全部 24 个 PIN acquisition 有完整 active/eviction-exclusion 范围，24 个 release/query 返回释放墓碑。热点输出在组间一致，未重复计算已发布前缀。

natural 的热点平均 TTFT 约降低 87.1%，工作吞吐约提高 0.49%；demote 的热点约降低 84.2%，工作吞吐约下降 0.13%。每种口径只有 3 个独立配对轮次，热点每组 N=12，不能据此宣称稳定尾延迟或稳定总吞吐收益。冷计算占多数，GPU 预填充成本主导整个混合工作负载。

natural 每轮 PIN 减按需的工作时长差为 +1.25s、−7.88s、+4.43s；demote 为 −5.71s、−3.54s、+9.80s，均保留。natural 的原始 inclusive 总时长为按需 513.087s、PIN 463.465s；其中一组起始 flush 约 59.3s、另一组约 11.9s，直接用 inclusive 差值会夸大服务收益。分析器同时保留这组计时。

## 证据与重算

两个子目录保留 schedule、manifest、每组完整记录、raw HTTP/SSE/control JSONL、原始未压缩 checksum 索引及压缩后的物理文件 checksum 索引。`provenance` 保存实际启动参数、执行命令和测试日志。本阶段 `resources.jsonl.gz` 是两种 L2 实验时段的原始监控记录截取，截取边界与 SHA-256 在 `provenance/l2-resource-capture.json`。后续总报告会补充完整长测监控。

```sh
.venv/bin/python benchmark/session_kv_control/analyze_stress.py \
  --results benchmark/session_kv_control/results/2026-11-04-mem08 \
  --output /tmp/mem08-l2-recheck
```

归档后的两种口径均通过独立离线重算，732 次原始调用与完成数相符，输出、完整 ready 范围、PIN 与释放凭证、文件哈希均通过。

合成 ledger 使用不同 salt 构成独立缓存域，没有扫描真实应用轨迹、共享前缀比例或多节点 RDMA。GPU2/3 有同机其他任务，未操作这些任务。CPU 回归及日常同机负载可能影响时段，配对随机顺序与原始资源记录供检查；这些数据不能建立跨环境吞吐保证。
