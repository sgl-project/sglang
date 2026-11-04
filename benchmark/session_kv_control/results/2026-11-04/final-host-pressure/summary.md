# Session KV 实验结果

尝试 6 条；有效 6 条。

TTFT 从恢复请求发出计时，到首个实际输出 token；完整周期包含源计算、发布、工具等待和恢复请求。

表内延迟包含全部完成的恢复请求，包括控制失败的尝试；valid 列只表示路径与输出校验通过数量。无效路径不可用于证明缓存收益。

| tokens / wait / lead / group | attempts | completed | valid | TTFT p50 ms | p95 ms | cycle p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [8192, 3.0, 2.5, "on_demand"] | 3 | 3 | 3 | 464.433 | 818.465 | 5713.871 |
| [8192, 3.0, 2.5, "prefetch"] | 3 | 3 | 3 | 67.837 | 145.847 | 4677.660 |

| paired group | eligible pairs | completed pairs | valid pairs | candidate − on-demand TTFT p50 ms | cycle p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| [8192, 3.0, 2.5, "prefetch"] | 3 | 3 | 3 | -383.530 | -1084.116 |

| paired group | N | mean TTFT delta ms | paired bootstrap 95% CI ms |
| --- | ---: | ---: | --- |
| [8192, 3.0, 2.5, "prefetch"] | 3 | -494.788 | [-790.915, -309.919] |

bootstrap 对完整配对差值采样；小样本、相关时序与状态变化会限制区间解释。ready 子集只描述已完成准备的样本，全部样本仍保留。

| group | ready-before-resume N / mean TTFT ms | not-ready N / mean TTFT ms | ambiguous usage N |
| --- | --- | --- | ---: |
| [8192, 3.0, 2.5, "on_demand"] | 0 / — | 0 / — | 0 |
| [8192, 3.0, 2.5, "prefetch"] | 3 / 96.413 | 0 / — | 0 |

原始请求、响应、控制状态、范围与错误均在 raw.jsonl。未观测到的物理 I/O、host 峰值和服务端 CPU 保留为空值。
