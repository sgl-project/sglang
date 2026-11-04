# Session KV 实验结果

尝试 12 条；有效 12 条。

TTFT 从恢复请求发出计时，到首个实际输出 token；完整周期包含源计算、发布、工具等待和恢复请求。

表内延迟包含全部完成的恢复请求，包括控制失败的尝试；valid 列只表示路径与输出校验通过数量。无效路径不可用于证明缓存收益。

| tokens / wait / lead / group | attempts | completed | valid | TTFT p50 ms | p95 ms | cycle p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [16384, 3.0, 2.5, "on_demand"] | 3 | 3 | 3 | 1494.945 | 2037.854 | 8475.798 |
| [16384, 3.0, 2.5, "prefetch"] | 3 | 3 | 3 | 409.029 | 509.593 | 7799.406 |
| [8192, 3.0, 2.5, "on_demand"] | 3 | 3 | 3 | 1046.542 | 1448.539 | 6053.155 |
| [8192, 3.0, 2.5, "prefetch"] | 3 | 3 | 3 | 175.400 | 318.167 | 5806.215 |

| paired group | eligible pairs | completed pairs | valid pairs | candidate − on-demand TTFT p50 ms | cycle p50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| [8192, 3.0, 2.5, "prefetch"] | 3 | 3 | 3 | -890.349 | -246.940 |
| [16384, 3.0, 2.5, "prefetch"] | 3 | 3 | 3 | -1199.371 | -831.744 |

| paired group | N | mean TTFT delta ms | paired bootstrap 95% CI ms |
| --- | ---: | ---: | --- |
| [8192, 3.0, 2.5, "prefetch"] | 3 | -936.125 | [-1317.805, -600.222] |
| [16384, 3.0, 2.5, "prefetch"] | 3 | -1287.566 | [-1689.149, -974.178] |

bootstrap 对完整配对差值采样；小样本、相关时序与状态变化会限制区间解释。ready 子集只描述已完成准备的样本，全部样本仍保留。

| group | ready-before-resume N / mean TTFT ms | not-ready N / mean TTFT ms | ambiguous usage N |
| --- | --- | --- | ---: |
| [16384, 3.0, 2.5, "on_demand"] | 0 / — | 0 / — | 0 |
| [16384, 3.0, 2.5, "prefetch"] | 3 / 355.542 | 0 / — | 0 |
| [8192, 3.0, 2.5, "on_demand"] | 0 / — | 0 / — | 0 |
| [8192, 3.0, 2.5, "prefetch"] | 3 / 221.875 | 0 / — | 0 |

原始请求、响应、控制状态、范围与错误均在 raw.jsonl。未观测到的物理 I/O、host 峰值和服务端 CPU 保留为空值。
