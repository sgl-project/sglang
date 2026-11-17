# Qwen3.5-4B Mamba HiCache 独有状态回迁：100 题报告

## 结论

**通过。** 100 道题的重放均从 Host 命中各自独有的 Mamba seed checkpoint，
`device=0`，`pool="mamba"` H2D 计数逐题增长，数值答案与首轮一致。
本用例带唯一数字填充与题间缓存清空，验证回迁功能；其 91% 正确率
**不是**标准 GSM8K 精度结论。

## 环境与运行

| 项目 | 值 |
| --- | --- |
| 日期 | 2026-11-16 |
| 服务器 / 容器 | `113.46.21.151` / `sglang-lzx-B103` |
| 模型 | `/mnt/paas/weights/Qwen3.5-4B` |
| NPU / TP | 物理卡 `4,5,6,7` / TP=4 |
| SGLang | `/home/l00889655/sglang/sglang`；该副本没有 Git 元数据，结果记录为 `unknown`；本地修改基于 `mha_op` 的 `9d5d86d5b1026921a4a25b6be3b8179846b1b678` |
| sgl-kernel-npu | `7977a2e176148a45d22d8952bb4866df1ec69ae3`；加载 `build/torch/package` 构建包 |
| HiCache | `kernel_ascend`、`write_through`、`acl` Host 内存；服务日志中 4 个 TP 进程均报告 `kv_split (2 components)` |
| 参数 | 100 题、8-shot、`UNIQUE_PAD_TOKENS=640`、`PAGE_SIZE=128`、`MAX_TOTAL_TOKENS=8192`、`HICACHE_RATIO=8`、每个 filler 4096 token |
| 用例脚本 SHA-256 | `294e6f01f9563d77a857401b57b2d5d015ec930dbe987c9c18c171807f270486`；本地与 151 上运行副本一致 |

运行方式：`TEST_CASE=stress GSM8K_NUM_QUESTIONS=100 NPU_DEVICES=4,5,6,7`
调用 `run_gsm8k.sh`；151 上显式设置 `PYTHONPATH` 优先使用上述
`build/torch/package`，使 GDN 和 `kv_exchange` 算子都已注册。

## 结果

| 检查项 | 实测 |
| --- | ---: |
| 首轮 / 重放正确率 | 91/100 / 91/100 |
| 重放数值答案与首轮一致 | 100/100 |
| 重放输出 token ID 完全一致 | 100/100 |
| Host 命中达到该题独有 seed checkpoint | 100/100 |
| Device 命中 | 0/100 |
| Mamba H2D 增长 | 100/100；每题指标增量 8，合计 800 |
| 题间 `/flush_cache` 成功 | 99 次 |
| filler 请求 | 200 次；总输入 819200 token |

共享前缀的页对齐边界为 1280 token，Mamba checkpoint 格点为 256 token。
79 题的 `seed_len=host=5376`，21 题的 `seed_len=host=5632`；
因此每题都至少回迁了 4096 个 token 的独有前缀。
服务端 TP0 的匹配日志有 100 次 `mamba_host=1`，与逐题 Host 命中及
Mamba H2D 增长相互印证。上述 H2D 指标是 4 个 TP rank 的计数之和；
各 rank 报告相同的逻辑数量，不能把 800 当成去重后的状态槽数。

首轮答错的题号（从 0 起）为 `3,12,21,58,62,70,75,87,89`；
这些题重放后的数值答案也与各自首轮一致。

## 证据与范围

- 151 容器内：`/tmp/mamba_unique_flush_100_151_20261116/{result.json,eval.log,server.log}`。
- `result.json` SHA-256：`3bf79ae0dbd633e8c7ea04e4049b7e2404fa974bb00b73d2b0956ecd26fa78a5`。
- `eval.log` SHA-256：`4e70cd6908975a8fa5dfde024a1312eb538acbfff87a6fb6895642fb52c67a8a`。
- 同目录 `TEST_REPORT_2026-11-16_151_UNIQUE_HOST_100_ROWS.json` 保存 100 题的 seed 深度、Host/Device 命中、Mamba H2D 增量与答案核对结果，便于离线审计；原始输出仍以远端 `result.json` 为准。
- 用例对每题执行 A0 seed → 首轮作答 → filler 驱逐并确认 D2H → 同一 token ID 重放并确认 Host H2D；仅在两道题之间清空缓存，避免前一题的 Device 残留影响下一题 seed。
- 这一结果证明上述隔离流程中的独有 Mamba 状态回迁；不代表连续共享缓存、多并发请求或标准 prompt 的精度表现。
