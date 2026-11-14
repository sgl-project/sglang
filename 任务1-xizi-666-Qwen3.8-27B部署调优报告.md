# Qwen3.8-27B部署调优报告

## 一、报告基本信息

| 项目 | 内容 |
|---|---|
| 报告编号 | 任务1-xizi-666-Qwen3.8-27B |
| 体验日期 | 2026-11-14 |
| SGLang镜像版本 | swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-910b-20261112 |
| 环境信息 | 昇腾A2（Ascend 910B3 × 4，单卡 64GB HBM），CANN 9.1.0，SGLang dev main 分支（/sgl-workspace/sglang） |

---

## 二、操作记录

### 使用Qwen3.8-27B权重（BF16）

#### 1. 启动命令

环境变量（参考 Qwen3.6-27B 最佳实践文档）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
export ASCEND_USE_FIA=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32
```

基线启动命令（参考最佳实践文档，按 A2 4 卡调整 tp-size）：

```bash
sglang serve \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
  --host 127.0.0.1 --port 30000 \
  --tp-size 4 \
  --attention-backend ascend --device npu \
  --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.8 \
  --max-running-requests 16 \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --mamba-radix-cache-strategy extra_buffer \
  --trust-remote-code
```

调优后启动命令（最终版）：

```bash
sglang serve \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
  --host 127.0.0.1 --port 30000 \
  --tp-size 4 \
  --attention-backend ascend --device npu \
  --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.82 \
  --max-running-requests 12 \
  --max-mamba-cache-size 160 \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --mamba-radix-cache-strategy extra_buffer \
  --cuda-graph-bs-decode 1 2 5 10 12 \
  --trust-remote-code
```

参数调整变更说明：

| 参数 | 基线 | 调优后 | 原因 |
|---|---|---|---|
| --mem-fraction-static | 0.8 | 0.82 | 增大 KV cache 池，128K 长上下文场景减少驱逐 |
| --max-running-requests | 16 | 12 | 限制解码批大小，保证 TPOT≤50ms |
| --max-mamba-cache-size | 默认 | 160 | 为混合注意力（GDN）层预留足够 mamba 状态槽位，避免状态驱逐重算 |
| --cuda-graph-bs-decode | 默认(1-16) | 1 2 5 10 12 | 覆盖实际解码批大小，避免未捕获 batch size 回退导致解码抖动 |
| 客户端并发 | 20 | 8~10 | 高并发下 TPOT 显著上升，8~10 为吞吐/TPOT 平衡点 |

另尝试过 NEXTN 投机解码（`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`，配合 `GDN_ATTN_BACKEND_TRITON=1 SGLANG_ENABLE_SPEC_V2=1`），accept length 3.45，但在并发 20 下 TPOT 从 46ms 恶化到 108ms（verify 开销随批大小放大），本场景不采用。

#### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
    --api-url http://127.0.0.1:30000/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gsm8k \
    --limit 200
```

测试结果：

```txt
┌─────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model       │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├─────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 98%     │
└─────────────┴───────────┴────────────┴──────────┴───────┴─────────┘

Model        Dataset      Num  Avg Lat    Avg TTFT    Avg TPOT    Avg Thpt       Avg In    Avg Out
-----------  ---------  -----  ---------  ----------  ----------  -----------  --------  ---------
Qwen3.8-27B  GSM8K        200  15.689 s   —           —           36.34 tok/s      729        570
```

#### 3. 性能测试

测试命令（128K-1K-90%cache：128000×90%=115200 共享前缀，128000×10%=12800 独立问题，输出 1000）：

```bash
python -m sglang.benchmark.serving \
  --dataset-name generated-shared-prefix --backend sglang \
  --host 127.0.0.1 --port 30000 \
  --gsp-num-groups 1 --gsp-prompts-per-group 20 \
  --gsp-system-prompt-len 115200 --gsp-question-len 12800 --gsp-output-len 1000 \
  --max-concurrency 8 --num-prompts 20 --request-rate inf
```

调优过程数据（同一服务实例，预热后）：

| 配置 | 客户端并发 | 输出吞吐 (tok/s) | 总吞吐 (tok/s) | Mean TPOT (ms) | P99 TPOT (ms) |
|---|---|---|---|---|---|
| 基线（max-running 16） | 20 | 101.03 | 13363.10 | 46.19 | 52.97 |
| 调优+NEXTN投机（max-running 20） | 20 | 102.09 | 13504.47 | 108.24 | 169.07 |
| 调优（max-running 20，无投机） | 20 | 177.29 | 23451.23 | 77.77 | 98.77 |
| 调优（max-running 12） | 12 | 139.10 | 18399.65 | 54.45 | 77.16 |
| **调优（max-running 12，最终）** | **8** | **174.47** | **23077.30** | **36.63** | **44.48** |
| 调优（max-running 12） | 10 | 200.35 | 26501.10 | 44.02 | 51.26 |

最终最佳结果（并发8，全分位 TPOT≤50ms）：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 8
Successful requests:                     20
Benchmark duration (s):                  114.64
Total input tokens:                      2625484
Total generated tokens:                  20000
Request throughput (req/s):              0.17
Input token throughput (tok/s):          22902.83
Output token throughput (tok/s):         174.47
Peak output token throughput (tok/s):    439.00
Total token throughput (tok/s):          23077.30
Concurrency:                             7.04
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   40349.74
Median E2E Latency (ms):                 41310.80
P99 E2E Latency (ms):                    50348.73
---------------Time to First Token----------------
Mean TTFT (ms):                          3758.96
Median TTFT (ms):                        3775.44
P99 TTFT (ms):                           7506.15
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          36.63
Median TPOT (ms):                        36.80
P90 TPOT (ms):                           44.06
P95 TPOT (ms):                           44.45
P99 TPOT (ms):                           44.48
---------------Inter-Token Latency----------------
Mean ITL (ms):                           36.63
Median ITL (ms):                         36.39
P99 ITL (ms):                            40.78
Max ITL (ms):                            6590.77
```

### 使用Qwen3.8-27B-w8a8权重（W8A8）

#### 1. 启动命令

环境变量同 BF16。启动命令（在 BF16 调优参数基础上增加 `--quantization modelslim`）：

```bash
sglang serve \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
  --host 127.0.0.1 --port 30000 \
  --tp-size 4 \
  --quantization modelslim \
  --attention-backend ascend --device npu \
  --dtype bfloat16 --mamba-ssm-dtype bfloat16 \
  --mem-fraction-static 0.82 \
  --max-running-requests 12 \
  --max-mamba-cache-size 160 \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --mamba-radix-cache-strategy extra_buffer \
  --cuda-graph-bs-decode 1 2 5 10 12 \
  --trust-remote-code
```

参数调整变更说明：与 BF16 调优版一致，仅新增 `--quantization modelslim` 加载 W8A8 量化权重（权重体积 52GB → 29GB）。

#### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --api-url http://127.0.0.1:30000/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gsm8k \
    --limit 200
```

测试结果：

```txt
┌──────────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model            │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├──────────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B-w8a8 │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 98.5%   │
└──────────────────┴───────────┴────────────┴──────────┴───────┴─────────┘

Model             Dataset      Num  Avg Lat    Avg TTFT    Avg TPOT    Avg Thpt       Avg In    Avg Out
----------------  ---------  -----  ---------  ----------  ----------  -----------  --------  ---------
Qwen3.8-27B-w8a8  GSM8K        200  15.82 s    —           —           34.23 tok/s      729        542
```

#### 3. 性能测试

测试命令同 BF16（128K-1K-90%cache 用例）。

调优过程数据：

| 配置 | 客户端并发 | 缓存状态 | 输出吞吐 (tok/s) | Mean TPOT (ms) | P99 TPOT (ms) |
|---|---|---|---|---|---|
| 调优（max-running 12） | 8 | 冷 | 122.50 | 44.91 | 55.10 |
| 调优（max-running 12） | 8 | 热 | 160.54 | 40.25 | 43.74 |
| **调优（max-running 12，最终）** | **10** | **热** | **199.43** | **44.91** | **47.83** |

最终最佳结果（并发10，全分位 TPOT≤50ms）：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 10
Successful requests:                     20
Benchmark duration (s):                  100.28
Total input tokens:                      2625484
Total generated tokens:                  20000
Request throughput (req/s):              0.20
Input token throughput (tok/s):          26180.36
Output token throughput (tok/s):         199.43
Peak output token throughput (tok/s):    644.00
Total token throughput (tok/s):          26379.79
Concurrency:                             9.81
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   49187.00
Median E2E Latency (ms):                 48449.54
P99 E2E Latency (ms):                    55206.08
---------------Time to First Token----------------
Mean TTFT (ms):                          4325.63
Median TTFT (ms):                        4472.62
P99 TTFT (ms):                           7422.20
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          44.91
Median TPOT (ms):                        45.35
P90 TPOT (ms):                           47.20
P95 TPOT (ms):                           47.45
P99 TPOT (ms):                           47.83
---------------Inter-Token Latency----------------
Mean ITL (ms):                           44.91
Median ITL (ms):                         44.96
P99 ITL (ms):                            50.08
Max ITL (ms):                            758.60
```

### 对比总结

| 指标 | Qwen3.8-27B (BF16) | Qwen3.8-27B-w8a8 (W8A8) |
|---|---|---|
| 权重体积 | 52 GB | 29 GB（-44%） |
| GSM8K 精度（200条） | 98% | 98.5%（无损失，误差范围内） |
| 最佳输出吞吐（TPOT≤50ms） | 174.47 tok/s @并发8 | 199.43 tok/s @并发10（+14%） |
| 最佳总吞吐 | 23077.30 tok/s | 26379.79 tok/s |
| Mean TPOT | 36.63 ms | 44.91 ms |
| P99 TPOT | 44.48 ms | 47.83 ms |
| Mean TTFT（热缓存） | 3758.96 ms | 4325.63 ms |

结论：

1. W8A8（modelslim）量化在 GSM8K 上无精度损失（98.5% vs 98%），权重体积减少约 44%，显著降低显存占用与加载时间。
2. 在 TPOT≤50ms 约束下，W8A8 可支撑更高并发（10 vs 8），输出吞吐比 BF16 高约 14%；BF16 在低并发下单 token 延迟更低（36.6ms vs 44.9ms）。
3. 128K-1K-90%cache 场景下，90% 共享前缀可被 radix cache 完全复用（热缓存后 TTFT 从 ~12s 降至 3-4s），性能瓶颈在独立 10% 问题的 prefill 与解码批调度。
4. 调优要点：限制解码批大小（max-running-requests 12 + 客户端并发 8~10）是保住 TPOT≤50ms 的关键；cuda-graph-bs-decode 需覆盖实际解码批；NEXTN 投机解码在高并发下反而恶化 TPOT，不适用于本场景。

---

## 三、参考文档评价

| 评价维度 | 评分（1~5） | 说明 |
|---|---|---|
| 安装部署文档（Qwen3.6-27B教程） | 5 | 步骤清晰，Qwen3.8-27B 可直接参考跑通 |
| 最佳实践文档 | 4 | 参数完备可复现，但参数面向 A3（tp-size 2），A2 4 卡需自行换算，投机解码参数未标注适用并发范围 |
| 精度测试文档 | 4 | evalscope 流程顺畅；任务书示例端口(6688)与最佳实践默认端口(30000)不一致，新手易漏改 |
| 性能测试文档 | 4 | gsp 用例说明清楚；建议补充 128K 用例的具体参数换算示例（115200/12800） |
| 参数调优文档 | 4 | 调优路径（profiling→参数→验证）清楚；建议补充"并发-TPOT"权衡的量化示例 |

---

## 四、问题与改进建议

### 发现的问题

1. 最佳实践文档的启动参数未标注适用的并发范围：服务端/客户端并发从 8 提升到 20 时，TPOT 从 36.6ms 恶化到 77.8ms，超出 TPOT≤50ms 的 SLA 场景容易踩坑。
2. NEXTN 投机解码在 A2 910B3×4、并发 20 场景下 TPOT 从 46ms 恶化到 108ms（accept length 3.45 但 verify 开销随批大小放大），文档未说明投机解码的适用硬件/并发边界。
3. 任务书精度测试示例使用端口 6688，与最佳实践文档默认端口 30000 不一致，直接复制命令会连接失败。
4. 容器环境 LD_LIBRARY_PATH 中若存在旧版 libsqlite3（<3.7），会导致 python sqlite3/torch_npu 导入失败，报错信息不直观（本环境已通过将 /usr/lib/aarch64-linux-gnu 前置解决）。

### 改进建议

1. 最佳实践文档按硬件（A2/A3）分别给出推荐参数，并补充"并发-TPOT-吞吐"实测曲线，帮助开发者按 SLA 选择并发。
2. 参数调优文档增加投机解码适用性说明：低并发受益、高并发（≥16）可能因 verify 开销恶化 TPOT。
3. 统一任务书与最佳实践文档的示例端口，或在任务书中注明"端口需与实际启动参数一致"。
4. 性能测试文档补充 128K-1K-90%cache 用例的完整命令示例（gsp-system-prompt-len 115200 / gsp-question-len 12800 / gsp-output-len 1000）。

> （注：部分内容由豆包 AI 生成）
