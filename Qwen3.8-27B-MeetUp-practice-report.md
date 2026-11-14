# Qwen3.8-27B部署调优报告

## 一、报告基本信息

| 项目 | 内容 |
|---|---|
| 报告编号 | 【待填】 |
| 体验日期 | 2026-11-14 |
| SGLang镜像版本 | swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-910b-20261112 |
| 环境信息 | A2（910B3 × 4卡，TP=4） |
| SGLang版本 | 0.0.0.dev22121+g0fa9ff55f.d20261112 |

---

## 二、操作记录

### 使用Qwen3.8-27B权重（BF16）

#### 1. 启动命令

```bash
# 系统配置（参考最佳实践文档）
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export ASCEND_USE_FIA=1
export GDN_ATTN_BACKEND_TRITON=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

python3 -m sglang.launch_server \
    --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
    --host 127.0.0.1 --port 6688 \
    --tp-size 4 --nnodes 1 \
    --device npu --attention-backend ascend \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
    --mamba-radix-cache-strategy extra_buffer \
    --trust-remote-code \
    --max-running-requests 20 \
    --max-mamba-cache-size 160 \
    --mem-fraction-static 0.82 \
    --cuda-graph-bs-decode 1 2 4 6 8 10 11 12 13 14 15 16 17 18 19 20 \
    --dtype bfloat16 \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3
```

**参数调整变更说明**（相对默认参数）：

| 参数 | 默认 | 调整后 | 调整原因 |
|---|---|---|---|
| `--mamba-ssm-dtype` | float32 | **bfloat16** | 【必须】910B 的 `chunk_gated_delta_rule` NPU 算子不支持 FP32 初始状态（报错 `FP32 initial_state is only supported on Ascend950 (arch35)`），FP32 会导致服务在首个请求崩溃 |
| `--device npu --attention-backend ascend` | 自动 | ascend | 按 NPU 最佳实践显式指定，配合 `ASCEND_USE_FIA=1`、`GDN_ATTN_BACKEND_TRITON=1` 使用 FIA/Triton GDN kernel |
| `--chunked-prefill-size` | 8192 | 32768 | 减少 128K 长 prefill 对 decode 的干扰次数（每个 chunk 会阻塞 decode 一次），TPOT 改善的主要来源之一 |
| `--mamba-radix-cache-strategy` | 无 | extra_buffer | 启用 GDN/线性注意力状态的 radix 缓存，是 90% 前缀缓存命中的前提 |
| `--cuda-graph-bs-decode` | 自动 | 显式枚举 1~20 | 补全列表避免解码图 pad 到更大 batch size 浪费算力 |
| `--max-running-requests` | 156 | 20 | 匹配 90%cache 用例的并发水位，配合 `--max-mamba-cache-size 160` |
| `--mem-fraction-static` | 默认 | 0.82 | 按 best-practice 固定显存配比 |

#### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
    --api-url http://localhost:6688/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gsm8k \
    --limit 200 \
    --generation-config 'reasoning_effort=none,temperature=0,max_tokens=1024'
```

> 注：相对任务书脚本增加 `--generation-config`。Qwen3.8 默认开启 thinking 模式，关闭思考并使用贪心解码（temperature=0）可避免 `<think>` 内容干扰答案抽取，并保证结果可复现。

测试结果：

```txt
gsm8k report table:
┌─────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model       │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├─────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 96.5%   │
└─────────────┴───────────┴────────────┴──────────┴───────┴─────────┘
```

#### 3. 性能测试

测试命令（128K-1K-90%cache，GSP 数据集，1 个共享前缀组）：

```bash
python -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
    --gsp-num-groups 1 --gsp-prompts-per-group 10 \
    --gsp-system-prompt-len 117965 \
    --gsp-question-len 13107 \
    --gsp-output-len 1000 \
    --max-concurrency 10 --num-prompts 10 --request-rate inf
```

> 数据集构造：input = 117965（90% 共享前缀）+ 13107（10% 独有后缀）= 131072 tokens，output 1024，即 128K-1K-90%cache 用例。并发 10 为 TPOT≤50ms 约束下的最优并发（实测：并发 12 → TPOT 52.46ms 超标；并发 10 → TPOT 42.52ms 达标）。

测试结果：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 10
Successful requests:                     10
Benchmark duration (s):                  53.79
Total input tokens:                      1344191
Total generated tokens:                  10000
Request throughput (req/s):              0.19
Input token throughput (tok/s):          24990.83
Output token throughput (tok/s):         185.92
Peak output token throughput (tok/s):    250.00
Total token throughput (tok/s):          25176.75
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   53294.10
---------------Time to First Token----------------
Mean TTFT (ms):                          10811.84
Median TTFT (ms):                        13038.32
P99 TTFT (ms):                           13116.04
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.52
Median TPOT (ms):                        40.71
P90 TPOT (ms):                           46.28
P95 TPOT (ms):                           47.79
P99 TPOT (ms):                           48.99
---------------Inter-Token Latency----------------
Mean ITL (ms):                           42.54
Median ITL (ms):                         40.85
==================================================
```

**TPOT=50ms 场景结论：Mean TPOT 42.52ms（P99 48.99ms）达标，输出吞吐 185.92 tok/s。**

### 使用Qwen3.8-27B-w8a8权重

#### 1. 启动命令

与 BF16 版本仅两处差异：模型路径与量化参数。

```bash
# 环境变量与 BF16 完全一致（略），启动命令：
python3 -m sglang.launch_server \
    --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --host 127.0.0.1 --port 6688 \
    --tp-size 4 --nnodes 1 \
    --device npu --attention-backend ascend \
    --quantization modelslim \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
    --mamba-radix-cache-strategy extra_buffer \
    --trust-remote-code \
    --max-running-requests 20 \
    --max-mamba-cache-size 160 \
    --mem-fraction-static 0.82 \
    --cuda-graph-bs-decode 1 2 4 6 8 10 11 12 13 14 15 16 17 18 19 20 \
    --dtype bfloat16 \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3
```

#### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --api-url http://localhost:6688/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gsm8k \
    --limit 200 \
    --generation-config 'reasoning_effort=none,temperature=0,max_tokens=1024'
```

测试结果：

```txt
gsm8k report table:
┌──────────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model            │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├──────────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B-w8a8 │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 95.5%   │
└──────────────────┴───────────┴────────────┴──────────┴───────┴─────────┘
```

#### 3. 性能测试

测试命令（与 BF16 相同，仅 --model 换为 w8a8 权重路径，并发 10）：

```bash
python -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --gsp-num-groups 1 --gsp-prompts-per-group 10 \
    --gsp-system-prompt-len 117965 \
    --gsp-question-len 13107 \
    --gsp-output-len 1000 \
    --max-concurrency 10 --num-prompts 10 --request-rate inf
```

测试结果：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 10
Successful requests:                     10
Benchmark duration (s):                  55.01
Total input tokens:                      1344191
Total generated tokens:                  10000
Request throughput (req/s):              0.18
Input token throughput (tok/s):          24434.98
Output token throughput (tok/s):         181.78
Peak output token throughput (tok/s):    220.00
Total token throughput (tok/s):          24616.76
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   54912.74
---------------Time to First Token----------------
Mean TTFT (ms):                          8992.16
Median TTFT (ms):                        9165.78
P99 TTFT (ms):                           9210.63
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.97
Median TPOT (ms):                        45.84
P90 TPOT (ms):                           46.00
P95 TPOT (ms):                           46.64
P99 TPOT (ms):                           47.15
---------------Inter-Token Latency----------------
Mean ITL (ms):                           45.99
Median ITL (ms):                         45.77
==================================================
```

**TPOT=50ms 场景结论：Mean TPOT 45.97ms（P99 47.15ms）达标，输出吞吐 181.78 tok/s。**

### 对比总结

| 指标 | Qwen3.8-27B (BF16) | Qwen3.8-27B-w8a8 | 对比 |
|---|---|---|---|
| GSM8K 精度（200 条） | **96.5%** | 95.5% | w8a8 精度损失 1.0pp |
| Mean TPOT | **42.52 ms** | 45.97 ms | 两者均满足 TPOT≤50ms |
| TPOT P99 | 48.99 ms | **47.15 ms** | w8a8 尾部更平稳（P90~P99 与 Mean 几乎一致） |
| Mean TTFT | 10811 ms（P99 13116 ms） | **8992 ms（P99 9211 ms）** | w8a8 prefill 提速约 20% |
| 输出吞吐 | **185.92 tok/s** | 181.78 tok/s | BF16 略优 2.3% |
| 输入吞吐 | **24991 tok/s** | 24435 tok/s | 基本持平 |
| 基准总耗时 | **53.79 s** | 55.01 s | 基本持平 |
| TPOT≤50ms 最优并发 | 10 | 10 | 并发 12 时两者均超标（52.1~52.5ms） |

**结论：**

1. **精度**：W8A8 量化损失仅 1.0pp（96.5%→95.5%），模型量化质量良好。
2. **性能**：在本场景（GDN 混合架构 + 90% 前缀缓存命中 + 910B3）下，W8A8 **未获得端到端性能收益**：decode TPOT 反而比 BF16 慢 3.5ms，仅 TTFT（prefill）提速约 20%。推测原因：decode 阶段 GDN/triton kernel 路径的 int8 反量化开销抵消了访存收益；BF16 与 W8A8 的 decode 都受 ITL 与并发近似线性关系的约束。
3. **价值判断**：W8A8 的收益主要在显存占用减半（权重 27B：BF16 约 54GB → INT8 约 27GB），可显著提升单机 KV cache 容量与部署密度；但纯 decode 密集场景下当前 kernel 实现未能兑现理论收益，建议推动 NPU 侧 W8A8 decode 算子优化（Issue 素材）。
4. **调优方法论**：本场景 TPOT 瓶颈 = 稳态解码 ITL（随并发线性上升）+ 长 prefill 干扰。前者通过选择并发（10）控制，后者通过 `chunked-prefill-size 32768` 与 90% 缓存命中消除。默认参数下（chunked-prefill 8192、无 GDN 缓存、无 NPU 环境变量），Mean TPOT 81.23ms、总耗时 929.7s；调优后 42.52ms、53.8s。

---

## 三、参考文档评价

| 评价维度 | 评分（1~5） | 说明 |
|---|---|---|
| 安装部署文档 | 4 | Qwen3.6-27B 教程可完整迁移，环境准备/权重路径清晰；但未覆盖 Qwen3.8（GDN 混合架构）与 910B 的 `--mamba-ssm-dtype` 兼容性问题，默认参数直接部署会崩溃 |
| 最佳实践文档 | 4 | 参数配置完整可复用（含环境变量），GSP benchmark 方法可直接套用；但仅覆盖 A3 单卡场景，A2 多卡场景需自行换算 TP/显存参数 |
| 参数调优文档 | 4 | 调优方向（chunked prefill、缓存策略、并发水位）与实测一致；缺少 TPOT 约束场景的并发选择方法论 |
| 精度测试文档 | 5 | evalscope 流程顺畅，任务书脚本可直接使用 |
| 性能测试文档 | 5 | bench_serving GSP 数据集参数文档清晰，90% cache 用例构造方式明确 |
| Profiling 文档 | 5 | 未深度使用，暂无问题 |

---

## 四、问题与改进建议

### 发现的问题

1. **【P0】Qwen3.8-27B（GDN 混合架构）在 910B 上默认参数启动后首个请求必崩**：`RuntimeError: FP32 initial_state is only supported on Ascend950 (arch35)`。根因是 `--mamba-ssm-dtype` 默认 float32，而 910B 的 `torch.ops.npu.chunk_gated_delta_rule` 算子仅支持 BF16 初始状态。该参数在任何 Qwen3.5/3.8 系列的 NPU 部署文档中均未提及。
2. **【P1】容器 NPU 逻辑卡号与物理卡号不一致时文档无指引**：物理 4-7 号卡映射为容器内 0-3 号，`ASCEND_RT_VISIBLE_DEVICES` 使用物理卡号会导致 `NPU device is not available or visible`，排查耗时较长。
3. **【P2】W8A8 在 GDN 混合模型上无 decode 收益**：TPOT 比 BF16 慢 3.5ms，量化收益仅在 prefill（TTFT -20%），int8 decode kernel 与 GDN kernel 组合存在优化空间。

### 改进建议

1. 在 Qwen3.6/3.8 NPU 部署文档与启动参数校验中显式加入 `--mamba-ssm-dtype bfloat16`（或让 sglang 在 910B + GDN 模型上自动检测并降级/告警），并提交 Issue 说明。
2. 最佳实践文档补充 A2 多卡（TP4）场景的推荐配置，并说明容器卡号映射问题的排查方法（`ls /dev/davinci*` + `npu-smi info` 对照）。
3. 参数调优文档增加「TPOT 约束 → 并发选择」的操作指引：先测稳态 ITL-并发曲线（如本报告 bs 12→52ms、bs 10→42ms），再按约束反推最优并发。
4. W8A8 权重建议配套提供 NextN/MTP 投机解码权重（当前 config 无 nextn 层），投机解码是降 TPOT 最直接的手段。

> （注：部分内容由豆包 AI 生成）

| 参数 | 默认 | 调整后 | 调整原因 |
|---|---|---|---|
| `--mamba-ssm-dtype` | float32 | **bfloat16** | 【必须】910B 的 `chunk_gated_delta_rule` NPU 算子不支持 FP32 初始状态（报错 `FP32 initial_state is only supported on Ascend950 (arch35)`），FP32 会导致服务在首个请求崩溃 |
| `--device npu --attention-backend ascend` | 自动 | ascend | 按 NPU 最佳实践显式指定，配合 `ASCEND_USE_FIA=1`、`GDN_ATTN_BACKEND_TRITON=1` 使用 FIA/Triton GDN kernel |
| `--chunked-prefill-size` | 8192 | 32768 | 减少 128K 长 prefill 对 decode 的干扰次数（每个 chunk 会阻塞 decode 一次），TPOT 改善的主要来源之一 |
| `--mamba-radix-cache-strategy` | 无 | extra_buffer | 启用 GDN/线性注意力状态的 radix 缓存，是 90% 前缀缓存命中的前提 |
| `--cuda-graph-bs-decode` | 自动 | 显式枚举 1~20 | 补全列表避免解码图 pad 到更大 batch size 浪费算力 |
| `--max-running-requests` | 156 | 20 | 匹配 90%cache 用例的并发水位，配合 `--max-mamba-cache-size 160` |
| `--mem-fraction-static` | 默认 | 0.82 | 按 best-practice 固定显存配比 |

#### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
    --api-url http://localhost:6688/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gsm8k \
    --limit 200 \
    --generation-config 'reasoning_effort=none,temperature=0,max_tokens=1024'
```

> 注：相对任务书脚本增加 `--generation-config`。Qwen3.8 默认开启 thinking 模式，关闭思考并使用贪心解码（temperature=0）可避免 `<think>` 内容干扰答案抽取，并保证结果可复现。

测试结果：

```txt
gsm8k report table:
┌─────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model       │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├─────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 96.5%   │
└─────────────┴───────────┴────────────┴──────────┴───────┴─────────┘
```

#### 3. 性能测试

测试命令（128K-1K-90%cache，GSP 数据集，1 个共享前缀组）：

```bash
python -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
    --gsp-num-groups 1 --gsp-prompts-per-group 10 \
    --gsp-system-prompt-len 117965 \
    --gsp-question-len 13107 \
    --gsp-output-len 1000 \
    --max-concurrency 10 --num-prompts 10 --request-rate inf
```

> 数据集构造：input = 117965（90% 共享前缀）+ 13107（10% 独有后缀）= 131072 tokens，output 1024，即 128K-1K-90%cache 用例。并发 10 为 TPOT≤50ms 约束下的最优并发（实测：并发 12 → TPOT 52.46ms 超标；并发 10 → TPOT 42.52ms 达标）。

测试结果：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 10
Successful requests:                     10
Benchmark duration (s):                  53.79
Total input tokens:                      1344191
Total generated tokens:                  10000
Request throughput (req/s):              0.19
Input token throughput (tok/s):          24990.83
Output token throughput (tok/s):         185.92
Peak output token throughput (tok/s):    250.00
Total token throughput (tok/s):          25176.75
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   53294.10
---------------Time to First Token----------------
Mean TTFT (ms):                          10811.84
Median TTFT (ms):                        13038.32
P99 TTFT (ms):                           13116.04
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          42.52
Median TPOT (ms):                        40.71
P90 TPOT (ms):                           46.28
P95 TPOT (ms):                           47.79
P99 TPOT (ms):                           48.99
---------------Inter-Token Latency----------------
Mean ITL (ms):                           42.54
Median ITL (ms):                         40.85
==================================================
```

**TPOT=50ms 场景结论：Mean TPOT 42.52ms（P99 48.99ms）达标，输出吞吐 185.92 tok/s。**

### 使用Qwen3.8-27B-w8a8权重

#### 1. 启动命令

与 BF16 版本仅两处差异：模型路径与量化参数。

```bash
# 环境变量与 BF16 完全一致（略），启动命令：
python3 -m sglang.launch_server \
    --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --host 127.0.0.1 --port 6688 \
    --tp-size 4 --nnodes 1 \
    --device npu --attention-backend ascend \
    --quantization modelslim \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
    --mamba-radix-cache-strategy extra_buffer \
    --trust-remote-code \
    --max-running-requests 20 \
    --max-mamba-cache-size 160 \
    --mem-fraction-static 0.82 \
    --cuda-graph-bs-decode 1 2 4 6 8 10 11 12 13 14 15 16 17 18 19 20 \
    --dtype bfloat16 \
    --mamba-ssm-dtype bfloat16 \
    --reasoning-parser qwen3
```

#### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --api-url http://localhost:6688/v1 \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets gsm8k \
    --limit 200 \
    --generation-config 'reasoning_effort=none,temperature=0,max_tokens=1024'
```

测试结果：

```txt
gsm8k report table:
┌──────────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model            │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├──────────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B-w8a8 │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 95.5%   │
└──────────────────┴───────────┴────────────┴──────────┴───────┴─────────┘
```

#### 3. 性能测试

测试命令（与 BF16 相同，仅 --model 换为 w8a8 权重路径，并发 10）：

```bash
python -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix --backend sglang \
    --host 127.0.0.1 --port 6688 \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --gsp-num-groups 1 --gsp-prompts-per-group 10 \
    --gsp-system-prompt-len 117965 \
    --gsp-question-len 13107 \
    --gsp-output-len 1000 \
    --max-concurrency 10 --num-prompts 10 --request-rate inf
```

测试结果：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 10
Successful requests:                     10
Benchmark duration (s):                  55.01
Total input tokens:                      1344191
Total generated tokens:                  10000
Request throughput (req/s):              0.18
Input token throughput (tok/s):          24434.98
Output token throughput (tok/s):         181.78
Peak output token throughput (tok/s):    220.00
Total token throughput (tok/s):          24616.76
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   54912.74
---------------Time to First Token----------------
Mean TTFT (ms):                          8992.16
Median TTFT (ms):                        9165.78
P99 TTFT (ms):                           9210.63
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          45.97
Median TPOT (ms):                        45.84
P90 TPOT (ms):                           46.00
P95 TPOT (ms):                           46.64
P99 TPOT (ms):                           47.15
---------------Inter-Token Latency----------------
Mean ITL (ms):                           45.99
Median ITL (ms):                         45.77
==================================================
```

**TPOT=50ms 场景结论：Mean TPOT 45.97ms（P99 47.15ms）达标，输出吞吐 181.78 tok/s。**

### 对比总结

| 指标 | Qwen3.8-27B (BF16) | Qwen3.8-27B-w8a8 | 对比 |
|---|---|---|---|
| GSM8K 精度（200 条） | **96.5%** | 95.5% | w8a8 精度损失 1.0pp |
| Mean TPOT | **42.52 ms** | 45.97 ms | 两者均满足 TPOT≤50ms |
| TPOT P99 | 48.99 ms | **47.15 ms** | w8a8 尾部更平稳（P90~P99 与 Mean 几乎一致） |
| Mean TTFT | 10811 ms（P99 13116 ms） | **8992 ms（P99 9211 ms）** | w8a8 prefill 提速约 20% |
| 输出吞吐 | **185.92 tok/s** | 181.78 tok/s | BF16 略优 2.3% |
| 输入吞吐 | **24991 tok/s** | 24435 tok/s | 基本持平 |
| 基准总耗时 | **53.79 s** | 55.01 s | 基本持平 |
| TPOT≤50ms 最优并发 | 10 | 10 | 并发 12 时两者均超标（52.1~52.5ms） |

**结论：**

1. **精度**：W8A8 量化损失仅 1.0pp（96.5%→95.5%），模型量化质量良好。
2. **性能**：在本场景（GDN 混合架构 + 90% 前缀缓存命中 + 910B3）下，W8A8 **未获得端到端性能收益**：decode TPOT 反而比 BF16 慢 3.5ms，仅 TTFT（prefill）提速约 20%。推测原因：decode 阶段 GDN/triton kernel 路径的 int8 反量化开销抵消了访存收益；BF16 与 W8A8 的 decode 都受 ITL 与并发近似线性关系的约束。
3. **价值判断**：W8A8 的收益主要在显存占用减半（权重 27B：BF16 约 54GB → INT8 约 27GB），可显著提升单机 KV cache 容量与部署密度；但纯 decode 密集场景下当前 kernel 实现未能兑现理论收益，建议推动 NPU 侧 W8A8 decode 算子优化（Issue 素材）。
4. **调优方法论**：本场景 TPOT 瓶颈 = 稳态解码 ITL（随并发线性上升）+ 长 prefill 干扰。前者通过选择并发（10）控制，后者通过 `chunked-prefill-size 32768` 与 90% 缓存命中消除。默认参数下（chunked-prefill 8192、无 GDN 缓存、无 NPU 环境变量），Mean TPOT 81.23ms、总耗时 929.7s；调优后 42.52ms、53.8s。

---

## 三、参考文档评价

| 评价维度 | 评分（1~5） | 说明 |
|---|---|---|
| 安装部署文档 | 4 | Qwen3.6-27B 教程可完整迁移，环境准备/权重路径清晰；但未覆盖 Qwen3.8（GDN 混合架构）与 910B 的 `--mamba-ssm-dtype` 兼容性问题，默认参数直接部署会崩溃 |
| 最佳实践文档 | 4 | 参数配置完整可复用（含环境变量），GSP benchmark 方法可直接套用；但仅覆盖 A3 单卡场景，A2 多卡场景需自行换算 TP/显存参数 |
| 参数调优文档 | 4 | 调优方向（chunked prefill、缓存策略、并发水位）与实测一致；缺少 TPOT 约束场景的并发选择方法论 |
| 精度测试文档 | 5 | evalscope 流程顺畅，任务书脚本可直接使用 |
| 性能测试文档 | 5 | bench_serving GSP 数据集参数文档清晰，90% cache 用例构造方式明确 |
| Profiling 文档 | 5 | 未深度使用，暂无问题 |

---

## 四、问题与改进建议

### 发现的问题

1. **【P0】Qwen3.8-27B（GDN 混合架构）在 910B 上默认参数启动后首个请求必崩**：`RuntimeError: FP32 initial_state is only supported on Ascend950 (arch35)`。根因是 `--mamba-ssm-dtype` 默认 float32，而 910B 的 `torch.ops.npu.chunk_gated_delta_rule` 算子仅支持 BF16 初始状态。该参数在任何 Qwen3.5/3.8 系列的 NPU 部署文档中均未提及。
2. **【P1】容器 NPU 逻辑卡号与物理卡号不一致时文档无指引**：物理 4-7 号卡映射为容器内 0-3 号，`ASCEND_RT_VISIBLE_DEVICES` 使用物理卡号会导致 `NPU device is not available or visible`，排查耗时较长。
3. **【P2】W8A8 在 GDN 混合模型上无 decode 收益**：TPOT 比 BF16 慢 3.5ms，量化收益仅在 prefill（TTFT -20%），int8 decode kernel 与 GDN kernel 组合存在优化空间。

### 改进建议

1. 在 Qwen3.6/3.8 NPU 部署文档与启动参数校验中显式加入 `--mamba-ssm-dtype bfloat16`（或让 sglang 在 910B + GDN 模型上自动检测并降级/告警），并提交 Issue 说明。
2. 最佳实践文档补充 A2 多卡（TP4）场景的推荐配置，并说明容器卡号映射问题的排查方法（`ls /dev/davinci*` + `npu-smi info` 对照）。
3. 参数调优文档增加「TPOT 约束 → 并发选择」的操作指引：先测稳态 ITL-并发曲线（如本报告 bs 12→52ms、bs 10→42ms），再按约束反推最优并发。
4. W8A8 权重建议配套提供 NextN/MTP 投机解码权重（当前 config 无 nextn 层），投机解码是降 TPOT 最直接的手段。

> （注：部分内容由豆包 AI 生成）
