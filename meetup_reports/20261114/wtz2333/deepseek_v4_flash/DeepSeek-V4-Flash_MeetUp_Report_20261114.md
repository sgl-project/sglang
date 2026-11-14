# DeepSeek-V4-Flash部署调优报告

保密：仅用于 SGLang NPU 众测文档优化、任务验收和研发复盘，未经授权不公开众测数据及日志。

在 4 卡 950PR/DT 系列 NPU 上，使用 SGLang 部署 DeepSeek-V4-Flash-0731，启用前缀缓存后，在名义 128K 输入、1K 输出、90% 共享前缀用例中，取得 Mean TPOT 9.45ms、输出吞吐 644.41 tokens/s、实际缓存命中率 89.1%。当前配置的 GSM8K 10 条评测得分为 100.00%，无截断、无请求错误。

按 Mean TPOT 口径，当前配置同时满足 50ms 和 20ms 两档性能门槛；P99 TPOT 为 26.97ms，满足 50ms，尚不满足 20ms。任务书未明确 TPOT 采用平均值还是分位数，本报告完整披露两种结果，最终以活动统一验收口径为准。

## 一、报告基本信息

| 项目 | 内容 |
|---|---|
| 报告编号 | DSV4F-NPU-20261114-01（本报告自编号） |
| 体验日期 | 2026-11-14 |
| SGLang镜像版本 | swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.1.0-950-B105（任务指定完整 tag；实际镜像 ID 和 digest 随环境归档） |
| 环境信息 | 950PR/DT 系列（A5），4 卡；具体 PR/DT 型号以 npu-smi 归档为准 |
| 执行环境 | Docker 容器；设备为 NPU，attention backend 为 dsv4 |
| 对应仓库 | https://github.com/sgl-project/sglang ，任务关联 main 分支；运行时 commit SHA 待环境归档 |
| 模型路径 | /workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731 |
| 服务地址 | http://127.0.0.1:6688 |
| 并行配置 | TP=4；启用 DP Attention，DP=4；启用 DP LM Head |
| 投机解码 | DSPARK；draft tokens=6，block size=5 |
| 最终配置编号 | C1：启用前缀缓存，chunked-prefill-size=65536，其余参数见完整启动命令 |
| 精度记录 A1 | GSM8K，10 条，single-shot，max-tokens=2048；run 目录末尾为 20261114-025657 |
| 性能记录 P1 | generated-shared-prefix，64 条，并发上限 16，预热 8 条，seed=42 |
| 当前结果范围 | 已测配置中输出吞吐最高的一组记录；未进行全参数空间搜索，不主张全局最优 |

### 验收指标判定

| 指标及口径 | 实测值 | 50ms 场景 | 20ms 场景 |
|---|---:|---|---|
| Mean TPOT | 9.45ms | 满足 | 满足 |
| P99 TPOT | 26.97ms | 满足 | 未满足 |
| GSM8K 本轮正确率 | 10/10，100.00% | 同一最终配置评测结果 | 同一最终配置评测结果 |
| 截断率 / 请求错误率 | 0.00% / 0.00% | 无截断、无请求错误 | 无截断、无请求错误 |
| 共享前缀构造比例 | 115200 / 128000 = 90% | 名义用例已构造 | 复用同一用例 |
| 实际缓存命中率 | 89.1% | 如实呈报 | 如实呈报 |

### 用例定义与计量说明

本次采用十进制目标长度：共享前缀 115200 tokens，独立问题后缀 12800 tokens，名义合计 128000 tokens，每条目标输出 1000 tokens。该定义对应本报告所称的 128K-1K-90% 共享前缀用例；如果活动统一要求 131072 / 1024，应按对应标准复测。

服务端实测平均输入长度为 8373572 / 64 = 130837.06 tokens，高于名义目标 128000。因此，报告区分生成参数中的目标长度与服务端实际计数，不将本轮表述为严格固定 128000 tokens 的实测。文本生成后重新分词、特殊 token 等可能带来偏差，具体原因需结合生成器与 tokenizer 核验。

实际缓存命中率为 7458816 / 8373572 = 89.0757%，控制台四舍五入显示 89.1%。其中 Device 100.0% 表示已命中缓存全部来自设备内存，不表示全部输入 100% 命中。输入吞吐包含缓存命中的逻辑输入 token，不能直接等同于未缓存 prefill 的实际计算吞吐。

## 二、操作记录

### TPOT=50ms 场景

本场景采用最终配置 C1。精度引用当前配置的同一次 GSM8K 评测 A1，性能引用同一次压测 P1。50ms 与 20ms 两节复用同一配置和同一组实测数据，分别与各自阈值比较；不计作两次独立实验。

#### 1. 启动命令

完整环境变量与启动命令如下；前缀缓存保持启用，未传入 `--disable-radix-cache`。

```bash
MODEL_PATH=/workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/:$ASCEND_CUSTOM_OPP_PATH
export DEEPEP_HCCL_BUFFSIZE=1600
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export FORCE_DRAFT_MODEL_NON_QUANT=1
export LD_LIBRARY_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160
export SGLANG_DSPARK_A5_EXTRA_OPS_SO=/usr/local/Ascend/extra-ops/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 127.0.0.1 --port 6688 \
    --mem-fraction-static 0.72 \
    --max-running-requests 64 \
    --chunked-prefill-size 65536 \
    --max-prefill-tokens 65536 \
    --cuda-graph-bs-decode 1 2 4 8 16 \
    --enable-dp-lm-head \
    --enable-dp-attention --dp-size 4 \
    --reasoning-parser deepseek-v4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --speculative-dspark-block-size 5
```

#### 2. 精度测试

测试命令：

```bash
sgl-eval run gsm8k \
    --model /workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731 \
    --base-url http://127.0.0.1:6688/v1 \
    --num-examples 10 \
    --max-tokens 2048
```

测试结果：

```txt
Run directory: /root/.sgl_eval/sgl_eval_gsm8k_20261114-025657
gsm8k: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:02<00:00,  4.59it/s, acc=100.00%, truncated=0/10 (0.00%)]
== gsm8k ==
10 examples (single-shot)  |  2.2s  |  489 tok/s  |  1K tokens

* score                =  100.00%
  stop_rate            =  100.00%
  truncated_rate       =  0.00%
  error_rate           =  0.00%
  avg_input_tokens     =  86.7 tokens
  avg_output_tokens    =  106.4 tokens
  avg_thinking_tokens  =  0.0 tokens

Metrics: /root/.sgl_eval/sgl_eval_gsm8k_20261114-025657/metrics.json
Predictions: /root/.sgl_eval/sgl_eval_gsm8k_20261114-025657  (1 jsonl file(s))
```

本次评测为 GSM8K 10 条、single-shot，得分 100.00%，停止率 100.00%，截断率与请求错误率均为 0.00%。该结论限定于本次 10 条样本，不表述为全量 GSM8K 的 100% 正确率。任务书未规定必须评测全量 1319 条，本报告按本次实际样本数量呈报。

#### 3. 性能测试

测试命令：

```bash
python -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix \
    --backend sglang \
    --host 127.0.0.1 \
    --port 6688 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 64 \
    --gsp-system-prompt-len 115200 \
    --gsp-question-len 12800 \
    --gsp-output-len 1000 \
    --gsp-range-ratio 1 \
    --max-concurrency 16 \
    --num-prompts 64 \
    --warmup-requests 8 \
    --seed 42 \
    --request-rate inf \
    --cache-report \
    --output-details \
    --output-file ./prefix_on_n64_c16.jsonl
```

测试结果：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     64
Benchmark duration (s):                  99.32
Total input tokens:                      8373572
Total input text tokens:                 8373572
Total generated tokens:                  64000
Total generated tokens (retokenized):    64000
Request throughput (req/s):              0.64
Input token throughput (tok/s):          84312.88
Output token throughput (tok/s):         644.41
Peak output token throughput (tok/s):    2496.00
Peak concurrent requests:                30
Total token throughput (tok/s):          84957.29
Concurrency:                             15.67
Accept length:                           5.75
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   24324.13
Median E2E Latency (ms):                 23500.90
P90 E2E Latency (ms):                    28808.63
P95 E2E Latency (ms):                    28877.35
P99 E2E Latency (ms):                    40739.81
---------------Time to First Token----------------
Mean TTFT (ms):                          14885.94
Median TTFT (ms):                        15278.22
P90 TTFT (ms):                           18334.01
P95 TTFT (ms):                           18344.29
P99 TTFT (ms):                           18470.22
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.45
Median TPOT (ms):                        8.02
P90 TPOT (ms):                           10.94
P95 TPOT (ms):                           17.17
P99 TPOT (ms):                           26.97
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.46
Median ITL (ms):                         6.53
P90 ITL (ms):                            6.83
P95 ITL (ms):                            6.99
P99 ITL (ms):                            49.49
Max ITL (ms):                            10083.52
----------------Cache Hit Details-----------------
Total prompt tokens:                     8373572
Total cached tokens:                     7458816
  Device:                                7458816
  Host:                                  0
Cache hit rate:                          89.1%
  Device:                                100.0%
  Host:                                  0.0%
==================================================
```

判定：Mean TPOT 9.45ms 和 P99 TPOT 26.97ms 均低于 50ms。 本轮 64 条请求均成功，实际生成 64000 tokens，整体输出吞吐为 644.41 tokens/s。实际缓存命中率为 89.1%，命中内容全部位于设备内存。

### TPOT=20ms 场景

本场景采用最终配置 C1。精度引用当前配置的同一次 GSM8K 评测 A1，性能引用同一次压测 P1。50ms 与 20ms 两节复用同一配置和同一组实测数据，分别与各自阈值比较；不计作两次独立实验。

#### 1. 启动命令

完整环境变量与启动命令如下；前缀缓存保持启用，未传入 `--disable-radix-cache`。

```bash
MODEL_PATH=/workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash

export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/:$ASCEND_CUSTOM_OPP_PATH
export DEEPEP_HCCL_BUFFSIZE=1600
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=2048
export DEEPEP_NORMAL_LONG_SEQ_ROUND=16
export FORCE_DRAFT_MODEL_NON_QUANT=1
export LD_LIBRARY_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=160
export SGLANG_DSPARK_A5_EXTRA_OPS_SO=/usr/local/Ascend/extra-ops/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=torch
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
export SGLANG_OPT_FP8_WO_A_GEMM=False
export SGLANG_OPT_FUSE_WQA_WKV=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=False
export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=False
export SGLANG_OPT_USE_TILELANG_MHC_POST=False
export SGLANG_OPT_USE_TILELANG_MHC_PRE=False
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

python3 -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 127.0.0.1 --port 6688 \
    --mem-fraction-static 0.72 \
    --max-running-requests 64 \
    --chunked-prefill-size 65536 \
    --max-prefill-tokens 65536 \
    --cuda-graph-bs-decode 1 2 4 8 16 \
    --enable-dp-lm-head \
    --enable-dp-attention --dp-size 4 \
    --reasoning-parser deepseek-v4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --speculative-dspark-block-size 5
```

#### 2. 精度测试

测试命令：

```bash
sgl-eval run gsm8k \
    --model /workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731 \
    --base-url http://127.0.0.1:6688/v1 \
    --num-examples 10 \
    --max-tokens 2048
```

测试结果：

```txt
Run directory: /root/.sgl_eval/sgl_eval_gsm8k_20261114-025657
gsm8k: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [00:02<00:00,  4.59it/s, acc=100.00%, truncated=0/10 (0.00%)]
== gsm8k ==
10 examples (single-shot)  |  2.2s  |  489 tok/s  |  1K tokens

* score                =  100.00%
  stop_rate            =  100.00%
  truncated_rate       =  0.00%
  error_rate           =  0.00%
  avg_input_tokens     =  86.7 tokens
  avg_output_tokens    =  106.4 tokens
  avg_thinking_tokens  =  0.0 tokens

Metrics: /root/.sgl_eval/sgl_eval_gsm8k_20261114-025657/metrics.json
Predictions: /root/.sgl_eval/sgl_eval_gsm8k_20261114-025657  (1 jsonl file(s))
```

本次评测为 GSM8K 10 条、single-shot，得分 100.00%，停止率 100.00%，截断率与请求错误率均为 0.00%。该结论限定于本次 10 条样本，不表述为全量 GSM8K 的 100% 正确率。任务书未规定必须评测全量 1319 条，本报告按本次实际样本数量呈报。

#### 3. 性能测试

测试命令：

```bash
python -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix \
    --backend sglang \
    --host 127.0.0.1 \
    --port 6688 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 64 \
    --gsp-system-prompt-len 115200 \
    --gsp-question-len 12800 \
    --gsp-output-len 1000 \
    --gsp-range-ratio 1 \
    --max-concurrency 16 \
    --num-prompts 64 \
    --warmup-requests 8 \
    --seed 42 \
    --request-rate inf \
    --cache-report \
    --output-details \
    --output-file ./prefix_on_n64_c16.jsonl
```

测试结果：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     64
Benchmark duration (s):                  99.32
Total input tokens:                      8373572
Total input text tokens:                 8373572
Total generated tokens:                  64000
Total generated tokens (retokenized):    64000
Request throughput (req/s):              0.64
Input token throughput (tok/s):          84312.88
Output token throughput (tok/s):         644.41
Peak output token throughput (tok/s):    2496.00
Peak concurrent requests:                30
Total token throughput (tok/s):          84957.29
Concurrency:                             15.67
Accept length:                           5.75
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   24324.13
Median E2E Latency (ms):                 23500.90
P90 E2E Latency (ms):                    28808.63
P95 E2E Latency (ms):                    28877.35
P99 E2E Latency (ms):                    40739.81
---------------Time to First Token----------------
Mean TTFT (ms):                          14885.94
Median TTFT (ms):                        15278.22
P90 TTFT (ms):                           18334.01
P95 TTFT (ms):                           18344.29
P99 TTFT (ms):                           18470.22
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          9.45
Median TPOT (ms):                        8.02
P90 TPOT (ms):                           10.94
P95 TPOT (ms):                           17.17
P99 TPOT (ms):                           26.97
---------------Inter-Token Latency----------------
Mean ITL (ms):                           9.46
Median ITL (ms):                         6.53
P90 ITL (ms):                            6.83
P95 ITL (ms):                            6.99
P99 ITL (ms):                            49.49
Max ITL (ms):                            10083.52
----------------Cache Hit Details-----------------
Total prompt tokens:                     8373572
Total cached tokens:                     7458816
  Device:                                7458816
  Host:                                  0
Cache hit rate:                          89.1%
  Device:                                100.0%
  Host:                                  0.0%
==================================================
```

判定：Mean TPOT 9.45ms 低于 20ms；P99 TPOT 26.97ms 高于 20ms。按平均值口径满足目标，按 P99 口径尚未满足。 本轮 64 条请求均成功，实际生成 64000 tokens，整体输出吞吐为 644.41 tokens/s。实际缓存命中率为 89.1%，命中内容全部位于设备内存。

## 三、参考文档评价

以下为根据本次部署、评测与调优过程形成的建议评分，提交前由体验者确认；未实际执行的 profiling 不给出体验分数。

| 评价维度 | 建议评分（1~5） | 说明 |
|---|---:|---|
| 安装部署文档 | 4 | 示例覆盖 NPU 环境初始化、模型路径与完整启动参数，本次已成功启动服务。建议补充镜像与驱动、CANN、源码 commit 的统一归档方式。 |
| 最佳实践文档 | 3 | 配置能够形成可用基线，但包含禁用前缀缓存的示例；面向 90% 共享前缀众测时，需要明确说明如何切换缓存配置及验证命中。 |
| 精度测试文档 | 4 | sgl-eval 命令可运行，能够生成得分、截断率、错误率、metrics 与预测记录。建议区分 10 条快速检查与更大样本评测。 |
| 性能测试文档 | 3 | 可输出 TPOT、TTFT、吞吐、ITL 和缓存详情。建议统一 K 的含义、目标长度与实际 token 计数，以及 TPOT 平均值或分位数的验收口径。 |
| 参数调优文档 | 4 | 参数维度覆盖缓存、prefill 分块、并发、图执行与投机解码。建议增加 4 卡 950 的 128K 共享前缀用例及逐项调整记录。 |
| Profiling 文档 | 未评分 | 本次未采集算子级 trace，不对未实际执行的流程给出体验评分。 |

## 四、问题与改进建议

### 发现的问题

1. **部署示例与缓存场景需要区分。** 初始启动配置包含 `--disable-radix-cache`。在共享前缀场景中，生成相同前缀并不自动保证缓存复用。删除该开关并通过缓存报告验证后，本次得到 89.1% 的实际命中率。
2. **TPOT 验收统计量未明确。** 最终 Mean TPOT 为 9.45ms，P99 为 26.97ms。对 20ms 阈值，采用两种口径会得出不同结论，应在任务书和报告模板中明确。
3. **目标长度和实际 token 计数有差异。** 名义输入为 128000 tokens，服务端平均实测为 130837.06 tokens。需要说明生成器、tokenizer 与特殊 token 的影响，并给出精确长度测试路径。
4. **缓存详情中百分比的分母容易混淆。** 本次全体输入的命中率为 89.1%；Device 100.0% 是命中内容的存储层级占比。建议在字段旁说明分母。
5. **峰值并发字段需要统计说明。** 客户端上限为 16，报告中的 Peak concurrent requests 为 30。当前 main 的统计实现按秒分桶，可能将同一秒结束与开始的请求同时计入；该字段不能直接证明客户端限流失效，本机具体实现应随 commit 核验。
6. **多轮复现实验需要固定缓存与预热流程。** 相同 seed 和相同请求反复运行时，历史完整请求缓存可能影响下一轮。本轮记录了预热 8 条；正式多轮复现还应固定清理缓存与预热的顺序并归档日志。本报告不追溯宣称本轮执行过未记录的额外清理步骤。

### 改进建议

1. 为 DeepSeek-V4-Flash 的 4 卡 950 部署增加独立的 128K-1K-90% 共享前缀示例，明确前缀缓存、DSPARK 和各 DP rank 的兼容性及预热方式。
2. 将 `--cache-report` 与 `--output-details` 纳入性能采集示例，明确保存原始 JSONL、服务日志及环境信息。
3. 在验收模板中同时记录 Mean / P95 / P99 TPOT，明确采用哪一项判定，并说明是否要求每请求达标率。
4. 统一 128K 和 1K 的十进制或二进制定义，要求报告同时给出目标 token 数和服务端实际 token 数。
5. 针对持续补入请求场景增加 prefill 分块与尾部停顿的对照案例。若活动以 P99≤20ms 验收，可进一步对比 chunk=16384 或 8192；这些参数尚未在本报告中实测，不能计为已取得的优化结果。

## 五、调优过程与效果分析

### 已完成的调整

初始配置使用 TP=4、DP Attention=4、DSPARK、chunked-prefill-size=65536，并显式关闭前缀缓存。最终配置保留上述并行、图执行、内存预算与投机解码参数，删除 `--disable-radix-cache`，启用前缀复用。最终压测增加了缓存详情和逐请求结果输出，固定预热次数与 seed。

| 阶段 | 名义输入 / 输出 | 请求数 / 并发 | 前缀缓存 | Mean TPOT（ms） | P99 TPOT（ms） | 输出吞吐（tokens/s） |
|---|---|---|---|---:|---:|---:|
| 初期探索 | 64000 / 1000 | 20 / 20 | 关闭 | 34.36 | 47.75 | 171.73 |
| 128K 基线 B1 | 128000 / 1000 | 64 / 16 | 关闭 | 79.51 | 171.34 | 112.13 |
| 启用缓存的小批验证 | 128000 / 1000 | 16 / 16 | 启用 | 9.13 | 16.25 | 298.27 |
| 最终记录 P1 | 128000 / 1000 | 64 / 16 | 启用 | 9.45 | 26.97 | 644.41 |

不同请求数或输入长度的探索记录只用于说明过程，不直接作为严格的单变量对比。以下比较采用同为名义 128K、64 条请求、并发 16 的 B1 与 P1。

| 指标 | 缓存关闭 B1 | 缓存启用 P1 | 根据已记录数值计算的变化 |
|---|---:|---:|---|
| 测试时长 | 570.77s | 99.32s | 缩短约 82.60% |
| Mean TTFT | 60240.70ms | 14885.94ms | 降低约 75.29% |
| Mean TPOT | 79.51ms | 9.45ms | 降低约 88.11% |
| P99 TPOT | 171.34ms | 26.97ms | 降低约 84.26% |
| 输出吞吐 | 112.13 tokens/s | 644.41 tokens/s | 提高至约 5.75 倍 |
| Max ITL | 81092.76ms | 10083.52ms | 长停顿减轻，尚未完全消除 |

上述结果支持将前缀缓存作为本次优化的关键改动。由于尚无多次重复试验的方差、算子 trace 或逐轮完全相同缓存初态的归档，不把全部差异归结为已隔离验证的算子级加速。

最终平均 E2E 为 24324.13ms，平均 TTFT 为 14885.94ms，首 token 之后的平均耗时约为 9438.19ms，与每条 1000 输出 token 对应的 Mean TPOT 9.45ms 一致。P99 ITL 为 49.49ms，Max ITL 仍达约 10.08 秒，说明平均 TPOT 达标并不等于流式输出完全无停顿。

## 六、交付件与提交状态

| 交付项 | 当前状态 | 说明 |
|---|---|---|
| 50ms 场景完整启动命令 | 已整理 | 含全部环境变量，见第二节及 scripts/launch_tpot50ms.sh |
| 20ms 场景完整启动命令 | 已整理 | 与 50ms 复用同一最终配置，见第二节及 scripts/launch_tpot20ms.sh |
| 精度数据 | 已采集 | 最终配置 GSM8K 10/10，100.00%，无截断、无错误；两场景复用 A1 |
| 性能数据 | 已采集 | P1 共 64 条；两场景复用一次实测并分别判断阈值 |
| 平均 TPOT 门槛 | 两档均满足 | Mean TPOT=9.45ms |
| P99 TPOT 门槛 | 50ms 满足，20ms 未满足 | P99 TPOT=26.97ms |
| 终端输出附件 | 已整理 | 精度及性能终端转录，另含 128K 缓存关闭基线 |
| 原机 JSONL / metrics / predictions | 服务器路径已记录 | 原始文件应从服务器导出后随最终验收归档；终端转录不替代这些原始文件 |
| Issue | 英文草稿已整理，提交链接待补 | issues/ 下含两份文档反馈草稿；尚未提交到 GitHub |
| PR | 暂无代码改动 PR | 若维护者要求以文档 PR 落地，再补关联任务编号与链接 |
| 环境归档 | 待补充设备及版本输出 | npu-smi、SGLang 版本、commit SHA、镜像 ID / digest |

本报告已形成按 Mean TPOT 口径的两档性能与本次 10 条精度交付材料。活动规定的完整任务闭环还包括问题反馈提交及原始附件归档，不将未提交的 Issue/PR 或未归档的文件记为已完成。

### 服务器上的原始文件位置

- 精度 metrics：`/root/.sgl_eval/sgl_eval_gsm8k_20261114-025657/metrics.json`。
- 精度预测目录：`/root/.sgl_eval/sgl_eval_gsm8k_20261114-025657/`，控制台记录有 1 个 JSONL 文件。
- 性能逐请求文件：`/workspace/prefix_on_n64_c16.jsonl`（按运行时 `/workspace` 工作目录解析）。若文件包含多行历史运行记录，应按对应配置和汇总指标识别 P1，不将多次结果误合并。

### Issue 与 PR 规范

Issue / PR 使用英文。建议 Issue 标签为 MeetUp、NPU、DeepSeek-V4-Flash，由仓库实际可用标签及维护者权限决定。任务正式编号尚未归档，本报告自编号不作为官方任务编号使用。PR 如后续创建，应附正式任务编号与优化场景。

Issue / PR 草稿只讨论公开文档中的配置与计量口径，不附本次保密性能数值、服务器信息或原始日志；众测材料仅在活动授权渠道归档。

## 七、参考资料

1. SGLang 仓库：https://github.com/sgl-project/sglang
2. NPU DeepSeek-V4-Flash 部署：https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/deepseek_v4_flash
3. NPU DeepSeek-V4-Flash 最佳实践：https://docs.sglang.io/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/deepseek_v4_flash
4. NPU 精度测试：https://docs.sglang.io/docs/hardware-platforms/ascend-npus/evaluation/accuracy_evaluation
5. NPU 性能测试：https://docs.sglang.io/docs/hardware-platforms/ascend-npus/evaluation/performance_testing
6. NPU 参数调优：https://docs.sglang.io/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning
7. NPU Profiling：https://docs.sglang.io/docs/hardware-platforms/ascend-npus/optimization/profiling

本报告中的性能和精度数值为本次终端实测记录；参考文档的示例性能不作为本次测试结果。
