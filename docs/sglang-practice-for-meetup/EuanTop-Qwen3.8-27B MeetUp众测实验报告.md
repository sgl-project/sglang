# Qwen3.8-27B MeetUp 众测实验报告

## 一、报告基本信息

| 项目 | 内容 |
|---|---|
| 任务名 | Qwen3.8-27B MeetUp 众测 |
| 开发者 ID | EuanTop |
| 体验日期 | 2026-11-14（北京时间） |
| 开发环境 | HiDevLab 容器，Ubuntu 22.04.5 LTS / aarch64 |
| 硬件 | Atlas 800T A2，4 张昇腾 910B3，每张 64 GiB HBM |
| 软件 | CANN 9.1.0，HDK 26.1.1，Python 3.12.13，PyTorch 2.10.0+cpu，torch-npu 2.10.0.post6，EvalScope 1.14.1 |
| SGLang | 0.0.0.dev22121+g0fa9ff55f.d20261112，提交 `0fa9ff55ffbfd80377ad0e0a1de123544c69a1f9` |
| 任务书关联镜像 | `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-910b-20261112`（平台预置容器的镜像摘要在容器内未核实） |

[众测任务书](https://docs.qq.com/doc/DVXpjTmxoWXB5bHdN?nlc=1)

## 二、操作记录

两套模型分别在四卡上运行，TP=4。每条压测请求输入 128,000 tokens（共享前缀 115,200，独立后缀 12,800），输出 1,000 tokens。每组清缓存后预热 1 条请求，随机种子为 1。

首轮每组 16 个请求。调优与独立复测均采用客户端并发 16、每组 32 个请求。采用 SGLang 昇腾性能测试的平均 TPOT 口径，在 ≤ 50 ms 的配置中按输出吞吐选型。GSM8K 采用 200 题、4-shot、种子 42、温度 0。 [SGLang 昇腾性能测试判定](https://github.com/sgl-project/sglang/blob/0fa9ff55ffbfd80377ad0e0a1de123544c69a1f9/python/sglang/test/ascend/e2e/test_npu_performance_utils.py#L1038-L1047)

实跑并发取服务端解码日志中的 `#running-req` 峰值。性能输出里的 `Peak concurrent requests` 按秒统计与该秒有重叠的请求数，同一秒内先后处理的请求也会计入。

### 调优改动

| 调整项 | 首轮 | 调优及复测 | 目的 |
|---|---|---|---|
| `--max-mamba-cache-size` | 64 | 128 | 给 16 个请求及前缀缓存保留状态槽 |
| `--max-running-requests` | 配置 16，实际 12 | 配置 16，实跑 16 | 通过状态槽容量提高实际并发 |
| `--chunked-prefill-size` | 16,384 | 16,384 与 32,768 对照 | 比较分块大小对吞吐和延迟的影响 |
| `--max-prefill-tokens` | 16,384 | 与分块大小配套设为 16,384 或 32,768 | 允许预填充批次使用完整分块配额 |
| 服务切换 | 向进程组发出退出信号，等待约 60 秒 | 向启动的服务主进程发出信号，检查其进程组是否退出 | 让服务按自身流程结束并等待四卡可用 |

### 启动环境变量

两套权重共用以下环境变量。

~~~bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
if [[ -r /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
unset ASCEND_LAUNCH_BLOCKING
export NO_PROXY=127.0.0.1,localhost
export no_proxy=$NO_PROXY
export ASCEND_USE_FIA=1
export GDN_ATTN_BACKEND_TRITON=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export STREAMS_PER_DEVICE=32
~~~

### 使用 Qwen3.8-27B 权重

#### 1. 启动命令

~~~bash
/usr/local/python3.12.13/bin/python -m sglang.launch_server \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 6688 \
  --tp-size 4 \
  --nnodes 1 \
  --attention-backend ascend \
  --device npu \
  --context-length 135168 \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --mamba-radix-cache-strategy extra_buffer \
  --trust-remote-code \
  --max-running-requests 16 \
  --max-mamba-cache-size 128 \
  --mem-fraction-static 0.80 \
  --cuda-graph-max-bs-decode 16 \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4
~~~

#### 2. GSM8K 精度测试

首轮 **98.5%（197/200）**，调优配置复测 **99.0%（198/200）**。

~~~bash
evalscope eval \
  --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
  --api-url http://127.0.0.1:6688/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --limit 200 \
  --eval-batch-size 16 \
  --work-dir /workspace/meetup-qwen38-rOIauf/tuning/bf16-p16-chunk32768-confirm-gsm8k
~~~

EvalScope 复测结果

~~~text
┌─────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model       │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├─────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 99%     │
└─────────────┴───────────┴────────────┴──────────┴───────┴─────────┘

Model        Dataset      Num  Avg Lat    Avg TTFT    Avg TPOT    Avg Thpt      Avg In    Avg Out
-----------  ---------  -----  ---------  ----------  ----------  ----------  --------  ---------
Qwen3.8-27B  GSM8K        200  14.454 s   —           —           46.1 tok/s       729        666
~~~

#### 3. 性能测试

先清理空闲服务的缓存，再运行复测命令。

~~~bash
mkdir -p tuning
curl -fsS --noproxy '*' -X POST http://127.0.0.1:6688/flush_cache
~~~

~~~bash
/usr/local/python3.12.13/bin/python -m sglang.benchmark.serving \
  --dataset-name generated-shared-prefix \
  --backend sglang \
  --host 127.0.0.1 \
  --port 6688 \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 32 \
  --gsp-system-prompt-len 115200 \
  --gsp-question-len 12800 \
  --gsp-output-len 1000 \
  --max-concurrency 16 \
  --num-prompts 32 \
  --request-rate inf \
  --seed 1 \
  --warmup-requests 1 \
  --tokenize-prompt \
  --output-file tuning/bf16-p16-chunk32768-confirm.jsonl \
  --output-details \
  --cache-report
~~~

SGLang 复测结果

~~~text
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     32
Benchmark duration (s):                  132.21
Total input tokens:                      4096000
Total input text tokens:                 4096000
Total generated tokens:                  32000
Total generated tokens (retokenized):    28722
Request throughput (req/s):              0.24
Input token throughput (tok/s):          30981.64
Output token throughput (tok/s):         242.04
Peak output token throughput (tok/s):    957.00
Peak concurrent requests:                21
Total token throughput (tok/s):          31223.68
Concurrency:                             15.43
Accept length:                           3.49
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   63765.83
Median E2E Latency (ms):                 63889.31
P90 E2E Latency (ms):                    98110.88
P95 E2E Latency (ms):                    104871.49
P99 E2E Latency (ms):                    110563.79
---------------Time to First Token----------------
Mean TTFT (ms):                          24915.11
Median TTFT (ms):                        15442.34
P90 TTFT (ms):                           49005.86
P95 TTFT (ms):                           49050.37
P99 TTFT (ms):                           49096.85
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          38.89
Median TPOT (ms):                        35.02
P90 TPOT (ms):                           62.47
P95 TPOT (ms):                           69.79
P99 TPOT (ms):                           72.10
---------------Inter-Token Latency----------------
Mean ITL (ms):                           38.91
Median ITL (ms):                         16.25
P90 ITL (ms):                            23.55
P95 ITL (ms):                            32.65
P99 ITL (ms):                            591.64
Max ITL (ms):                            37280.52
----------------Cache Hit Details-----------------
Total prompt tokens:                     4096000
Total cached tokens:                     3635712
  Device:                                3635712
  Host:                                  0
Cache hit rate:                          88.8%
  Device:                                100.0%
  Host:                                  0.0%
==================================================
~~~

首轮、筛选与复测对照

| 配置 | 成功请求 | 平均 TTFT（s） | 平均 TPOT（ms） | P99 TPOT（ms） | 输出吞吐（token/s） | 实际缓存命中 |
|---|---:|---:|---:|---:|---:|---:|
| 首轮客户端并发 4 | 16 | 6.05 | 18.54 | 29.34 | 158.87 | 89.92% |
| 首轮客户端并发 8 | 16 | 11.75 | 33.64 | 57.95 | 171.97 | 89.92% |
| 首轮客户端并发 16 | 16 | 30.75 | 30.02 | 52.16 | 225.05 | 89.92% |
| 16K 分块筛选，实跑并发 16 | 32 | 18.84 | 46.14 | 76.97 | 239.35 | 89.96% |
| 32K 分块筛选，实跑并发 16 | 32 | 24.09 | 39.18 | 66.57 | 247.30 | 88.42% |
| 32K 分块独立复测，实跑并发 16 | 32 | 24.92 | 38.89 | 72.10 | 242.04 | 88.76% |

### 使用 Qwen3.8-27B-w8a8 权重

#### 1. 启动命令

~~~bash
/usr/local/python3.12.13/bin/python -m sglang.launch_server \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
  --host 127.0.0.1 \
  --port 6688 \
  --tp-size 4 \
  --nnodes 1 \
  --attention-backend ascend \
  --device npu \
  --context-length 135168 \
  --chunked-prefill-size 16384 \
  --max-prefill-tokens 16384 \
  --mamba-radix-cache-strategy extra_buffer \
  --trust-remote-code \
  --max-running-requests 16 \
  --max-mamba-cache-size 128 \
  --mem-fraction-static 0.80 \
  --cuda-graph-max-bs-decode 16 \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --quantization modelslim
~~~

#### 2. GSM8K 精度测试

首轮 **99.0%（198/200）**，调优配置复测 **98.0%（196/200）**。

~~~bash
evalscope eval \
  --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
  --api-url http://127.0.0.1:6688/v1 \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets gsm8k \
  --limit 200 \
  --eval-batch-size 16 \
  --work-dir /workspace/meetup-qwen38-rOIauf/tuning/w8a8-p16-chunk16384-confirm-gsm8k
~~~

EvalScope 复测结果

~~~text
┌──────────────────┬───────────┬────────────┬──────────┬───────┬─────────┐
│ Model            │ Dataset   │ Metric     │ Subset   │   Num │ Score   │
├──────────────────┼───────────┼────────────┼──────────┼───────┼─────────┤
│ Qwen3.8-27B-w8a8 │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 98%     │
└──────────────────┴───────────┴────────────┴──────────┴───────┴─────────┘

Model             Dataset      Num  Avg Lat    Avg TTFT    Avg TPOT    Avg Thpt       Avg In    Avg Out
----------------  ---------  -----  ---------  ----------  ----------  -----------  --------  ---------
Qwen3.8-27B-w8a8  GSM8K        200  13.93 s    —           —           45.02 tok/s       729        627
~~~

#### 3. 性能测试

先清理空闲服务的缓存，再运行复测命令。

~~~bash
mkdir -p tuning
curl -fsS --noproxy '*' -X POST http://127.0.0.1:6688/flush_cache
~~~

~~~bash
/usr/local/python3.12.13/bin/python -m sglang.benchmark.serving \
  --dataset-name generated-shared-prefix \
  --backend sglang \
  --host 127.0.0.1 \
  --port 6688 \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 32 \
  --gsp-system-prompt-len 115200 \
  --gsp-question-len 12800 \
  --gsp-output-len 1000 \
  --max-concurrency 16 \
  --num-prompts 32 \
  --request-rate inf \
  --seed 1 \
  --warmup-requests 1 \
  --tokenize-prompt \
  --output-file tuning/w8a8-p16-chunk16384-confirm.jsonl \
  --output-details \
  --cache-report
~~~

SGLang 复测结果

~~~text
============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 16
Successful requests:                     32
Benchmark duration (s):                  114.31
Total input tokens:                      4096000
Total input text tokens:                 4096000
Total generated tokens:                  32000
Total generated tokens (retokenized):    29751
Request throughput (req/s):              0.28
Input token throughput (tok/s):          35831.58
Output token throughput (tok/s):         279.93
Peak output token throughput (tok/s):    881.00
Peak concurrent requests:                21
Total token throughput (tok/s):          36111.52
Concurrency:                             15.70
Accept length:                           3.52
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   56068.51
Median E2E Latency (ms):                 55216.44
P90 E2E Latency (ms):                    85407.83
P95 E2E Latency (ms):                    89565.19
P99 E2E Latency (ms):                    92712.55
---------------Time to First Token----------------
Mean TTFT (ms):                          16617.82
Median TTFT (ms):                        12019.10
P90 TTFT (ms):                           35220.60
P95 TTFT (ms):                           35261.06
P99 TTFT (ms):                           35303.40
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          39.49
Median TPOT (ms):                        38.22
P90 TPOT (ms):                           53.74
P95 TPOT (ms):                           64.25
P99 TPOT (ms):                           73.72
---------------Inter-Token Latency----------------
Mean ITL (ms):                           39.51
Median ITL (ms):                         17.56
P90 ITL (ms):                            24.64
P95 ITL (ms):                            35.67
P99 ITL (ms):                            530.61
Max ITL (ms):                            29991.45
----------------Cache Hit Details-----------------
Total prompt tokens:                     4096000
Total cached tokens:                     3684864
  Device:                                3684864
  Host:                                  0
Cache hit rate:                          90.0%
  Device:                                100.0%
  Host:                                  0.0%
==================================================
~~~

首轮、筛选与复测对照

| 配置 | 成功请求 | 平均 TTFT（s） | 平均 TPOT（ms） | P99 TPOT（ms） | 输出吞吐（token/s） | 实际缓存命中 |
|---|---:|---:|---:|---:|---:|---:|
| 首轮客户端并发 4 | 16 | 3.92 | 17.12 | 21.43 | 186.90 | 89.92% |
| 首轮客户端并发 8 | 16 | 8.49 | 23.70 | 34.93 | 238.88 | 89.92% |
| 首轮客户端并发 16 | 16 | 26.71 | 26.91 | 47.01 | 247.29 | 89.92% |
| 16K 分块筛选，实跑并发 16 | 32 | 15.43 | 38.80 | 75.32 | 288.81 | 89.96% |
| 32K 分块筛选，实跑并发 16 | 32 | 21.48 | 38.43 | 77.18 | 259.72 | 88.42% |
| 16K 分块独立复测，实跑并发 16 | 32 | 16.62 | 39.49 | 73.72 | 279.93 | 89.96% |

### 对比总结

| 权重 | 最佳分块 | 实跑并发 | GSM8K（200 题） | 复测输出吞吐（token/s） | 平均 TPOT（ms） | P99 TPOT（ms） | 实际缓存命中 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 32K | 16 | 99.0% | 242.04 | 38.89 | 72.10 | 88.76% |
| W8A8 | 16K | 16 | 98.0% | 279.93 | 39.49 | 73.72 | 89.96% |

W8A8 的复测输出吞吐比 BF16 高 15.7%，GSM8K 少答对 2 题。

筛选中 BF16 从 16K 调到 32K 后，输出吞吐增加 3.3%，代价是首 token 延迟增加。W8A8 的 16K 配置同时取得更高吞吐、更低首 token 延迟和更高实测缓存命中率。

## 三、参考文档评价

| 文档 | 评分（1–5） | 说明 |
|---|---:|---|
| [安装部署](https://github.com/sgl-project/sglang/blob/0fa9ff55ffbfd80377ad0e0a1de123544c69a1f9/docs/docs/hardware-platforms/ascend-npus/model-deployment/tutorials/qwen3_6_27b.mdx) | 4 | 服务和接口验证可复用。建议补充预置 HiDevLab 容器中 `python` 与系统 `python3` 的路径区别。 |
| [最佳实践](https://github.com/sgl-project/sglang/blob/0fa9ff55ffbfd80377ad0e0a1de123544c69a1f9/docs/docs/hardware-platforms/ascend-npus/model-deployment/best-practices/qwen3_6_27b.mdx) | 4 | 给出了 BF16 的共享前缀和 W8A8 的 128K 参考配置。建议补充 A2 四卡上的 128K 共享前缀示例。 |
| [精度测试](https://github.com/sgl-project/sglang/blob/0fa9ff55ffbfd80377ad0e0a1de123544c69a1f9/docs/docs/hardware-platforms/ascend-npus/evaluation/accuracy_evaluation.mdx) | 5 | EvalScope 的 OpenAI 兼容接口示例可直接用于两套权重。 |
| [性能测试](https://github.com/sgl-project/sglang/blob/0fa9ff55ffbfd80377ad0e0a1de123544c69a1f9/docs/docs/hardware-platforms/ascend-npus/evaluation/performance_testing.mdx) | 4 | 指标定义可用于验收。建议增加共享前缀、测得缓存命中率与实际并发的核对步骤。 |
| [参数调优](https://github.com/sgl-project/sglang/blob/0fa9ff55ffbfd80377ad0e0a1de123544c69a1f9/docs/docs/hardware-platforms/ascend-npus/optimization/parameter_tuning.mdx) | 4 | 分块预填充和缓存调优入口清晰。建议说明 Mamba 状态槽如何限制有效并发。 |

## 四、问题与改进建议

1. **服务实际并发受 Mamba 状态槽限制。** 在 4 张昇腾 910B3 上，BF16 和 W8A8 均配置 `--max-running-requests 16`，预填充均为 16K。状态槽为 64 时，启动日志显示每个请求需要 5 个槽，有效并发为 12，压测最高也为 12。仅将启动参数 `--max-mamba-cache-size` 从 64 调到 128 后，两套权重的启动上限和压测峰值均达到 16。建议在 Ascend 调优指南中说明如何用启动日志的 `max_running_requests` 和压测日志的 `#running-req` 核验并发。[SGLang Issue #38846](https://github.com/sgl-project/sglang/issues/38846)
2. **预置容器的系统 Python 与 SGLang Python 是两个环境。** 系统 `/usr/bin/python3` 缺少 pip 模块。部署脚本使用容器预装的 Python 3.12，建议前置检查随安装命令明确解释器路径。
3. **向整个服务进程组发送 SIGTERM 时，首轮权重切换超时。** 服务诊断等待约 60 秒，原切换脚本同时超时。后续从服务主进程发起退出并等待关联进程结束。
4. **实测缓存命中率低于输入共享比例。** 32K 分块首测的实际命中率为 88.42%，输入共享比例为 90%。筛选脚本已改为记录实际值，避免提前丢弃有效结果。
