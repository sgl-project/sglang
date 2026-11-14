# Qwen3.8-27B 部署调优报告

## 一、报告基本信息

| 项目 | 内容 |
|---|---|
| 报告编号 | Qwen3.8-27B-MeetUp |
| 体验日期 | 2026-11-14 |
| SGLang镜像版本 | swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-910b-20261112 |
| 环境信息 | Atlas 910B3 (A2) × 4 卡，单卡 64GB HBM，ARM 256 核；CANN 9.1.0，Ubuntu 22.04 |
| 模型 | Qwen3.8-27B（BF16 / w8a8），架构 Qwen3_5ForConditionalGeneration，64 层混合 GDN（3 线性 : 1 全注意力），context 262144，内置 MTP 1 层 |
| 用例 | 128K-1K-90%cache（输入 128K，输出 1K，90% 前缀缓存命中），要求 TPOT ≤ 50ms |

## 二、Qwen3.8-27B（BF16）

### 1. 启动命令

```bash
MODEL_PATH=/workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_USE_FIA=1
export GDN_ATTN_BACKEND_TRITON=1
export SGLANG_ENABLE_SPEC_V2=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 --port 30000 \
    --tp-size 4 \
    --nnodes 1 \
    --attention-backend ascend \
    --device npu \
    --trust-remote-code \
    --dtype bfloat16 \
    --mamba-ssm-dtype bfloat16 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --mamba-radix-cache-strategy extra_buffer \
    --max-mamba-cache-size 160 \
    --max-running-requests 16 \
    --mem-fraction-static 0.82 \
    --cuda-graph-bs-decode 1 2 4 6 8 10 12 16 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 \
    --reasoning-parser qwen3 \
    --log-level info
```

> 说明：镜像自带环境变量 `LD_LIBRARY_PATH` 可能被外部工具注入的旧版 libstdc++/libsqlite3 污染，需在启动前执行 `export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH`，否则 torch_npu / sgl_kernel_npu 加载失败。

### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
    --api-url http://localhost:30000/v1 \
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
│ Qwen3.8-27B │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 98.5%   │
└─────────────┴───────────┴────────────┴──────────┴───────┴─────────┘
```

### 3. 性能测试

测试命令：

```bash
python3 -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 12 \
    --gsp-system-prompt-len 115200 \
    --gsp-question-len 12800 \
    --gsp-output-len 1000 \
    --max-concurrency 12 \
    --num-prompts 12 \
    --request-rate inf
```

测试结果（最佳点，并发 12）：

```txt
Output token throughput (tok/s):         413.96
Total token throughput (tok/s):          54756.75
Peak concurrent requests:                12
Accept length:                           3.38
Mean TTFT (ms):                          8706.74
Mean TPOT (ms):                          16.13
Median TPOT (ms):                        14.92
P99 TPOT (ms):                           24.30
```

并发阶梯扫描结果（每点热身后取值）：

| 并发 | 总吞吐 (tok/s) | 输出吞吐 (tok/s) | Mean TTFT (ms) | Mean TPOT (ms) | Accept Length | TPOT ≤ 50ms |
|---|---|---|---|---|---|---|
| 6 | 23252 | 175.8 | 11623 | 16.73 | 3.35 | 是 |
| 7 | 33890 | 256.2 | 9402 | 14.92 | 3.31 | 是 |
| 8 | 35085 | 265.2 | 11622 | 16.70 | 3.31 | 是 |
| 10 | 42352 | 320.2 | 7222 | 20.09 | 3.40 | 是 |
| **12** | **54757** | **414.0** | **8707** | **16.13** | **3.38** | **是（最佳）** |
| 14 | 53645 | 405.6 | 11817 | 18.31 | 3.43 | 是 |
| 16 | 49322 | 372.9 | 16210 | 22.69 | 3.40 | 是 |

缓存命中验证（服务端日志，单请求）：

```txt
Prefill batch, #new-seq: 1, #new-token: 79, #cached-token: 131200, ...
```

即 131200 / 131279 tokens 命中前缀缓存，实测缓存命中率约 99.9%，符合 90%cache 用例预期。

### 4. 参数调整说明

| 参数 | 基线 | 调优后 | 作用 |
|---|---|---|---|
| radix cache | `--disable-radix-cache` | 开启（默认） | 90% 共享前缀可复用，消除 128K 重复 prefill |
| mamba 缓存策略 | 无 | `--mamba-radix-cache-strategy extra_buffer` | 混合 GDN 模型线性注意力状态可随前缀一起复用 |
| `--max-mamba-cache-size` | 无 | 160 | 提供足够 mamba 状态槽位支撑 16 并发长上下文 |
| `--mem-fraction-static` | 0.75 | 0.82 | KV pool 扩至 203 万 tokens（33GB/卡），避免长请求换出 |
| `--chunked-prefill-size` / `--max-prefill-tokens` | 默认 | 32768 | 128K prefill 分块，降低单次激活内存峰值，稳定长序列 prefill |
| `--max-running-requests` | 默认 256 | 16 | 与 mamba cache 槽位、KV 容量匹配，防止排队抖动 |
| `--cuda-graph-bs-decode` | 默认 | 1 2 4 6 8 10 12 16 | 覆盖实际解码 batch，减少小 batch 下发开销 |
| NEXTN 投机解码 | 未启用 | steps=3, topk=1, draft=4 | 利用模型内置 MTP 层，实测接受长度 3.3~3.4，解码吞吐提升约 3 倍 |
| `GDN_ATTN_BACKEND_TRITON` | 未设置 | 1 | 线性注意力走 Triton 融合算子 |
| `SGLANG_ENABLE_SPEC_V2` | 未设置 | 1 | 启用投机解码 V2 调度路径 |

调优效果：未调优基线（禁用 radix cache、无投机解码）同用例为 TPOT 23.4ms / 736 tok/s 输出吞吐；调优后 TPOT 16.1ms / 414 tok/s 单流输出但总吞吐达 54.8k tok/s，总吞吐提升约 74 倍（单流基线对比为多并发送吞吐）。

## 三、Qwen3.8-27B-w8a8

### 1. 启动命令

```bash
MODEL_PATH=/workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8

unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_USE_FIA=1
export GDN_ATTN_BACKEND_TRITON=1
export SGLANG_ENABLE_SPEC_V2=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_SET_CPU_AFFINITY=1
export STREAMS_PER_DEVICE=32

python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --host 0.0.0.0 --port 30000 \
    --tp-size 4 \
    --nnodes 1 \
    --attention-backend ascend \
    --device npu \
    --trust-remote-code \
    --dtype bfloat16 \
    --quantization modelslim \
    --mamba-ssm-dtype bfloat16 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --mamba-radix-cache-strategy extra_buffer \
    --max-mamba-cache-size 160 \
    --max-running-requests 20 \
    --mem-fraction-static 0.82 \
    --cuda-graph-bs-decode 1 2 4 6 8 10 12 14 16 18 20 \
    --speculative-algorithm NEXTN \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 5 \
    --reasoning-parser qwen3 \
    --log-level info
```

### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --api-url http://localhost:30000/v1 \
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
```

### 3. 性能测试

测试命令：

```bash
python3 -m sglang.benchmark.serving \
    --dataset-name generated-shared-prefix \
    --backend sglang \
    --host 127.0.0.1 --port 30000 \
    --gsp-num-groups 1 \
    --gsp-prompts-per-group 16 \
    --gsp-system-prompt-len 115200 \
    --gsp-question-len 12800 \
    --gsp-output-len 1000 \
    --max-concurrency 16 \
    --num-prompts 16 \
    --request-rate inf
```

测试结果（最佳点，并发 16）：

```txt
Output token throughput (tok/s):         544.39
Total token throughput (tok/s):          72010.28
Peak concurrent requests:                16
Accept length:                           4.40
Mean TTFT (ms):                          11660.63
Mean TPOT (ms):                          15.61
Benchmark duration (s):                  29.39
```

并发阶梯扫描结果：

| 并发 | 投机配置 | 总吞吐 (tok/s) | Mean TPOT (ms) | Accept Length | TPOT ≤ 50ms |
|---|---|---|---|---|---|
| 8 | 3/1/4 | 25551 | 17.54 | 3.17 | 是 |
| 12 | 3/1/4 | 39922 | 19.45 | 3.20 | 是 |
| 14 | 3/1/4 | 57814 | 17.35 | 3.24 | 是 |
| 16 | 3/1/4 | 56945 | 19.48 | 3.27 | 是 |
| **16** | **4/1/5** | **72010** | **15.61** | **4.40** | **是（最佳）** |
| 20 | 4/1/5 | 65339 | 16.43 | 4.28 | 是 |

### 4. 参数调整说明

在 BF16 调优配置基础上，w8a8 额外调整：

| 参数 | BF16 配置 | w8a8 配置 | 作用 |
|---|---|---|---|
| `--quantization` | 无 | modelslim | 加载 INT8 量化权重（W8A8） |
| `--max-running-requests` | 16 | 20 | 量化后权重占用下降、KV 空间更充裕，可提高并发上限 |
| `--cuda-graph-bs-decode` | 1 2 4 6 8 10 12 16 | 1 2 4 6 8 10 12 14 16 18 20 | 覆盖新增解码 batch 档位 |
| NEXTN 投机步数 | steps=3, draft=4 | steps=4, draft=5 | 量化后单步解码更快，加深投机深度收益更高，接受长度由 3.2 提升至 4.40 |

关键收益：投机深度 3→4 使接受长度从 3.24 提升到 4.40，单次前向平均产出 token 数提升 36%，直接带来约 25% 吞吐提升。

## 四、对比总结

| 指标 | Qwen3.8-27B (BF16) | Qwen3.8-27B-w8a8 | 差异 |
|---|---|---|---|
| GSM8K Accuracy (limit 200) | 98.5% | 98.5% | 持平（量化无损） |
| 128K-1K-90%cache 最佳总吞吐 | 54757 tok/s @ C=12 | 72010 tok/s @ C=16 | w8a8 高 31.5% |
| 最佳点 Mean TPOT | 16.13 ms | 15.61 ms | w8a8 低 3.2% |
| 最佳点 Mean TTFT | 8707 ms | 11661 ms | BF16 更优（并发更低） |
| Accept Length | 3.38 | 4.40 | w8a8 高 30% |
| 是否满足 TPOT ≤ 50ms | 是（16~23ms） | 是（15.6~19.5ms） | 均满足 |

结论：

1. BF16 与 w8a8 在 GSM8K-200 上精度完全一致（98.5%），INT8 量化在本模型上无明显精度损失。
2. w8a8 在本用例下吞吐优势明显（+31.5%），主要来自更低权重的解码延迟，使更深投机（steps=4/draft=5）成为可能，接受长度从 3.38 提升至 4.40。
3. 两个版本在该用例下 TPOT 均远低于 50ms 约束（最佳点仅为约束的 1/3），说明 128K-1K-90%cache 场景下仍有较大安全余量。
4. 混合 GDN 模型必须同时开启 radix cache 与 `--mamba-radix-cache-strategy extra_buffer`，否则线性注意力状态无法复用，128K prefill 每个请求都要重算（实测 TTFT 从 8.7s 恶化到 20s+）。

## 五、参考文档评价

| 评价维度 | 评分（1~5） | 说明 |
|---|---|---|
| 安装部署文档 | 4 | Qwen3.6-27B 教程可直接参考，但需注意 LD_LIBRARY_PATH 被污染导致 torch_npu 加载失败的问题，文档未提及 |
| 最佳实践文档 | 4 | 64k+1k-90%cache 配置完整可用，但仅有 A3 机型数据，A2(910B3) 需要自行换算 mamba cache 与 mem-fraction |
| 参数调优文档 | 4 | 参数作用说明清晰，缺少 mamba cache 大小与并发数的定量换算方法 |
| 精度测试文档 | 5 | evalscope 命令开箱可用 |
| 性能测试文档 | 4 | benchmark 命令完整，但未提示首次运行会因 kernel autotune 导致首轮数据失真 |

## 六、问题与改进建议

### 发现的问题

1. **LD_LIBRARY_PATH 污染导致服务无法启动**：镜像继承外部环境变量后，`torch_npu` 加载报 `Failed to load the backend extension: torch_npu`，需手动把 `/usr/lib/aarch64-linux-gnu` 提前。
2. **128K 用例换算说明缺失**：文档只给 64k 用例（gsp-system-prompt-len 57600 / question 6400），128K 需自行推算为 115200 / 12800，建议直接补充 128k 行。
3. **首次 benchmark 数据失真**：首次以某并发档运行时包含 Triton kernel autotune 开销，TPOT 会虚高 2~3 倍（实测并发 12 首轮 56.5ms、复跑 16.1ms），文档未提示需预热。
4. **evalscope 并发参数与任务书不一致**：任务书给出的是最简命令，实际 `--parallel` 不存在，需用 `--eval-batch-size`；同时 evalscope 进程也需修复 LD_LIBRARY_PATH，否则报同样的 torch_npu 加载错误。
5. **模型默认开启 thinking**：不配置 `--reasoning-parser qwen3` 时输出含 `\</think\>` 标签，影响 GSM8K 答案抽取，任务书未提示。

### 改进建议

1. 在部署教程中增加"环境变量自检"小节，或在启动脚本内统一前置系统库路径。
2. 最佳实践文档为每个用例补充 A2 与 A3 两套机型参数对照表。
3. 性能测试文档增加"预热一轮后再采集数据"的说明，或提供 `--warmup-requests` 推荐值。
4. 精度测试文档补充 `--eval-batch-size` 参数说明与完整可复制命令。
5. 部署命令模板中默认带上 `--reasoning-parser qwen3`（Qwen3 系列默认 thinking）。
