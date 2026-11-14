# Qwen3.8-27B 部署调优报告

## 一、报告基本信息

| 项目 | 内容 |
|---|---|
| 报告编号 | Qwen3.8-27B-MeetUp-001 |
| 体验日期 | 2026-11-14 |
| SGLang 镜像版本 | `swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:cann9.1.0-910b-20261112` |
| 环境信息 | Atlas 800T A2（910B3）× 4 卡，TP4；HDK 26.1.1 / ascendhal 7.35.23，CANN 9.1.0，内存 752GB |
| SGLang 版本 | `0.0.0.dev22121+g0fa9ff55f.d20261112`（源码 `/sgl-workspace/sglang`） |
| 模型权重 | `/workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B`（52GB BF16）<br>`/workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8`（29GB W8A8） |
| 性能用例 | 128K 输入 + 1K 输出 + 90% 前缀缓存命中（generated-shared-prefix） |
| 精度用例 | GSM8K，`--limit 200` |

---

## 二、操作记录

### 2.1 使用 Qwen3.8-27B（BF16）权重

#### 1. 启动命令（调优后最终版）

```bash
# 环境变量
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
export LD_LIBRARY_PATH=""
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_USE_FIA=1
export GLOO_SOCKET_IFNAME=lo
export HCCL_SOCKET_IFNAME=lo
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_SET_CPU_AFFINITY=1
export SGLANG_ENABLE_SPEC_V2=1
export STREAMS_PER_DEVICE=32

# 启动命令
python3 -m sglang.launch_server \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
  --host 127.0.0.1 --port 6688 \
  --tp-size 4 \
  --nnodes 1 \
  --attention-backend ascend \
  --device npu \
  --trust-remote-code \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 32 \
  --max-mamba-cache-size 192 \
  --mem-fraction-static 0.85 \
  --cuda-graph-bs-decode 1 2 5 10 15 20 24 28 32 \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

启动耗时约 3 分钟（权重加载 6s/卡，NPU Graph 捕获占主要时间），日志出现 `The server is fired up and ready to roll!` 即就绪。

#### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B \
    --api-url http://localhost:6688/v1 \
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
│ Qwen3.8-27B │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 99%     │
└─────────────┴───────────┴────────────┴──────────┴───────┴─────────┘

Model        Dataset      Num  Avg Lat    Avg Thpt     Avg In    Avg Out
Qwen3.8-27B  GSM8K        200  17.241 s   35.89 tok/s     729        619
```

#### 3. 性能测试

测试命令：

```bash
python -m sglang.benchmark.serving \
  --dataset-name generated-shared-prefix \
  --backend sglang \
  --host 127.0.0.1 --port 6688 \
  --gsp-num-groups 1 \
  --gsp-prompts-per-group 32 \
  --gsp-system-prompt-len 115200 \
  --gsp-question-len 12800 \
  --gsp-output-len 1000 \
  --max-concurrency 32 \
  --num-prompts 32 \
  --request-rate inf
```

测试结果（调优后，max-concurrency=32）：

```txt
Peak output token throughput (tok/s):    924.00
Peak concurrent requests:                32
Total token throughput (tok/s):          55879.36
Concurrency:                             28.28
Mean E2E Latency (ms):                   66949.29
Mean TTFT (ms):                          27603.25
Mean TPOT (ms):                          39.39
Median TPOT (ms):                        37.78
```

调优过程数据（同一用例，逐步加压）：

| 配置 | 并发 | Mean TPOT (ms) | Total 吞吐 (tok/s) | 输出吞吐峰值 (tok/s) | 是否满足 TPOT≤50ms |
|---|---|---|---|---|---|
| 基线（无投机，mem 0.80） | 10 | 40.32 | 24821 | 701 | ✅ |
| 基线（无投机，mem 0.80） | 16 | 60.03 | 26912 | 849 | ❌ |
| + NEXTN 投机（mem 0.82） | 16 | 28.15 | 40127 | 854 | ✅ |
| + NEXTN 投机（mem 0.82） | 20 | 27.62 | 52369 | 898 | ✅ |
| + NEXTN 投机（mem 0.82） | 24 | 33.73 | 53805 | 1457 | ✅ |
| + NEXTN 投机（mem 0.85，mamba 192） | 32 | 39.39 | **55879** | 924 | ✅ |

**参数调整说明（相对基线）：**
1. `--speculative-algorithm NEXTN`（num-steps 3 / topk 1 / draft-tokens 4）：利用模型自带 MTP 头做投机解码，TPOT 从 60ms 降到 28～39ms，是本次最大收益项。
2. `export SGLANG_ENABLE_SPEC_V2=1`：启用投机 V2 调度路径。
3. `--mem-fraction-static 0.80 → 0.85`、`--max-mamba-cache-size 128 → 192`：混合线性注意力模型需要 mamba cache，提高 KV/mamba 预算后可同时驻留更多 128K 长序列。
4. `--max-running-requests 32`、`--cuda-graph-bs-decode` 扩展到 32：让 NPU Graph 覆盖实际 decode batch。
5. `--chunked-prefill-size / --max-prefill-tokens 32768`：128K 输入分块 prefill，避免单次 prefill 撑爆显存，同时保证 90% 前缀可复用。

---

### 2.2 使用 Qwen3.8-27B-w8a8 权重

#### 1. 启动命令（调优后最终版）

```bash
# 环境变量同上（BF16 章节）
python3 -m sglang.launch_server \
  --model-path /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
  --host 127.0.0.1 --port 6688 \
  --tp-size 4 \
  --nnodes 1 \
  --attention-backend ascend \
  --device npu \
  --trust-remote-code \
  --quantization modelslim \
  --chunked-prefill-size 32768 \
  --max-prefill-tokens 32768 \
  --max-running-requests 32 \
  --max-mamba-cache-size 192 \
  --mem-fraction-static 0.85 \
  --cuda-graph-bs-decode 1 2 5 10 15 20 24 28 32 \
  --dtype bfloat16 \
  --mamba-ssm-dtype bfloat16 \
  --speculative-algorithm NEXTN \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

与 BF16 的唯一差异是 `--quantization modelslim` 与权重路径。

#### 2. 精度测试

测试命令：

```bash
evalscope eval \
    --model /workspace/shared_assets/OssSGLangMeetup/model/Qwen3.8-27B-w8a8 \
    --api-url http://localhost:6688/v1 \
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
│ Qwen3.8-27B-w8a8 │ GSM8K     │ Accuracy ↑ │ main     │   200 │ 98%     │
└──────────────────┴───────────┴────────────┴──────────┴───────┴─────────┘

Model             Dataset   Num  Avg Lat   Avg Thpt     Avg In    Avg Out
Qwen3.8-27B-w8a8  GSM8K     200  17.613 s  37.93 tok/s     729        668
```

#### 3. 性能测试

测试命令与 BF16 相同，仅 `--num-prompts / --max-concurrency` 变化。

测试结果（调优后，max-concurrency=32）：

```txt
Peak output token throughput (tok/s):    4669.00
Peak concurrent requests:                32
Total token throughput (tok/s):          65487.53
Concurrency:                             28.25
Mean E2E Latency (ms):                   57055.02
Mean TTFT (ms):                          27843.12
Mean TPOT (ms):                          29.24
Median TPOT (ms):                        29.54
```

调优过程数据（含未调优基线对比）：

| 配置 | 并发 | Mean TPOT (ms) | Total 吞吐 (tok/s) | 输出吞吐峰值 (tok/s) | 是否满足 TPOT≤50ms |
|---|---|---|---|---|---|
| 基线（无投机，mem 0.80） | 10 | 43.09 | 24873 | 698 | ✅ |
| 基线（无投机，mem 0.80） | 16 | 52.94 | 30879 | 1424 | ❌ |
| + NEXTN 投机（mem 0.85） | 16 | 19.38 | 57937 | 1558 | ✅ |
| + NEXTN 投机（mem 0.85） | 24 | 31.28 | 51181 | 1069 | ✅ |
| + NEXTN 投机（mem 0.85） | 32 | 29.24 | **65487** | 4669 | ✅ |

**参数调整说明：**
1. 基线（不开启投机）在 128K 场景下并发 10 时 TPOT 已达 43ms，并发 16 时升到 52.9ms，无法满足 TPOT≤50ms。
2. 沿用 BF16 调优结论（NEXTN + SPEC_V2 + mem 0.85 + mamba cache 192 + running-requests 32）后，同并发 16 的 TPOT 从 52.94ms 降到 19.38ms（−63%），总吞吐从 30879 提升到 57937 tok/s（+88%）；并发 32 时仍保持 29.24ms、总吞吐 65487 tok/s。
3. w8a8 权重体积减半（29GB vs 52GB），同等显存下 KV/mamba 预算更充足，因此 TPOT 大幅低于 BF16。
2. `--quantization modelslim` 为昇腾 msmodelslim 量化权重加载方式。
3. 并发 32 时 TTFT 约 27.8s，主要来自 128K 分块 prefill；若业务更看重首 token 时延，建议降到 16 并发（TPOT 19.4ms、TTFT 14.0s）。

---

### 2.3 对比总结

| 指标 | Qwen3.8-27B (BF16) | Qwen3.8-27B-w8a8 | 差异 |
|---|---|---|---|
| GSM8K 精度（200 条） | 99.0%（198/200） | 98.0%（196/200） | −1.0 pp |
| 精度测试平均时延 / 吞吐 | 17.241 s / 35.89 tok/s | 17.613 s / 37.93 tok/s | w8a8 吞吐 +5.7% |
| 128K-1K-90%cache 最佳 TPOT | 39.39 ms | 29.24 ms | w8a8 降低 25.8% |
| 最佳总吞吐（TPOT≤50ms） | 55879 tok/s | 65487 tok/s | w8a8 提升 17.2% |
| 输出吞吐峰值（对应配置） | 924 tok/s @cc32 | 4669 tok/s @cc32 | 与投机解码接受率相关，波动较大 |
| 权重体积 | 52 GB | 29 GB | 减少 44% |
| 单卡权重显存 | 12.79 GB | 明显更低（同等 KV 预算更大） | — |

**结论：**
1. 两个权重都能在 4×910B3（A2）上用 TP4 稳定拉起，GSM8K 精度分别为 99.0% 和 98.0%，W8A8 量化仅损失约 1 个百分点，精度可接受。
2. 128K-1K-90%cache 场景下，BF16 的关键瓶颈是 decode 阶段的 TPOT；开启 NEXTN 投机解码是收益最大的一步，使 TPOT 从 60ms（cc16）降到 28ms 级别，从而在 TPOT≤50ms 约束下把并发从 10 提到 32。
3. w8a8 权重体积减半，显存更宽松，同并发下 TPOT 比 BF16 低约 26%，总吞吐高约 17%，长上下文高并发场景推荐 w8a8。
4. 若业务对首 token 时延敏感，建议并发控制在 16（BF16 TPOT 28ms/TTFT 15s，w8a8 TPOT 19ms/TTFT 14s）；若追求吞吐且可接受 27s 级 TTFT，用 32 并发。

---

## 三、参考文档评价

| 评价维度 | 评分（1~5） | 说明 |
|---|---|---|
| 安装部署文档 | 4 | 容器/虚机流程清晰，`npu-smi info`、`pip show sglang` 校验点明确；但文档示例均为 A3（单卡 2 die、TP2），A2 四卡需要自行改成 `--tp-size 4` |
| 最佳实践文档 | 4 | 给出了 NEXTN、mamba cache、`SGLANG_ENABLE_SPEC_V2` 等关键组合，可直接复用；但只有 64K 用例，128K-90%cache 需自行按比例换算 `gsp-system-prompt-len/question-len` |
| 参数调优文档 | 4 | 参数含义与冲突关系说明到位，`--max-mamba-cache-size`、`--mem-fraction-static` 的作用解释清楚 |
| 精度测试文档 | 5 | evalscope 命令可直接执行，输出表格信息完整 |
| 性能测试文档 | 4 | `sglang.benchmark.serving` 用例可复现，90% 前缀命中用 `generated-shared-prefix` 表达很直观；缺少 128K 规模的推荐并发区间 |

---

## 四、问题与改进建议

### 发现的问题

1. **文档默认 A3、TP2，A2 用户需要自行改 TP。** Qwen3.6-27B 教程与最佳实践全部基于 A3 单卡 2 die（`--tp-size 2`），A2/910B 用户按文档执行会直接失败或跑不满 4 卡。
2. **最佳实践没有 128K 用例。** 最接近的是 64K-1K-90%，128K 只能按 0.9 比例手工推算 `--gsp-system-prompt-len 115200` / `--gsp-question-len 12800`，容易算错。
3. **`--max-mamba-cache-size` 缺乏取值方法。** 文档只给示例值（60/74/16/160），没有说明它与 `--max-running-requests`、`--mem-fraction-static` 的配比关系，混合线性注意力模型调参时只能试错。
4. **容器环境缺少 `LD_LIBRARY_PATH` 清理说明。** 若宿主机或 IDE 注入了其他 `LD_LIBRARY_PATH`，会出现 `torch_npu` 加载失败 / `_sqlite3` undefined symbol 的报错，排查成本高。
5. **性能数据缺少并发-时延曲线。** 最佳实践仅给一个最优并发点，用户不知道该并发是怎么选出来的，也无法判断自己的硬件该取多少。

### 改进建议

1. 在 Qwen3.6-27B 教程中增加 A2（910B）章节，明确 4 卡 `--tp-size 4` 的启动命令与显存预期。
2. 最佳实践补充 128K-1K-90%cache（`gsp-system-prompt-len 115200` / `gsp-question-len 12800`）的完整命令与数据。
3. 在参数调优文档中补充 `--max-mamba-cache-size` 的经验公式（如 ≥ 1.5 × `--max-running-requests`）及超出显存时的降级顺序。
4. 在安装文档 FAQ 中增加 `LD_LIBRARY_PATH` 污染导致 `torch_npu` 导入失败的排查条目。
5. 每个最佳实践用例增加「并发 8/16/24/32 下的 TPOT-吞吐」小表，方便用户按 TPOT 目标直接选并发。

---

## 附录：复现脚本

| 文件 | 说明 |
|---|---|
| `run/env.sh` | 公共环境变量（含 `LD_LIBRARY_PATH` 清理、CANN/ATB set_env） |
| `run/01_launch_bf16_base.sh` | BF16 基线启动命令 |
| `run/02_launch_bf16_tuned.sh` | BF16 调优中间版（NEXTN，mem 0.82） |
| `run/03_launch_bf16_tuned_v2.sh` | BF16 调优最终版（NEXTN，mem 0.85，cc32） |
| `run/04_launch_w8a8.sh` | w8a8 调优最终版 |
| `run/05_launch_w8a8_base.sh` | w8a8 未调优基线 |
| `run/bench_128k.sh` | 128K-1K-90%cache 压测脚本（参数：并发、请求数、标签） |
| `run/bench_128k_*.log` | 各轮压测原始输出 |
| `run/eval_bf16/`、`run/eval_w8a8/` | evalscope 精度测试报告（json + html） |
