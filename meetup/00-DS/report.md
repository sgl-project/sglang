# GLM5.2-w4a8部署调优报告

## 一、报告基本信息

| 项目 | 内容 |
|---|---|
| 报告编号 | 待填 |
| 体验日期 | 2026-11-14 |
| SGLang镜像版本 | 待填（完整镜像tag） |
| 环境信息 | 待填（A2/A3/A5，卡数；脚本已知配置：Ascend NPU，TP=8、DP=2，ascend attention backend，DeepEP auto，NEXTN 投机解码，modelslim W4A8） |

---

## 二、操作记录

### TPOT=50ms场景

#### 1. 启动命令

```bash
待补充
```

#### 2. 精度测试

测试命令：

```bash
待补充
```

测试结果：

```txt
待补充
```

#### 3. 性能测试

测试命令：

```bash
待补充
```

测试结果：

```txt
待补充
```

### TPOT=20ms场景

<!-- 本场景数据来自实际运行：压测输出 Mean TPOT 20.25ms，与 20ms 目标吻合 -->

#### 1. 启动命令

```bash
# high performance cpu
echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG_SET_CPU_AFFINITY=1

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING
# cann
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export STREAMS_PER_DEVICE=32
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export DEEPEP_HCCL_BUFFSIZE=1000
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo


MODEL_PATH=/home/weights/GLM-5.2-w4a8
export DEEPEP_NORMAL_LONG_SEQ_ROUND=72
export DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS=1024
export DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ=1

export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16

export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
export DEEPEP_HYBRID_DEPLOYMENT=1
python3 -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --attention-backend ascend \
        --device npu \
        --tp-size 8 --dp-size 2 \
        --nnodes 1 \
	--disable-shared-experts-fusion \
        --chunked-prefill-size 8192 \
        --max-prefill-tokens 280000 \
        --trust-remote-code \
        --mem-fraction-static 0.90 \
        --served-model-name GLM-5.2-w4a8 \
        --max-running-requests 16 \
	--cuda-graph-max-bs-decode 16 \
        --quantization modelslim \
        --moe-a2a-backend deepep --deepep-mode auto \
        --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4  \
        --load-balance-method round_robin \
        --device npu --host 127.0.0.1 --port 5599
```

#### 2. 精度测试

测试命令：

```bash
待补充
```

测试结果：

```txt
待补充
```

#### 3. 性能测试

测试命令：

```bash
#!/usr/bin/env bash
set -e

# 修改这里的 DP 数、压测并发数和服务地址。
DP_SIZE=1
CONCURRENCY=1
BENCH_HOST=127.0.0.1
BENCH_PORT=5599

MODEL_PATH=/home/weights/GLM-5.2-w4a8
DATASET_DIR=/home/datasets/completion-accept3-128k-prefix90-4096

# 保留 prefix cache，避免 SGLang 在 CI 模式下自动 flush。
export SGLANG_IS_IN_CI=false

# 第一阶段：服务端启用 prefix cache 和 round_robin，DP_SIZE 与实际 DP 数一致。
# 用 DP_SIZE * 2 条请求预热各 rank 的 KV cache，每条生成 4 token，丢弃预热结果。
echo "正在预热 prefix 到各个 DP rank：DP_SIZE=$DP_SIZE，请求数=$((DP_SIZE * 2))..."
python -m sglang.benchmark.serving \
  --backend sglang \
  --host "$BENCH_HOST" --port "$BENCH_PORT" \
  --model "$MODEL_PATH" \
  --served-model-name GLM-5.2-w4a8 \
  --tokenizer "$MODEL_PATH" \
  --dataset-name custom \
  --dataset-path "$DATASET_DIR/requests-64.jsonl" \
  --sharegpt-output-len 4 \
  --num-prompts "$((DP_SIZE * 2))" \
  --max-concurrency "$DP_SIZE" \
  --temperature 1.1 --top-p 0.95 \
  --seed 42 --warmup-requests 0 \
  --output-file /dev/null \
  > /dev/null 2>&1

# 第二阶段：不 flush cache，发送 CONCURRENCY * 4 条请求，输出正式结果。
echo "Prefix 预热完成，开始正式压测：并发=$CONCURRENCY，请求数=$((CONCURRENCY * 4))..."
python -m sglang.benchmark.serving \
  --backend sglang \
  --host "$BENCH_HOST" --port "$BENCH_PORT" \
  --model "$MODEL_PATH" \
  --served-model-name GLM-5.2-w4a8 \
  --tokenizer "$MODEL_PATH" \
  --dataset-name custom \
  --dataset-path "$DATASET_DIR/requests.jsonl" \
  --sharegpt-output-len 1024 \
  --num-prompts "$((CONCURRENCY * 4))" \
  --max-concurrency "$CONCURRENCY" \
  --temperature 1.1 --top-p 0.95 \
  --seed 42 --warmup-requests 0
```

测试结果：

```txt
#Input tokens: 524288
#Output tokens: 4096
Starting warmup with 0 sequences...
Warmup completed with 0 sequences. Starting main benchmark run...
100%|██████████| 4/4 [01:54<00:00, 28.63s/it]

============ Serving Benchmark Result ============
Backend:                                 sglang
Traffic request rate:                    inf
Max request concurrency:                 1
Successful requests:                     4
Benchmark duration (s):                  114.53
Total input tokens:                      524288
Total input text tokens:                 524288
Total generated tokens:                  4096
Total generated tokens (retokenized):    4089
Request throughput (req/s):              0.03
Input token throughput (tok/s):          4577.63
Output token throughput (tok/s):         35.76
Peak output token throughput (tok/s):    67.00
Peak concurrent requests:                2
Total token throughput (tok/s):          4613.40
Concurrency:                             1.00
Accept length:                           2.75
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   28626.70
Median E2E Latency (ms):                 28427.62
P90 E2E Latency (ms):                    32527.91
P95 E2E Latency (ms):                    33392.36
P99 E2E Latency (ms):                    34083.92
---------------Time to First Token----------------
Mean TTFT (ms):                          7906.89
Median TTFT (ms):                        7910.24
P90 TTFT (ms):                           7978.53
P95 TTFT (ms):                           7985.13
P99 TTFT (ms):                           7990.40
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          20.25
Median TPOT (ms):                        20.00
P90 TPOT (ms):                          24.11
P95 TPOT (ms):                          24.98
P99 TPOT (ms):                          25.67
---------------Inter-Token Latency----------------
Mean ITL (ms):                           20.25
Median ITL (ms):                         15.07
P90 ITL (ms):                            30.14
P95 ITL (ms):                            30.54
P99 ITL (ms):                            60.00
Max ITL (ms):                            61.01
==================================================
```

---

## 三、参考文档评价

| 评价维度 | 评分（1~5） | 说明 |
|---|---|---|
| 安装部署文档 | 待填 | |
| 最佳实践文档 | 待填 | |
| 参数调优文档 | 待填 | |
| ...| | |

---

## 四、问题与改进建议

### 发现的问题

1. 待补充
2. 
3. 

### 改进建议

1. 待补充
2. 
3. 
