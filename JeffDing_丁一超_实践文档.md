# DeepSeek-V4-Flash部署调优报告

## 一、报告基本信息

| 项目 | 内容 |
|---|---|
| 报告编号 | JeffDing-001|
| 体验日期 | 2026-11-14 |
| SGLang镜像版本 | swr.cn-southwest-2.myhuaweicloud.com/base_image/dockerhub/lmsysorg/sglang:main-cann9.1.0-950-B105 |
| 环境信息 | 4*950PR |

---

## 二、操作记录

### TPOT=50ms场景

#### 1. 启动命令

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
export INF_NAN_MODE_FORCE_DISABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=32
export SGLANG_DSPARK_A5_EXTRA_OPS_SO=/usr/local/Ascend/extra-ops/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_NPU_USE_MULTI_STREAM=1
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

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 127.0.0.1 --port 6688 \
    --mem-fraction-static 0.72 \
    --max-running-requests 16 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --cuda-graph-bs-decode 1 2 4 6 8 \
    --enable-dp-lm-head \
    --enable-dp-attention --dp-size 4 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --reasoning-parser deepseek-v4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 6 \
    --speculative-dspark-block-size 5
```

#### 2. 性能测试

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
    --max-concurrency 16 \
    --num-prompts 64 \
    --warmup-requests 8 \
    --request-rate inf \
    --seed 1
```

测试结果：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 16        
Successful requests:                     64        
Benchmark duration (s):                  197.33    
Total input tokens:                      8374342   
Total input text tokens:                 8374342   
Total generated tokens:                  64000     
Total generated tokens (retokenized):    63898     
Request throughput (req/s):              0.32      
Input token throughput (tok/s):          42439.21  
Output token throughput (tok/s):         324.34    
Peak output token throughput (tok/s):    2360.00   
Peak concurrent requests:                23        
Total token throughput (tok/s):          42763.55  
Concurrency:                             15.40     
Accept length:                           5.72      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   47479.15  
Median E2E Latency (ms):                 37199.86  
P90 E2E Latency (ms):                    78724.05  
P95 E2E Latency (ms):                    99077.61  
P99 E2E Latency (ms):                    108139.60 
---------------Time to First Token----------------
Mean TTFT (ms):                          15832.93  
Median TTFT (ms):                        8910.36   
P90 TTFT (ms):                           40502.43  
P95 TTFT (ms):                           40510.61  
P99 TTFT (ms):                           40519.32  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          31.68     
Median TPOT (ms):                        26.48     
P90 TPOT (ms):                           52.23     
P95 TPOT (ms):                           62.45     
P99 TPOT (ms):                           90.72     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           31.70     
Median ITL (ms):                         6.85      
P90 ITL (ms):                            10.52     
P95 ITL (ms):                            58.55     
P99 ITL (ms):                            664.02    
Max ITL (ms):                            27724.73  
==================================================
```

相对原命令的改动及依据

| 参数 | 原值 | 新值 | 原因 |
|---|---|---|---|
| `--disable-radix-cache` | 开启 | 删除 | 90% 共享前缀必须靠 radix cache 命中；关闭后每条请求全量 prefill 128k，用例失真 |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 160 | 32 | 官方公式：per-rank 并发 × draft = (16/4)×6 = 24，32 留余量；160 每步 padding，抬高 TPOT |
| `--max-running-requests` | 64 | 16 | 128k KV 占用大；dp4 下每 rank 4 并发，TPOT 更稳 |
| `--chunked-prefill-size` | 65536 | 32768 | 长 prefill 切片，减少对 decode 的阻塞；降低激活显存 |
| `--max-prefill-tokens` | 65536 | 32768 | 与 chunk 对齐；90% 命中后实际仅约 12800 token，单批完成 |
| `--cuda-graph-bs-decode` | 1 2 4 8 16 | 1 2 4 6 8 | 按 per-rank 4 并发覆盖并留扫描余量，减少 graph 捕获显存 |
| `SGLANG_NPU_USE_MULTI_STREAM` | 未设 | 1 | 官方 DeepSeek-V4-Flash 配置项，共享专家与路由专家重叠执行 |
| `--moe-a2a-backend` / `--deepep-mode` | 未显式 | deepep / auto | 对齐官方 V4-Flash 8P 混合部署配置 |
| `INF_NAN_MODE_FORCE_DISABLE` | 未设 | 1 | 官方 V4-Flash 配置项，关闭 INF/NAN 检查开销 |

未加入的项：

- `HCCL_OP_EXPANSION_MODE=AIV`：官方 V4-Flash 脚本有，但文档明确 Atlas A2 + DeepEP 下会报未知算子错误。确认是 A3 再加。
- `--quantization modelslim` / `--kv-cache-dtype bfloat16`：原命令能正常启动，说明权重加载方式已确定，避免引入启动风险。
- `SGLANG_RAGGED_VERIFY_MODE=static`、`SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`、`ZBAL_*` / `SGLANG_ZBAL_*`：官方 8P 配置项，但与当前版本/部署形态强相关，先不动，作为后续候选。

### 3. 精度测试命令
```bash
sgl-eval run gsm8k \
 --model /workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731 \
 --base-url http://localhost:6688/v1 \
 --num-examples 10 --max-tokens 2048
```

### TPOT=20ms场景

#### 1. 启动命令

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
export INF_NAN_MODE_FORCE_DISABLE=1
export LD_LIBRARY_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16
export SGLANG_DSPARK_A5_EXTRA_OPS_SO=/usr/local/Ascend/extra-ops/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so
export SGLANG_DSPARK_FAST_KERNEL=0
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
export SGLANG_ENABLE_SPEC_V2=1
export SGLANG_ENABLE_WAR_BARRIER=1
export SGLANG_FORCE_COARSE_WAR_BARRIER=1
export SGLANG_NPU_USE_MULTI_STREAM=1
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

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --page-size 128 \
    --tp-size 4 \
    --trust-remote-code \
    --device npu \
    --attention-backend dsv4 \
    --watchdog-timeout 9000 \
    --host 127.0.0.1 --port 6688 \
    --mem-fraction-static 0.72 \
    --max-running-requests 8 \
    --chunked-prefill-size 32768 \
    --max-prefill-tokens 32768 \
    --cuda-graph-bs-decode 1 2 4 6 8 \
    --enable-dp-lm-head \
    --enable-dp-attention --dp-size 4 \
    --moe-a2a-backend deepep \
    --deepep-mode auto \
    --reasoning-parser deepseek-v4 \
    --speculative-algorithm DSPARK \
    --speculative-draft-attention-backend ascend \
    --speculative-num-draft-tokens 7 \
    --speculative-dspark-block-size 6
```

#### 2. 性能测试

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
    --max-concurrency 16 \
    --num-prompts 64 \
    --warmup-requests 8 \
    --request-rate inf \
    --seed 1
```

测试结果：

```txt
============ Serving Benchmark Result ============
Backend:                                 sglang    
Traffic request rate:                    inf       
Max request concurrency:                 8         
Successful requests:                     32        
Benchmark duration (s):                  96.69     
Total input tokens:                      4187102   
Total input text tokens:                 4187102   
Total generated tokens:                  32000     
Total generated tokens (retokenized):    31988     
Request throughput (req/s):              0.33      
Input token throughput (tok/s):          43302.62  
Output token throughput (tok/s):         330.94    
Peak output token throughput (tok/s):    1616.00   
Peak concurrent requests:                12        
Total token throughput (tok/s):          43633.56  
Concurrency:                             7.65      
Accept length:                           6.09      
----------------End-to-End Latency----------------
Mean E2E Latency (ms):                   23109.12  
Median E2E Latency (ms):                 21936.40  
P90 E2E Latency (ms):                    30343.88  
P95 E2E Latency (ms):                    32639.82  
P99 E2E Latency (ms):                    43799.95  
---------------Time to First Token----------------
Mean TTFT (ms):                          5738.50   
Median TTFT (ms):                        4455.00   
P90 TTFT (ms):                           10645.66  
P95 TTFT (ms):                           10653.59  
P99 TTFT (ms):                           11744.50  
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.39     
Median TPOT (ms):                        16.69     
P90 TPOT (ms):                           25.28     
P95 TPOT (ms):                           26.79     
P99 TPOT (ms):                           35.00     
---------------Inter-Token Latency----------------
Mean ITL (ms):                           17.40     
Median ITL (ms):                         5.30      
P90 ITL (ms):                            8.96      
P95 ITL (ms):                            46.28     
P99 ITL (ms):                            335.14    
Max ITL (ms):                            8299.24   
==================================================
```
相对 TPOT ≤ 50ms 配置的改动

| 参数 | TPOT50 配置 | TPOT20 配置 | 原因 |
|---|---|---|---|
| `--max-running-requests` | 16 | **8** | dp4 后每 rank 2 并发，decode batch 减半，单步时间显著下降 |
| `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK` | 32 | **16** | 官方公式：2 × 7 = 14，16 留最小余量；过大时每步 padding 抬高 TPOT |
| `--speculative-num-draft-tokens` | 6 | **7** | 对齐官方 V4-Flash 8P 值，单步多验证 1 个 token，摊薄 TPOT |
| `--speculative-dspark-block-size` | 5 | **6** | 与 draft tokens 配套（block + 1 = draft tokens） |
| `SGLANG_SCHEDULER_SKIP_ALL_GATHER` | 未设 | **保持不设/0** | 实测本镜像开启后 DSPARK verify 报 `DP CUDA graph replay requires raw per-rank request counts`，不可用 |

保持不变的项：`--chunked-prefill-size 32768`、`--max-prefill-tokens 32768`、`--mem-fraction-static 0.72`、`--cuda-graph-bs-decode 1 2 4 6 8`、radix cache 保持开启（**不要加回 `--disable-radix-cache`**）。

### 3. 精度测试命令
```bash
sgl-eval run gsm8k \
 --model /workspace/shared_assets/OssSGLangMeetup/model/DeepSeek-V4-Flash-0731 \
 --base-url http://localhost:6688/v1 \
 --num-examples 10 --max-tokens 2048
```

---

## 三、参考文档评价

| 评价维度 | 评分（1~5） | 说明 |
|---|---|---|
| 安装部署文档 | | |
| 最佳实践文档 | | |
| 参数调优文档 | | |
| ...| | |

---

## 四、问题与改进建议

### 发现的问题

1. 根据活动给出的操作文档操作，文档中在拉起服务的时候使用的是`python3`，但是在hidevlab环境中可能会报错找不到sglang，但是实际是安装了sglang。这个问题解决方法是要使用`python`启动部署
2. 
3. 

### 改进建议

1. 暂无
2. 
3. 
