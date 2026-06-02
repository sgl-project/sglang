# Hybrid PD 测试方案: 3P1D vs 3(P+D)+1D 对比测试

## 测试目标

通过设计不同的流量 pattern，对比 **3P1D 完全分离** 和 **3(P+D)+1D 混合模式** 在以下维度的表现：

- TTFT (Time to First Token)
- ITL (Inter-Token Latency)
- 吞吐量 (tok/s)
- P99 延迟
- 资源利用率 (GPU Utilization, KV Cache Usage)
- 网络开销 (KV Transfer 带宽)

---

## 测试环境配置

### 硬件要求

```
4 台同构 GPU 节点 (确保公平对比)
├── Node 1-3: 计算节点
└── Node 4: Decode 节点
网络: RDMA (推荐) 或 TCP (fallback)
```

### 部署配置 A: 3P1D 完全分离

```bash
# === Node 1/2/3: Prefill Only ===
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --host 0.0.0.0 --port 8000 \
    --tp-size $TP \
    --max-running-requests 64 \
    --mem-fraction-static 0.88

# === Node 4: Decode Only ===
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --host 0.0.0.0 --port 8000 \
    --tp-size $TP \
    --max-running-requests 64 \
    --mem-fraction-static 0.88

# === Router ===
python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://$NODE1:8000 8998 \
    --prefill http://$NODE2:8000 8998 \
    --prefill http://$NODE3:8000 8998 \
    --decode http://$NODE4:8000 \
    --policy cache_aware \
    --port 30000
```

### 部署配置 B: 3(P+D)+1D 混合模式

```bash
# === Node 4: Decode Only (先启动) ===
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --host 0.0.0.0 --port 8000 \
    --tp-size $TP \
    --max-running-requests 64 \
    --mem-fraction-static 0.88

# === Node 1/2/3: Hybrid P+D ===
python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --disaggregation-mode hybrid \
    --disaggregation-transfer-backend mooncake \
    --disaggregation-bootstrap-port 8998 \
    --hybrid-external-decode-addresses "http://$NODE4:8000" \
    --hybrid-offload-watermark 0.80 \
    --hybrid-local-decode-limit 32 \
    --hybrid-long-output-threshold 1024 \
    --host 0.0.0.0 --port 8000 \
    --tp-size $TP \
    --max-running-requests 64 \
    --mem-fraction-static 0.88

# === Router ===
python -m sglang_router.launch_router \
    --hybrid-mode \
    --hybrid-node http://$NODE1:8000 \
    --hybrid-node http://$NODE2:8000 \
    --hybrid-node http://$NODE3:8000 \
    --hybrid-decode http://$NODE4:8000 \
    --policy round_robin \
    --port 30000
```

---

## 流量 Pattern 设计

### Pattern 1: 短对话 (ChatBot 场景)

**特征:** 短 input + 短 output, 高并发, 对 TTFT 敏感

```
┌─────────────────────────────────────┐
│ Pattern 1: Short Chat               │
│                                     │
│ Input:  256 tokens  (用户问题)       │
│ Output: 128 tokens  (简短回复)       │
│ QPS:    20 req/s                    │
│ 并发:    32                          │
│ 总请求:  500                         │
│ 持续时间: ~25s                       │
└─────────────────────────────────────┘
```

```bash
# 两种部署分别执行:
python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:30000 \
    --model $MODEL_PATH \
    --dataset-name random \
    --random-input-len 256 \
    --random-output-len 128 \
    --num-prompts 500 \
    --request-rate 20 \
    --max-concurrency 32 \
    --save-result \
    --result-dir ./results/pattern1_short_chat \
    --label "pattern1"
```

**预期对比:**

| 指标 | 3P1D | 3(P+D)+1D | 原因 |
|------|------|-----------|------|
| TTFT | 较高 | **更低** | 混合模式无 KV transfer 延迟 |
| ITL | 相当 | 相当 | Decode 本身速度一致 |
| 吞吐 | 相当 | 相当 | 短请求不会触发溢出 |
| Node4 负载 | 100% | ~0% | 混合模式几乎不溢出 |

---

### Pattern 2: 长文档生成 (内容创作场景)

**特征:** 中等 input + 长 output, 中等并发, 对吞吐和 ITL 敏感

```
┌─────────────────────────────────────┐
│ Pattern 2: Long Generation          │
│                                     │
│ Input:  1024 tokens (文档上下文)     │
│ Output: 2048 tokens (长文生成)       │
│ QPS:    5 req/s                     │
│ 并发:    16                          │
│ 总请求:  100                         │
│ 持续时间: ~200s                      │
└─────────────────────────────────────┘
```

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:30000 \
    --model $MODEL_PATH \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 2048 \
    --num-prompts 100 \
    --request-rate 5 \
    --max-concurrency 16 \
    --save-result \
    --result-dir ./results/pattern2_long_gen \
    --label "pattern2"
```

**预期对比:**

| 指标 | 3P1D | 3(P+D)+1D | 原因 |
|------|------|-----------|------|
| TTFT | 较高 | 稍高(溢出部分) | 混合模式长请求触发 offload |
| ITL | **可能更低** | 相当 | 3P1D 的 D 节点满载时 ITL 恶化 |
| 吞吐 | 受限于单 D 节点 | **更高** | 混合模式 3 节点+1 节点共同 decode |
| Node4 负载 | 100% | ~60-80% | 仅接收溢出的长请求 |

---

### Pattern 3: 突发高并发 (Flash Sale / API 突增)

**特征:** 短时间内大量请求涌入，测试系统弹性

```
┌─────────────────────────────────────┐
│ Pattern 3: Burst Traffic            │
│                                     │
│ Input:  512 tokens                  │
│ Output: 256 tokens                  │
│ QPS:    inf (尽可能快)               │
│ 并发:    128                         │
│ 总请求:  300                         │
│ 特点:    瞬间打满                    │
└─────────────────────────────────────┘
```

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:30000 \
    --model $MODEL_PATH \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 256 \
    --num-prompts 300 \
    --request-rate inf \
    --max-concurrency 128 \
    --save-result \
    --result-dir ./results/pattern3_burst \
    --label "pattern3"
```

**预期对比:**

| 指标 | 3P1D | 3(P+D)+1D | 原因 |
|------|------|-----------|------|
| TTFT P99 | 高 | **更低** | 混合模式 3 节点分摊 decode |
| 吞吐 | 受限 D 瓶颈 | **更高** | 4 个节点都在 decode |
| 失败率 | 可能有排队超时 | 更低 | decode 容量更大 |
| KV Transfer 带宽 | **极高** | 中等 | 仅溢出部分需要 transfer |

---

### Pattern 4: 混合负载 (真实生产场景)

**特征:** 70% 短请求 + 30% 长请求，模拟真实用户分布

```
┌─────────────────────────────────────────────┐
│ Pattern 4: Mixed Workload (Production-like) │
│                                             │
│ Phase 1 (0-30s):   低负载 warmup            │
│   - Input: 256, Output: 128, QPS: 5        │
│                                             │
│ Phase 2 (30-90s):  正常负载                  │
│   - 70%: Input 256, Output 128, QPS: 15    │
│   - 30%: Input 1024, Output 2048, QPS: 3   │
│                                             │
│ Phase 3 (90-120s): 突发高峰                  │
│   - 70%: Input 512, Output 256, QPS: 30    │
│   - 30%: Input 2048, Output 4096, QPS: 5   │
│                                             │
│ Phase 4 (120-150s): 回落                     │
│   - Input: 256, Output: 128, QPS: 5        │
└─────────────────────────────────────────────┘
```

```bash
#!/bin/bash
# === mixed_workload_test.sh ===
# 需要分别对两种部署执行

BASE_URL="http://localhost:30000"
MODEL=$MODEL_PATH
RESULT_DIR="./results/pattern4_mixed"

echo "=== Phase 1: Warmup (30s) ==="
python -m sglang.bench_serving \
    --backend sglang --base-url $BASE_URL --model $MODEL \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts 50 --request-rate 5 --max-concurrency 8 \
    --save-result --result-dir $RESULT_DIR --label "phase1_warmup"

echo "=== Phase 2: Normal Load (60s) ==="
# 短请求 (后台)
python -m sglang.bench_serving \
    --backend sglang --base-url $BASE_URL --model $MODEL \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts 200 --request-rate 15 --max-concurrency 32 \
    --save-result --result-dir $RESULT_DIR --label "phase2_short" &
PID_SHORT=$!

# 长请求 (后台)
python -m sglang.bench_serving \
    --backend sglang --base-url $BASE_URL --model $MODEL \
    --dataset-name random \
    --random-input-len 1024 --random-output-len 2048 \
    --num-prompts 50 --request-rate 3 --max-concurrency 8 \
    --save-result --result-dir $RESULT_DIR --label "phase2_long" &
PID_LONG=$!

wait $PID_SHORT $PID_LONG

echo "=== Phase 3: Peak Load (30s) ==="
# 高频短请求
python -m sglang.bench_serving \
    --backend sglang --base-url $BASE_URL --model $MODEL \
    --dataset-name random \
    --random-input-len 512 --random-output-len 256 \
    --num-prompts 150 --request-rate 30 --max-concurrency 64 \
    --save-result --result-dir $RESULT_DIR --label "phase3_short" &
PID_SHORT=$!

# 超长请求
python -m sglang.bench_serving \
    --backend sglang --base-url $BASE_URL --model $MODEL \
    --dataset-name random \
    --random-input-len 2048 --random-output-len 4096 \
    --num-prompts 30 --request-rate 5 --max-concurrency 16 \
    --save-result --result-dir $RESULT_DIR --label "phase3_long" &
PID_LONG=$!

wait $PID_SHORT $PID_LONG

echo "=== Phase 4: Cooldown (30s) ==="
python -m sglang.bench_serving \
    --backend sglang --base-url $BASE_URL --model $MODEL \
    --dataset-name random \
    --random-input-len 256 --random-output-len 128 \
    --num-prompts 30 --request-rate 5 --max-concurrency 8 \
    --save-result --result-dir $RESULT_DIR --label "phase4_cooldown"

echo "=== Test Complete ==="
```

---

### Pattern 5: 极长上下文 (RAG / 长文档 QA)

**特征:** 超长 input + 短 output, 测试 prefill-heavy 场景

```
┌─────────────────────────────────────┐
│ Pattern 5: Long Context QA          │
│                                     │
│ Input:  8192 tokens (长文档)         │
│ Output: 256 tokens  (简短回答)       │
│ QPS:    2 req/s                     │
│ 并发:    8                           │
│ 总请求:  50                          │
│ 特点:    Prefill 极重               │
└─────────────────────────────────────┘
```

```bash
python -m sglang.bench_serving \
    --backend sglang \
    --base-url http://localhost:30000 \
    --model $MODEL_PATH \
    --dataset-name random \
    --random-input-len 8192 \
    --random-output-len 256 \
    --num-prompts 50 \
    --request-rate 2 \
    --max-concurrency 8 \
    --save-result \
    --result-dir ./results/pattern5_long_ctx \
    --label "pattern5"
```

**预期对比:**

| 指标 | 3P1D | 3(P+D)+1D | 原因 |
|------|------|-----------|------|
| TTFT | **极高** | 高但更好 | 3P1D 需 transfer 大量 KV (8192 tokens) |
| ITL | 正常 | 正常 | output 短，decode 压力不大 |
| KV Transfer | 每请求 ~720KB (MLA) | 仅溢出时 | 混合模式省 70%+ 带宽 |
| Prefill 效率 | 高 (专用) | 稍低 (需共享资源) | 3P1D 的 prefill 无 decode 干扰 |

---

### Pattern 6: 阶梯式压力 (容量规划)

**特征:** QPS 逐步递增，找到系统拐点

```bash
#!/bin/bash
# === staircase_test.sh ===
# 从低到高逐步增加 QPS，找到两种架构的吞吐拐点

BASE_URL="http://localhost:30000"
MODEL=$MODEL_PATH
RESULT_DIR="./results/pattern6_staircase"

for QPS in 2 5 10 15 20 30 50 80; do
    echo "=== Testing QPS=$QPS ==="
    python -m sglang.bench_serving \
        --backend sglang --base-url $BASE_URL --model $MODEL \
        --dataset-name random \
        --random-input-len 512 --random-output-len 512 \
        --num-prompts 100 --request-rate $QPS --max-concurrency 64 \
        --save-result --result-dir $RESULT_DIR --label "qps_${QPS}"
    
    # 等待系统冷却
    sleep 10
done
```

**预期结果曲线:**

```
TTFT (ms)
│
│         3P1D ──────────╱ (D 节点满载后急剧上升)
│                      ╱
│    3(P+D)+1D ──────╱── (更晚到达拐点)
│               ╱──╱
│          ╱──╱
│     ╱──╱
│──╱─╱
└──────────────────────── QPS
   2  5  10  15  20  30  50  80

Throughput (tok/s)
│
│    3(P+D)+1D ──────── ← (更高天花板: 4节点都能decode)
│   ╱───────────────
│  ╱  3P1D ─────── ← (受限于单D节点)
│ ╱──╱──────────
│╱─╱
└──────────────────────── QPS
   2  5  10  15  20  30  50  80
```

---

## 监控与数据收集

### 实时监控脚本

```bash
#!/bin/bash
# === monitor.sh ===
# 后台运行，每 5 秒采集一次指标

LOG_FILE="./results/monitor_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,node,kv_usage,running_reqs,gpu_util" > $LOG_FILE

while true; do
    TS=$(date +%s)
    for NODE in $NODE1 $NODE2 $NODE3 $NODE4; do
        INFO=$(curl -s http://$NODE:8000/get_server_info 2>/dev/null)
        if [ -n "$INFO" ]; then
            KV=$(echo $INFO | jq -r '.kv_cache_usage // 0')
            RUNNING=$(echo $INFO | jq -r '.num_running_reqs // 0')
            GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0")
            echo "$TS,$NODE,$KV,$RUNNING,$GPU" >> $LOG_FILE
        fi
    done
    sleep 5
done
```

### 日志关键词监控

```bash
# Hybrid 节点: 溢出事件
grep -c "Offload triggered" node1.log node2.log node3.log

# 溢出原因分布
grep "Offload triggered" node*.log | grep -c "KV watermark"
grep "Offload triggered" node*.log | grep -c "local decode limit"
grep "Offload triggered" node*.log | grep -c "long output"

# Decode 节点: 接收到的请求数
grep -c "KV received\|transfer complete" node4.log

# KV Transfer 耗时
grep "transfer_time" node*.log | awk -F'=' '{print $NF}' | sort -n | tail -5
```

---

## 结果汇总模板

### 单 Pattern 结果对比表

| 指标 | 3P1D | 3(P+D)+1D | Δ (改善%) |
|------|------|-----------|-----------|
| TTFT Mean (ms) | | | |
| TTFT P99 (ms) | | | |
| ITL Mean (ms) | | | |
| ITL P99 (ms) | | | |
| Output tok/s | | | |
| Total tok/s | | | |
| Failed Requests | | | |
| KV Transfer 次数 | | | |
| Node4 GPU Util (%) | | | |

### 全 Pattern 汇总

```
┌──────────┬──────────────┬──────────────┬─────────┐
│ Pattern  │ 3P1D 胜出    │ Hybrid 胜出   │ 持平    │
├──────────┼──────────────┼──────────────┼─────────┤
│ P1 短对话 │              │ TTFT ✓       │ 吞吐    │
│ P2 长生成 │              │ 吞吐 ✓       │ ITL     │
│ P3 突发   │              │ 吞吐+TTFT ✓  │         │
│ P4 混合   │              │ 综合 ✓       │         │
│ P5 长上下文│ Prefill效率? │ TTFT ✓       │         │
│ P6 阶梯   │              │ 拐点更高 ✓   │         │
└──────────┴──────────────┴──────────────┴─────────┘
```

---

## 执行 Checklist

```
□ 环境准备
  □ 4 台 GPU 节点就绪
  □ Mooncake Transfer Engine 安装验证
  □ SGLang (含 hybrid PD) 代码部署到所有节点
  □ 模型权重已加载到共享存储

□ 配置 A (3P1D) 测试
  □ 启动 3 Prefill + 1 Decode + Router
  □ 验证健康检查通过: curl http://localhost:30000/health
  □ 执行 Pattern 1-6
  □ 收集结果 + 监控数据
  □ 关停所有进程

□ 配置 B (Hybrid) 测试
  □ 启动 1 Decode → 3 Hybrid → Router (注意顺序)
  □ 验证健康检查通过
  □ 执行 Pattern 1-6 (相同参数)
  □ 收集结果 + 监控数据
  □ 关停所有进程

□ 结果分析
  □ 填写每个 Pattern 的对比表
  □ 绘制阶梯测试的 TTFT/吞吐曲线
  □ 统计 Hybrid 模式的溢出比例
  □ 计算 KV Transfer 带宽节省
  □ 生成最终报告
```

---

## 调优参数矩阵

根据测试结果调整 Hybrid 参数:

| 测试发现 | 调整方向 | 参数变化 |
|----------|----------|----------|
| P99 TTFT 仍高于 3P1D | 溢出太多 | watermark 0.80→0.90 |
| 突发时 Node4 满载 | 溢出太频繁 | local_decode_limit 32→48 |
| 长请求影响短请求 ITL | 长请求应早溢出 | long_output_threshold 1024→512 |
| Node4 长期空闲 | 可减少到 0.5 节点 | 考虑移除 Node4 |
| Hybrid 节点 OOM | 本地 decode 太多 | local_decode_limit 降低 |
