# 附录 D：SM89/L40S 生产性能分析与优化建议

> 本文基于当前主线源码（本仓库）与线上配置（SGLang 0.5.17.dev459）分析。个别参数在 0.5.17 分支上可能不同，落地前用
> `python3.12 -m sglang.launch_server --help | grep <参数名>` 复核。

## 1. 问题现象

线上环境：6x L40S（SM89，46GB），TP=2 DP=3，Qwen3.6-35B-A3B-FP8，fp8_e5m2 KV cache，96K 上下文，mamba triton + flashinfer，关闭 CUDA graph，跳过预热。

| 配置 | 现象 |
|------|------|
| A：`--mem-fraction-static 0.78` + `--max-running-requests 8` | 平均 TTFT 30s+，E2E 60s+ |
| B：`--mem-fraction-static 0.85` + `--max-running-requests 12` | 明显更好，但担心 OOM |

**关键观察**：A→B 同时改了两个参数，无法判断各自贡献。以下分析先把两个变量拆开。

## 2. 延迟构成模型

```
TTFT = 排队等待 + prefill 计算（+ 冷启动开销）
E2E  = TTFT + decode 生成
```

三个主要成分各自的决定因素：

| 成分 | 决定因素 |
|------|---------|
| 排队等待 | `--max-running-requests`（per-worker 槽位数）与到达并发的关系 |
| prefill | 上下文长度（96K）/ chunked prefill 合并批大小 / 冷启动内核编译 |
| decode | 批大小、是否 CUDA graph、输出长度 |

## 3. 为什么 0.78/8 慢

### 3.1 `--max-running-requests`（主因）

它是 **per-worker** 的调度槽位数，直接决定每次 prefill/decode 的批大小：

- chunked prefill（4096）会把多个请求的 chunk 合并成一个大 batch，8 个请求凑出来的批比 12 个小；
- L40S 是带宽受限卡，小批量时算力利用率掉得很快；
- DP=3 时 8/worker = 24 总并发，12/worker = 36 总并发。如果到达并发在 24~36 之间，8/worker 必然排队，TTFT 里会包含很长的队列时间。

30 秒级的 TTFT 更像"排队 + 小批量 prefill"，而不是单请求计算本身慢。

### 3.2 `--mem-fraction-static`（次要）

当前源码中 KV 池大小近似为：

```
池子预算 = 加载权重后的剩余显存 − 加载前空闲显存 × (1 − mem_fraction_static)
```

46GB 卡上 0.85 与 0.78 的池子差约 **3GB/卡**（前者留 15% slack，后者留 22%）。池子变小的影响：

- 前缀缓存更容易被逐出 → `sglang:cache_hit_rate` 下降 → 重复前缀被重新 prefill；
- 如果流量有大量共享前缀（agent/工具调用场景），这个影响会被放大；
- 对一次性长 prompt（无前缀复用）几乎没有影响。

所以 0.85 可能"有用"，但大概率不是提速主因。

### 3.3 `--skip-server-warmup`（冷启动成本）

跳过预热意味着：

- mamba triton 内核的 JIT 编译发生在线上第一批请求上（按 shape 逐个编译）；
- flashinfer attention 的 autotune 也发生在第一批请求上（结果缓存于 `~/.cache/flashinfer`）。

如果"平均 TTFT 30s"包含了冷启动段，这一项会显著拉高均值。**这是最容易解释 30 秒级数字的原因之一。**

## 4. OOM 风险评估（0.85/12）

启动时的安全检查只保证"权重 + KV 池"放得下（`pool_configurator.py` 的 `_profile_available_bytes`），**不保证运行时 activation 峰值**：

- 0.85 → 每卡 slack ≈ 15% × 加载前空闲（约 6~7GB），留给 activation/workspace；
- 0.78 → 每卡 slack ≈ 22%（约 9~10GB）。

风险点在于一直 `--skip-server-warmup`，最坏情况的峰值内存从没被真正踩过，属于"没验证过"而不是"肯定不行"。

验证方法：

```bash
# 峰值 soak：36 并发 × 最长 prompt，盯 5 分钟
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 5
# 每卡保持 2~3GB 余量即安全；不稳就退回 0.82
```

另外启动日志会打印 `Memory pool size: ... tokens`，记下来便于对账。

## 5. 优化清单

### 5.1 配置层（低风险，先做）

| 变更 | 预期收益 | 说明 |
|------|---------|------|
| `--max-running-requests 12`（先配 0.78） | 大概率拿回大部分收益 | 解耦验证两个变量；per-worker 值 |
| 去掉 `--skip-server-warmup` 或上线前预热 | 消除冷启动 JIT/autotune 抖动 | 若 warmup OOM，先降 context 或先发几个小请求预热 |
| `--schedule-policy lpm` | 提高前缀缓存命中率 | 默认是 fcfs；配合前缀感知 DP 路由 |
| `--chunked-prefill-size 8192` | 长 prompt 时减少 prefill 步数 | 默认 4096 是本卡档位；8192 需要 activation 余量，配合 0.78~0.80 测 |
| 按实际负载收紧 `--context-length` | 最大单项杠杆 | 若 prompt 实际只有几十 K，64K 立省约 1/3 池子需求 |

### 5.2 稳态微调（中等投入）

| 变更 | 说明 |
|------|------|
| flashinfer autotune 预热后加 `--disable-flashinfer-autotune` | 让 autotune 缓存（~/.cache/flashinfer）先落盘，稳态去掉每 shape 的 autotune 开销 |
| 指标闭环 | 用 `/metrics` 的 queue_time / TTFT / E2E / gen_throughput / cache_hit_rate 判断瓶颈后再动参数（见第 6 节） |

### 5.3 结构性优化（远期）

| 方向 | 预期收益 | 前提 |
|------|---------|------|
| 重开 CUDA graph | decode 段 20~30% | 需驱动升级，解决 capture 失败问题 |
| 投机解码（EAGLE 等） | decode 大收益 | 需验证 Qwen3.6 是否有可用 draft |
| overlap schedule | 提升混合负载利用率 | 当前主线默认开启；用 `--help` 确认 0.5.17 分支 |

### 5.4 并发还能不能加（12→24 的判断方法）

**机制：并发会被自动钳制。** SGLang 启动时会按以下公式计算实际允许的并发（per dp worker）：

```
实际并发 = min(用户填的 max-running-requests,
               KV 池 token 数 ÷ 2,
               mamba 缓存槽位 ÷ 5)
```

Qwen3.6 是混合 SSM 模型，`extra_buffer` + 默认 overlap 调度下，每个并发请求需要约 **5 个 mamba 缓存槽位**（base 3 + extra 2，见 `kv_cache_configurator.py` 的 `_calculate_mamba_ratio`）。如果内存装不下，服务会静默降级并在启动日志打警告：

```
max_running_requests was reduced from the requested 24 to X (per dp worker)
due to the available KV cache capacity
```

**内存账**：24/worker × 最长 96K ≈ 每 worker 需同时容纳约 230 万 token 的 KV+mamba 状态。46GB 卡上池子约 16~19GB，按每 token 8~15KB（含 mamba state）估算只能装 120~240 万 token——24×96K 正处于临界，大概率被钳制或频繁逐出缓存。

**判断方法**：

1. 启动日志 `grep "reduced from the requested"`：出现即被钳制，加大参数无效；
2. `sglang:queue_time_seconds` 为 0：没有排队，加并发没有意义；
3. 阶梯测试：12 → 16 → 20 → 24，固定同一批请求，找 TTFT/E2E/吞吐拐点；过了拐点后每个请求反而变慢（decode 批变大，无 CUDA graph 时每步更慢）。

**什么情况下 24 成立**：实际平均上下文远小于 96K（如 2 万 token，24×2 万=48 万 token 轻松装下）；或把 `--context-length` 收紧到 64K；或显式调大 `--max-mamba-cache-size`（有 OOM 风险，不推荐）。

**提醒**：并发提高会同步推高内存需求，与 0.85 的 OOM 担忧叠加。稳妥路径是先 12 看排队，明显排队再阶梯上调并盯显存余量，而不是一步跳到 24。

## 6. 验证与决策树

### 6.1 对照实验

同一批请求（覆盖线上最长 prompt 与最高并发），各跑几分钟：

```bash
# 配置 1：0.78 / 8（当前慢配置，作基线）
# 配置 2：0.78 / 12（解耦：验证并发是否主因）
# 配置 3：0.85 / 12（当前快配置，对照）
# 配置 4/5/6：0.78 / 16 → 20 → 24（找并发拐点，观察是否被钳制）
```

预期：配置 2 接近配置 3 → 0.85 没必要，直接用 0.78/12；配置 2 明显更慢 → 缓存容量确实敏感，再试 0.82。

### 6.2 指标决策树

```text
queue_time_seconds 占比高  → 加 max-running-requests（批大小/排队）
queue_time_seconds 一直为 0 → 并发已够，加参数无意义
启动日志出现 "reduced from the requested" → 并发被钳制，先收紧 context 或调大池子
cache_hit_rate 低且有重复前缀 → schedule-policy lpm；加大池子；收紧 context
TTFT 里 prefill 段长        → chunked-prefill-size 8192；预热；收紧 context
E2E 里 decode 段长          → CUDA graph；投机解码
```

关键指标（`--enable-metrics` 已开启，Prometheus 文本格式在 `/metrics`）：

| 指标 | 含义 |
|------|------|
| `sglang:queue_time_seconds` | 排队等待时间 |
| `sglang:time_to_first_token_seconds` | TTFT |
| `sglang:e2e_request_latency_seconds` | E2E |
| `sglang:gen_throughput` | 生成吞吐（token/s） |
| `sglang:cache_hit_rate` | 前缀缓存命中率 |
| `sglang:num_queue_reqs` | 当前排队请求数 |

### 6.3 参考目标

在长 prompt（数万 token）场景下，TTFT P50 以 10s 内为合理目标；如果你们实际 prompt 只有几千 token，30s 就说明问题主要在排队或冷启动，而不是 prefill 计算本身。

## 7. 结论

1. **先切 `0.78 + 12`**（配合 lpm、预热），大概率拿回大部分性能且 OOM 风险最低；
2. **不要同时动两个变量**，每次只改一个，用第 6 节指标判断；
3. OOM 用 peak soak + 显存余量 2~3GB 兜底，而不是靠降低并发来"预防"；
4. 真正的结构性收益在 CUDA graph（驱动升级）与投机解码，配置层优化做完后再评估。

### 建议线上配置

```bash
export LD_PRELOAD=/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3.12 -m sglang.launch_server \
    --model-path /usr1/project/models/Qwen3.6-35B-A3B-FP8 \
    --served-model-name Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --tp-size 2 --dp-size 3 \
    --mem-fraction-static 0.78 \
    --max-running-requests 12 \
    --context-length 96256 \
    --chunked-prefill-size 8192 \
    --schedule-policy lpm \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder \
    --kv-cache-dtype fp8_e5m2 \
    --mamba-radix-cache-strategy extra_buffer \
    --mamba-backend triton \
    --enable-flashinfer \
    --attention-backend flashinfer \
    --enforce-disable-flashinfer-allreduce-fusion \
    --disable-cuda-graph \
    --enable-cache-report \
    --enable-metrics \
    --log-level info
```

> 去掉 `--skip-server-warmup` 前先确认 warmup 不再 OOM；若仍 OOM，保留该参数但在放量前用几个代表性请求预热。
