# Qwen3.6-27B-FP8 生产部署数据分析报告

> 最终生产配置: 6卡 DP3 + MTP + 网关控 max_tokens

## 0. 五层防护全景（先读这里）

线上稳定不是某一项优化的功劳，而是**五个独立的问题在五个关卡上被逐个填平**的结果。按请求生命周期看：

```
请求进来
  │
  ▼
① 显存地基 ── mem 0.85 + fp8 KV + mamba extra_buffer
  │            └ 坑：KV 池不够 → OOM/abort（0.78 时 abort 13.2%）
  ▼
② 单请求提速 ── MTP（NEXTN）
  │            └ 坑：decode 92ms/tok → 一条请求 89 秒（E2E 全是 decode）
  ▼
③ 流量阀门 ── 网关 max_tokens + 代理限流 + priority
  │            └ 坑：thinking 长输出占死 KV → 其他请求排队 → TTFT 爆炸
  ▼
④ 路由均衡 ── round_robin
  │            └ 坑：DP Router 把相同前缀请求全堆一个 worker → 4 卡当 2 卡用
  ▼
⑤ 稳定性兜底 ── 预热 + keep-alive + 关 CUDA graph
               └ 坑：冷启动 JIT 抖动、内核被回收、capture 崩溃
```

### 0.1 每层防什么崩（缺了它的后果）

| 层 | 技术 | 不做的后果 | 对应数据 |
|----|------|-----------|---------|
| ① 地基 | mem=0.85 / fp8 / extra_buffer | KV 只有 685K，请求排队 OOM | abort 13.2%、TTFT 45.4s（0.78 版） |
| ② 提速 | MTP | 一条 thinking 请求 decode 86 秒 | E2E 89.2s → 32.1s |
| ③ 阀门 | max_tokens / 限流 / priority | 长输出占满 KV，新请求全部排队 | abort 5.4%、TTFT 14.36s → 1.56%、5.69s |
| ④ 均衡 | round_robin | 4 卡闲置、TTFT 20s+（4+2 拆分的教训） | 负载差 50% → <2% |
| ⑤ 兜底 | 预热/keep-alive | 前几百请求 TTFT 持续恶化 | 冷启动被摊进均值 |

### 0.2 层间依赖（不是平行的，是有先后关系的）

1. **①是②③的前提**：MTP 要占 46K draft 空间、长输出要占 KV，池子不够大后面全白搭。mem=0.85 是"地基"；
2. **③是②的刹车**：MTP 把 decode 提速 3.36x，但高并发时 verify 会和 prefill 抢 GPU——③（控输出、限并发）给"快引擎"装限速器，防止挤爆系统。没有③，②会自己把 TTFT 拉爆（4081 req 时 14.36s）；
3. **②是③的放大器**：同样限 max_tokens=2048，无 MTP 时一条要 17.6s，有 MTP 只要 9.4s——阀门放行的每个请求，引擎都能更快处理完，吞吐才上得去。

### 0.3 一句话总结

①把"房间"做大，②让"每个人办事"变快，③防止"慢的人"占着房间不走，④保证"所有人"都分到房间，⑤防止"刚开门"时服务不稳——五个问题互相独立，但都要填，任何一环断了，其他优化都会被拖下水。1.5 天稳态数据（abort 1.56%、TTFT 5.69s、负载差 <2%）就是五层同时工作的结果。

## 1. 最终生产配置

### 1.1 服务端

| 项目 | 值 |
|------|-----|
| 模型 | Qwen3.6-27B-FP8 |
| GPU | 6×L40S (46GB) |
| 架构 | TP=2 DP=3 |
| Context Length | 98,304 |
| mem-fraction-static | 0.85 |
| KV/card | 746,595 |
| max-running-requests | 12/worker (36 total) |
| Speculative | NEXTN, steps=3, topk=1, draft=4 |
| KV Cache dtype | fp8_e5m2 |

### 1.2 网关侧（API 控制）

| 场景 | enable_thinking | max_tokens | temperature | repetition_penalty |
|------|----------------|------------|-------------|-------------------|
| Thinking (代码检视) | true | 8192 | 0.1 | 1.05 |
| Non-Thinking (tool call) | false | 2048 | 0.0 | 1.0 |

### 1.3 关键优化手段及解决的问题

| 优化手段 | 解决的问题 | 数据证据 | 代价 |
|---------|-----------|---------|------|
| **mem-fraction-static=0.85** | KV cache 容量不足 | 0.78 时 KV=685K, 9GB闲置 → abort 13.2%, TTFT 45.4s; 0.85 时 KV=746K → abort 1.56%, TTFT 5.69s | 剩余显存仅 7.6GB |
| **MTP (NEXTN)** | decode ITL 过高，E2E 被 decode 占满 | 无MTP: ITL=92ms, E2E=89s (decode占95%); 有MTP: 有效ITL=15ms, E2E=32s | verify 挤 prefill, TTFT 从 3.75s 升到 5.69s |
| **LOAD_BALANCE_METHOD=round_robin** | DP Router prefix-aware 热点 | Prefix-Aware: Worker0=0 running, Worker1=12 queued → 4卡当2卡用; Round Robin: 均衡分配 | cache 命中率下降 (prefix 不聚合) |
| **mamba-radix-cache-strategy=extra_buffer** | hybrid Mamba 模型 cache 管理冲突 | Mamba state + Attention KV 争空间; extra_buffer 让 Mamba state 额外预分配，避免挤占 KV | 额外显存开销 (~10%) |
| **网关控 max_tokens** | 长输出占满 KV → evict/abort 暴增 | 无限制: Gen/req=935, abort=5.4%; 限制8192/2048: Gen/req=645, abort=1.56% | 长输出被截断 |

#### 1.3.1 mem-fraction-static — 显存分配

```
问题: SGLang 预分配多少 GPU 显存给 KV cache？
0.78: 46GB × 0.78 = 35.9GB → 9GB 空闲 → KV 只有 685K → 请求排队 OOM
0.85: 46GB × 0.85 = 39.1GB → 7.6GB 空闲 → KV 恢复 746K → 运行正常
      ↑ 多出的 3.2GB 全给了 KV cache
```

#### 1.3.2 MTP — decode 加速

```
问题: 无 MTP 时每步只出 1 token，ITL=92ms，935 tok 输出要 86s
解决: MTP 每步出 3.36 token (accept_len)，有效 ITL=40.9/3.36≈12ms
      935 tok 输出只要 ~11s，decode 时间降 87%
代价: verify forward 占 GPU → prefill 被延迟 → TTFT 升高

MTP 与 prefill 是零和博弈: 同一批 GPU，给了 verify 就没空给 prefill
- 低并发: 两者都够用 → TTFT 1.35s + 有效 ITL 10ms → E2E ~8s
- 高并发: verify 挤占 prefill → TTFT 14s + 有效 ITL 15ms → E2E ~25s
```

#### 1.3.3 LOAD_BALANCE_METHOD — DP 负载均衡

```
问题: Prefix-Aware Router 把相同 system prefix 的请求全路由到一个 worker
      代码审查/tool call 场景下所有请求共享 prefix → Worker 1 堆满，Worker 0 空转

Prefix-Aware:  优势=cache 命中高  劣势=单一 prefix 场景热点严重
Round Robin:   优势=负载均衡      劣势=cache 不聚合，命中率降

生产结论:
  - 1.5天稳态下 prefix-aware 已自然均衡(<2%差异)
    因为生产流量 prefix 多样性够，不会全撞一个 worker
  - 单一 prefix 的压测/特殊场景才会出热点，此时需 round_robin
  - 建议: 默认 prefix-aware，特殊场景手动切 round_robin

> 注: `sglang_start.sh` 默认 `LOAD_BALANCE_METHOD=round_robin`（防热点保险）。
> 若生产流量 prefix 多样、已自然均衡（如上 1.5 天稳态），可考虑显式
> `--load-balance-method auto` 恢复 prefix-aware 以提升 cache 命中。
```

#### 1.3.4 mamba-radix-cache-strategy — Hybrid 模型 cache 协调

```
问题: Qwen3.6-27B 是 hybrid 架构 (Mamba + Attention)
      Mamba 层需要 per-request state cache (146MB/req)
      Attention 层需要 KV cache
      两者争同一块显存

extra_buffer 策略: 为 Mamba state 预留额外 buffer，不挤占 KV pool
                   KV pool 大小不受 Mamba state 影响
                   代价: 多占 ~10% 显存

无此策略: Mamba state 从 KV pool 里扣 → KV 可用空间更少 → 更容易 OOM
         单卡 TP=1 + MTP 时此问题最严重 (剩余 0.65GB 不够 Mamba cache)
```

#### 1.3.5 网关控 max_tokens — 输出长度控制

```
问题: thinking 模式下模型可能输出几千 token，长时间占住 KV 不释放
      其他请求进不来 → queue → TTFT 恶化 → abort

解决: 网关侧限制 thinking=8192, non-thinking=2048
      Gen/req 从 935 降到 645, 请求更快完成, KV 释放更快
      abort 从 5.4% 降到 1.56%

与 MTP 的关系: 网关控 max_tokens 是 MTP 的互补优化
  MTP 解决: 单请求 decode 速度 (ITL 92ms→15ms)
  网关控解决: 请求占 KV 的时间 (Gen/req 935→645)
  两者叠加: E2E 从 89s 降到 32s
```

#### 1.3.6 优化手段依赖关系

```
mem-fraction=0.85  ← 基础，其他优化都依赖 KV 够大
    ├─ MTP         ← 需要 KV 放得下 draft tokens (KV 从 793K→746K)
    ├─ mamba策略   ← 需要 KV 不被 Mamba state 挤占
    └─ max_tokens  ← 需要 KV 周转快，否则长输出占满

LOAD_BALANCE      ← 独立优化，解决 Router 层面问题
```

## 2. Dashboard 历史均值

| 指标 | 值 |
|------|-----|
| **Avg TTFT** | **7.19s** |
| **Avg E2E** | **32.9s** |

> 1.5 天稳态口径：TTFT 5.69s / E2E ~32.1s（见第 6 节）。

## 3. 全方案对比汇总表

### 3.1 单一 6 卡方案演进

| 指标 | 2卡TP2 (65K/0.85) | 6卡0.78 (旧) | 6卡0.85 无MTP | 6卡+MTP 692req | 6卡+MTP 2162req | 6卡+MTP 4081req | 6卡+MTP 混合 | **6卡+MTP 1.5天稳态** |
|------|--------------------|--------------|---------------|----------------|-----------------|-----------------|--------------|----------------------|
| **配置** | | | | | | | | |
| GPU | 2×L40S | 6×L40S | 6×L40S | 6×L40S | 6×L40S | 6×L40S | 6×L40S | **6×L40S** |
| 架构 | TP2 DP1 | TP2 DP3 | TP2 DP3 | TP2 DP3 | TP2 DP3 | TP2 DP3 | TP2 DP3 | **TP2 DP3** |
| context-length | 65,536 | 96,256 | 98,304 | 98,304 | 98,304 | 98,304 | 98,304 | **98,304** |
| mem-fraction-static | 0.85 | 0.78 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | **0.85** |
| MTP | 无 | 无 | 无 | NEXTN s=3 | NEXTN s=3 | NEXTN s=3 | NEXTN s=3 | **NEXTN s=3** |
| KV/card | 792,805 | 685,264 | 793,189 | 746,595 | 746,595 | 746,595 | 746,595 | **746,595** |
| 剩余显存/card | 6.09 GB | 9.15 GB | 6.09 GB | 7.62 GB | 7.62 GB | 7.62 GB | 7.62 GB | **7.62 GB** |
| max-running/worker | 16 | 8 | 12 | 12 | 12 | 12 | 12 | **12** |
| 网关控 max_tokens | 无 | 无 | 无 | 无 | 无 | 无 | 有(8192/2048) | **有(8192/2048)** |
| **请求统计** | | | | | | | | |
| 总请求 | 170 | 204 | 2,294 | 692 | 2,162 | 4,081 | 2,275 | **23,324** |
| Aborted | 2 (1.2%) | 27 (13.2%) | 50 (2.2%) | 2 (0.3%) | 71 (3.3%) | 220 (5.4%) | 30 (1.3%) | **363 (1.56%)** |
| Prompt tokens | 1.5M | - | 28.8M | 4.8M | 23.5M | 41.7M | 22.5M | **193.5M** |
| Gen tokens | 114K | 69K | 2.1M | 322K | 1.4M | 2.8M | 1.1M | **15.0M** |
| Gen/req | 672 | 339 | 935 | 465 | 648 | 674 | 501 | **645** |
| **延迟** | | | | | | | | |
| ITL avg | 94.3ms | 133.5ms | 92.2ms | 34.8ms | 50.6ms | 52.2ms | 50.5ms | **40.9ms** |
| 有效 ITL | 94.3ms | 133.5ms | 92.2ms | ~9.7ms | ~15.1ms | ~15.1ms | ~15.0ms | **~15.0ms** |
| TTFT stream avg | 3.87s | 45.4s | 3.75s | 1.35s | 8.76s | 14.36s | 6.18s | **5.69s** |
| TTFT non-stream avg | 10.12s | - | 5.94s | 8.92s | 16.96s | 26.48s | 15.43s | **13.20s** |
| Dashboard TTFT | - | - | 4.37s | - | - | - | 7.19s | **5.69s*** |
| Dashboard E2E | - | - | 89.2s | - | - | - | 32.9s | **~32.1s*** |
| **MTP** | | | | | | | | |
| Accept Rate | - | - | - | 86.1% | 78.8% | 82.1% | 74.7-81.0% | **27-47%**** |
| Accept Len | - | - | - | 3.58 | 3.36 | 3.46 | 3.24-3.43 | **1.8-2.4**** |
| Verify calls | - | - | - | 92,635 | 409,269 | 802,803 | 331,915 | **4,387,029** |
| **Cache** | | | | | | | | |
| Cache hit | 55.0% | 29.2% | 43.2% | 59.9% | 37.6% | 31.9% | 36.3% | **45.8%** |
| Evicted | 119K | 1.2M | 30.3M | 599K | 23.4M | 47.3M | 22.3M | **195.2M** |
| Evict/Prompt | 0.08 | - | 1.05 | 0.12 | 0.99 | 1.13 | 0.99 | **1.01** |
| kv_available | 3,126 | 822-3,962 | 3-4K | 97K-178K | 1.2K-3.9K | 185-4,557 | 0-4,136 | **377-6,997** |
| **吞吐** | | | | | | | | |
| Gen throughput | 159 tok/s | 187 tok/s | 112 tok/s | 335 tok/s | 723 tok/s | 497 tok/s | 403 tok/s | **-*** |
| **运行时 (采样)** | | | | | | | | |
| Running | 13 | 20 | 0 | 0-5 | 5-11 | 6-11 | 10-11 | **0** |
| Queue | 0 | 29 | 0 | 0 | 0-2 | 0-3 | 0-10 | **0** |
| 负载均衡 | N/A | Worker0=7.5tok/s | 3 worker差<3% | 均衡 | Worker1热点 | Worker0热点 | Worker1热点 | **均衡(<2%)** |

> *E2E 按稳态数据计算: TTFT 5.69s + 645×40.9ms = 32.1s。** Accept Rate/Accept Len 为空闲采样瞬时值，batch=1 时波动大，不代表累计效果。

### 3.2 双架构方案对比

| 指标 | 4+2 Prefix-Aware | 4+2 Round Robin | **6卡统一+MTP(最终)** |
|------|-----------------|----------------|----------------------|
| **Tool Call 实例** | | | |
| GPU | 4卡 TP2 DP2 | 4卡 TP2 DP2 | **6卡 TP2 DP3** |
| 总请求 | 278 | 952 | **2275 (混合)** |
| TTFT stream | 9.95s | 19.91s | **6.18s** |
| ITL avg | 69.7ms | 59.9ms | **50.5ms** |
| Aborted | 1 (0.4%) | 57 (6.0%) | **30 (1.3%)** |
| Cache hit | 30.8% | 39.6% | **36.3%** |
| Worker 负载 | Worker0空转 | 均衡但排队 | 有热点 |
| **Thinking 实例** | | | |
| GPU | 2卡 TP2 DP1 | 2卡 TP2 DP1 | **(同上统一处理)** |
| 总请求 | 69 | 172 | - |
| TTFT stream | 2.17s | 2.37s | - |
| Accept Rate | 60.8% | 86.8% | **74.7-81.0%** |
| **整体评价** | 4卡当2卡用 | TTFT 20s + 6% abort | **TTFT 7.19s, E2E 32.9s** |

## 4. MTP 效果分析

### 4.1 Accept Rate / Accept Len 全轮次演进

| 轮次 | 请求数 | Accept Rate | Accept Len | 有效 ITL | TTFT stream | Gen/req | Aborted |
|------|--------|------------|------------|---------|-------------|---------|---------|
| 低并发 | 692 | 86.1% | 3.58 | ~9.7ms | 1.35s | 465 | 0.3% |
| 中并发 | 2,162 | 78.8% | 3.36 | ~15.1ms | 8.76s | 648 | 3.3% |
| 高并发 | 4,081 | 82.1% | 3.46 | ~15.1ms | 14.36s | 674 | 5.4% |
| **混合+网关控** | **2,275** | **78.6%** | **3.36** | **~15.0ms** | **6.18s** | **501** | **1.3%** |

> 网关控 max_tokens 后：Gen/req 从 935 降至 501，Abort 从 5.4% 降至 1.3%，TTFT 从 14.36s 降至 6.18s。有效 ITL 稳定在 ~15ms，MTP decode 加速 3.36x 持续有效。

### 4.2 瓶颈迁移

```
无MTP:  decode 瓶颈 (95% E2E)     → TTFT 3.75s, E2E 89.2s
+MTP 低并发: decode 加速, prefill 快 → TTFT 1.35s, E2E 7.8s
+MTP 高并发: prefill 成为瓶颈       → TTFT 14.36s, E2E 24.6s
+MTP +网关控: 输出缩短, KV 释放快   → TTFT 6.18s, E2E 32.9s

E2E 构成:
┌────────────────────────────────────────────────────────────┐
│          │  无MTP    │  +MTP高并发  │  +MTP+网关控(最终)   │
├──────────┼───────────┼─────────────┼──────────────────────┤
│ TTFT     │  4.37s 5% │ 14.36s 58%  │  7.19s 22%           │
│ Decode   │ 84.83s 95%│ 10.24s 42%  │ 25.71s 78% ← 仍主因  │
│ Queue    │  ≈0s      │  ≈0s        │  ≈0s                 │
└──────────┴───────────┴─────────────┴──────────────────────┘
```

### 4.3 E2E 场景估算

| 场景 | 无 MTP | **有 MTP + 网关控** | 降幅 |
|------|--------|-------------------|------|
| Thinking (~700 tok) | 3.75s + 700×92ms = **68s** | 7.19s + 700×15ms = **17.7s** | **-74%** |
| Non-Thinking (~150 tok) | 3.75s + 150×92ms = **17.6s** | 7.19s + 150×15ms = **9.4s** | **-47%** |
| Dashboard 实测 | **89.2s** | **32.9s** | **-63%** |

## 5. RadixTree Cache 表现

| 版本 | Cache 命中率 | Evicted | Evict/Prompt | KV 紧张度 |
|------|-------------|---------|-------------|----------|
| 2卡单实例 (无MTP) | 55.0% | 119K | 0.08 | 低 |
| 6卡0.85 (无MTP) | 43.2% | 30.3M | 1.05 | 高 |
| 6卡+MTP 低并发 (692req) | 59.9% | 599K | 0.12 | 低 |
| 6卡+MTP 中并发 (2162req) | 37.6% | 23.4M | 0.99 | 高 |
| 6卡+MTP 高并发 (4081req) | 31.9% | 47.3M | 1.13 | 极高 |
| **6卡+MTP 混合+网关控** | **36.3%** | **22.3M** | **0.99** | **高** |
| **6卡+MTP 1.5天稳态** | **45.8%** | **195.2M** | **1.01** | **可接受** |

Per-Worker Cache (最终混合数据):

| Worker | prefill_compute | prefill_cache | 命中率 | kv_available |
|--------|----------------|---------------|--------|-------------|
| 0 | 4,814,254 | 2,649,669 | 35.5% | 4,136 |
| 1 | 4,958,706 | 2,652,773 | 34.8% | 2,581 |
| 2 | 4,689,181 | 2,998,524 | 39.0% | **0** |

Per-Worker Cache (1.5天稳态):

| Worker | prefill_compute | prefill_cache | decode | 命中率 | kv_available |
|--------|----------------|---------------|--------|--------|-------------|
| 0 | 35.3M | 29.7M | 4.94M | 45.7% | 3,643 |
| 1 | 34.6M | 28.4M | 5.11M | 45.1% | 6,997 |
| 2 | 34.9M | 30.6M | 5.08M | 46.7% | 377 |

> MTP 与 RadixTree 互增强: 低并发时 MTP 加速 KV 释放 → evict 少 → 命中率 59.9%。高并发时 KV 容量硬约束，网关控 max_tokens 缓解了压力。1.5 天稳态下 RadixTree 积累更多 prefix，命中率从 36.3% 提升到 45.8%。

## 6. 1.5 天稳态分析

> 数据来源: 2026_08_05_metrics.log, 运行 ~1.5 天, 23,324 请求

### 6.1 稳态 vs 短期对比

| 指标 | 混合场景 (2,275req) | **1.5天稳态 (23,324req)** | 变化 |
|------|-------------------|------------------------|------|
| 总请求 | 2,275 | **23,324** | 10x |
| Aborted | 30 (1.3%) | **363 (1.56%)** | 稳定 |
| Gen/req | 501 | **645** | +29% |
| TTFT stream | 6.18s | **5.69s** | **-8%** |
| TTFT non-stream | 15.43s | **13.20s** | **-14%** |
| ITL avg | 50.5ms | **40.9ms** | **-19%** |
| Cache hit | 36.3% | **45.8%** | **+10pp** |
| Evict/Prompt | 0.99 | **1.01** | 稳定 |
| 负载均衡 | Worker1热点 | **3 worker差<2%** | **改善** |

### 6.2 稳态改善原因

**1. TTFT 改善 (6.18s → 5.69s)**

短期采样时恰逢高并发，1.5 天均值覆盖了高低峰，平均更低。

**2. ITL 改善 (50.5ms → 40.9ms)**

短采样时高并发 batch 大，ITL 偏高。1.5 天含大量低并发时段，均值更优。

**3. Cache 命中率提升 (36.3% → 45.8%)**

长时间运行后 RadixTree 积累更多 prefix 缓存，命中率自然提升 10pp。

**4. 负载均衡改善**

1.5 天累计数据中 3 个 worker 请求差 <2%，短期采样的热点在长周期下被摊平。

### 6.3 Per-Worker 均衡度 (1.5天)

| Worker | prefill_compute | prefill_cache | decode | 命中率 | kv_available |
|--------|----------------|---------------|--------|--------|-------------|
| 0 | 35.3M | 29.7M | 4.94M | 45.7% | 3,643 |
| 1 | 34.6M | 28.4M | 5.11M | 45.1% | 6,997 |
| 2 | 34.9M | 30.6M | 5.08M | 46.7% | 377 |

```
Worker 请求分布差: (35.3-34.6)/34.9 = 2% ← 非常均衡
Worker 2 的 kv_available=377 仍偏低，但采样时 0 running，不影响
```

### 6.4 运行健康度

| 维度 | 评价 | 数据 |
|------|------|------|
| 稳定性 | **优秀** | 1.56% abort，0 queue，0 running 空闲采样 |
| TTFT | **良好** | 5.69s (stream)，比短期 6.18s 更好 |
| ITL | **良好** | 40.9ms，比短期 50.5ms 更好 |
| Cache | **良好** | 45.8%，比短期 36.3% 更好 |
| 负载均衡 | **优秀** | 3 worker 差异 <2% |
| Evict | **可接受** | 195M/1.5天，evict/prompt=1.01 |

### 6.5 E2E 验证

```
1.5天稳态 E2E 计算:
  TTFT 5.69s + 645 × 40.9ms = 5.69 + 26.4 = 32.1s
  Dashboard 实测: 32.9s ← 吻合

按场景拆解:
  Thinking (~700 tok): 5.69s + 700×40.9ms = 5.69 + 28.6 = 34.3s
  Non-Thinking (~150 tok): 5.69s + 150×40.9ms = 5.69 + 6.1 = 11.8s
```

## 7. mem-fraction-static 对比（反面教材）

| mem-fraction | KV/card | 剩余显存 | Aborted | Evicted | TTFT stream | Cache hit |
|-------------|---------|---------|---------|---------|-------------|-----------|
| **0.78** | 685,264 | **9.15 GB** | 27 (13.2%) | 1.2M | 45.4s | 29.2% |
| **0.85** | 793,189/746,595 | 6.09/7.62 GB | 50 (2.2%) | 30.3M | 3.75s | 43.2% |

> 教训: mem-fraction-static=0.78 对 96K context 太保守，9GB 显存闲置导致 KV 不足，级联恶化。0.85 是正确选择。

## 8. 最终结论

### 8.1 生产方案选定

**6卡 DP3 + MTP + 网关控 max_tokens** 为当前最优方案:

```bash
# 服务端（脚本默认已内置 mem=0.85 / max-running=12 / MTP / round_robin / 预热 / keep-alive）
bash sglang_start.sh \
    --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --context-length 98304 \
    --mem-fraction-static 0.85 \
    --max-running-requests 12 \
    --enable-speculative

# 网关侧
# Thinking:  enable_thinking=true,  max_tokens=8192, temperature=0.1
# Tool Call: enable_thinking=false, max_tokens=2048, temperature=0.0
```

### 8.2 关键收益

| 指标 | 无 MTP 基线 | 短期混合 | **1.5天稳态** | 改善(vs基线) |
|------|-----------|---------|-------------|-------------|
| Dashboard E2E | 89.2s | 32.9s | **~32.1s** | **-64%** |
| Dashboard TTFT | 4.37s | 7.19s | **5.69s** | +30% (MTP代价) |
| ITL avg | 92.2ms | 50.5ms | **40.9ms** | **-56%** |
| 有效 ITL | 92.2ms | ~15.0ms | **~15.0ms** | **-84%** |
| Gen/req | 935 | 501 | **645** | -31% (网关控) |
| Aborted | 2.2% | 1.3% | **1.56%** | -29% |
| Cache hit | 43.2% | 36.3% | **45.8%** | +6% |

### 8.3 各优化手段贡献

| 优化手段 | 解决的问题 | 代价 |
|---------|-----------|------|
| **mem-fraction 0.85** | KV 不足 → abort/evict 暴增 | 剩余显存仅 6-7.6GB，不能再高 |
| **MTP (steps=3)** | decode ITL 92ms→15ms，E2E 降 63% | verify 挤 prefill，TTFT 从 3.75s 升到 7.19s |
| **网关控 max_tokens** | Gen/req 935→501，Abort 5.4%→1.3% | 长输出被截断，需业务确认可接受 |
| **6卡统一 (非 4+2 拆分)** | 避免双架构 Router 热点/空转 | 无法按场景独立优化 |

### 8.4 已知限制与风险

| 限制 | 影响 | 缓解措施 |
|------|------|---------|
| TTFT 5.69s (vs 无MTP 3.75s) | 首 token 延迟增加 | MTP 代价，可接受；业务侧用 stream 模式 |
| Worker 热点 | 短期采样时 Worker1 有 queue | 1.5天稳态下 <2% 差异，已被摊平 |
| Worker 2 kv_available=377 | 采样时空闲时仍偏低 | 网关控 max_tokens 已缓解，持续观察 |
| Cache hit 45.8% | DP cache 不共享 + 高 evict | 比 6卡无MTP 43.2% 略好，DP 架构固有代价 |
| MTP + 高并发 = TTFT 恶化 | 并发 >12 时 TTFT 急剧上升 | 网关侧限并发，横向扩 pod 而非调高 max-running |

### 8.5 后续优化方向

| 方向 | 预期收益 | 复杂度 | 优先级 |
|------|---------|-------|-------|
| PD 分离 (Prefill/Decode) | prefill 不受 verify 干扰，TTFT 回到 ~3s | 高 | 长期 |
| 降低 speculative_num_steps 3→2 | verify 开销减 33%，TTFT 降 | 低 | 可试 |
| 多 pod 横向扩容 | 总吞吐线性增长 | 中 | 按需 |
| SGLang Router 权重平衡 | 缓解 Worker 热点 | 中 | 中期 |
| 更大 KV (更大显存 GPU) | cache hit 提升，evict 减少 | 高 | 换硬件时 |

> 注：PD 分离在**当前 6 卡 L40S 上收益有限**（权重双份加载、并发下降、PCIe KV 传输），
> 详见附录 D 11.12；"长期"是指扩容到 12+ 卡后再评估。
