# 附录 D：SM89/L40S 生产性能分析与优化建议

> 本文基于当前主线源码（本仓库）与线上配置（SGLang 0.5.17）分析。个别参数在 0.5.17 分支上可能不同，落地前用
> `python3.12 -m sglang.launch_server --help | grep <参数名>` 复核。

## 0. 快速导航（先读这里）

**当前生产配置（2026-08 定版）**

```bash
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --port 8000 --proxy-port 8080
```

6 卡 TP2 DP3 + MTP（NEXTN steps=3 draft=4）+ 代理（tool_call=16 / thinking=12）+ priority + round_robin + mem=0.85 / context=98304 / max-running=12/worker（均为脚本默认值）。

**当前生产基线（2026-08 线上）**

| 指标 | 值 | 评估 |
|------|-----|------|
| 单日总请求 | 16,454 | 稳定 |
| Aborted | 310（1.88%） | 健康（<2%，各版本最低） |
| AVG TTFT | 8.37s | MTP 高并发正常水位 |
| AVG E2E | 33.6s | decode ~25.2s（长输出占比上升） |

**后续观察点速查（详见 11.11）**

| 指标 | 阈值 | 行动 |
|------|------|------|
| abort 率 | >2% 且持续 | 降 tool_call / 查 KV 紧张 |
| token_usage | ≥0.92 持续 5m | 降并发或 mem 调回 0.85 |
| TTFT p90 | >10s 持续 | 查排队占比，调 priority/限流 |
| queue | >10 持续 | 容量瓶颈，考虑扩副本 |

**文档结构导览**

| 章节 | 性质 | 说明 |
|------|------|------|
| 1~8 | 过程分析（已定稿） | 早期 0.78/8 慢的归因与验证，结论已固化进脚本默认值 |
| 9 | 请求参数建议 | 代码检视 / tool call 的 max_tokens 与 sampling 配置 |
| 10 | MTP 落地记录 | 可行性→试点→上线→冷启动→全版对比（10.9 两实例方案**已废弃**，见 11.7） |
| 11.1~11.8 | 双架构教训与定版 | 路由热点 → 回退 6 卡统一 + round_robin |
| 11.9 | 当前机制 | 优先调度 + 代理限流 + 自适应 + 监控（附录 G 详述监控平台） |
| 11.10 | 扩展分析 | 96K → 256K 可行性 |
| 11.11 | **后续观察点** | 监控阈值与行动矩阵 |
| 11.12 | 扩展分析 | 6 卡 PD 分离可行性 |

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

> 生产数据验证已补充到第 8 节：线上最终采用 0.85/12/98304，实测稳定（见下）。

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

## 8. 生产数据验证（Qwen3.6-27B-FP8，6 卡 DP=3）

> 数据源：sglang_data.log；线上配置 `context-length=98304`、`mem-fraction-static=0.85`、`max-running-requests=12/worker`、`chunked-prefill-size=4096`。

### 8.1 关键指标

| 指标 | 值 | 判断 |
|------|-----|------|
| TTFT (stream) | 3.75s（全量加权 4.37s） | 达标（<10s 目标） |
| ITL | 92.2ms/tok（聚合约 390 tok/s） | 接近该卡/模型组合的可达区间 |
| E2E | 89.2s ≈ 935 tok/req × 92ms | 输出长度固有开销，非服务问题 |
| Cache 命中率 | 43.2%（3 worker：43.0 / 45.4 / 40.9%） | 良好 |
| 队列 | 0 | 并发充足 |
| Abort | 2.2%（50/2294） | 可接受（需确认原因分布） |
| 负载均衡 | 3 worker 请求差 <3% | 很好 |
| 剩余显存 | 6.09 GB/卡 | 0.85 经 2294 请求实测稳定 |

### 8.2 对前面结论的验证

1. **0.85/12 已被实测验证**：2294 请求、0 排队、TTFT 3.75s、Abort 2.2%、每卡 6.09GB 余量——第 4 节"0.85 没验证过"的担心可以划掉。
2. **旧 0.78/8 的 45.4s TTFT + queue=29 + Abort 13.2%** 印证第 3.1 节判断：当时慢的主因是并发/排队，而非缓存容量。
3. **E2E 大头是输出长度**：89.2s 中 decode 占约 95%（935 tok × 92ms）。配置层调参（chunk、mem-fraction、并发）几乎不影响 E2E；真正杠杆在业务侧（`max_tokens` / `thinking_budget` / 关 thinking）与投机解码。
4. **ITL 的可优化空间取决于模型 active 参数规模**：92ms 对应聚合约 390 tok/s。若 27B 是 A3B 结构，理论带宽上限还有余量，值得排查 decode 侧额外开销（mamba triton、KV 读取、无 CUDA graph）；若接近 active 权重带宽上限，则只能靠投机解码或业务侧砍输出。
5. **"Evict 105%" 是误导性口径**：淘汰次数与 prompt tokens 对比无意义。池子总共约 2.38M token（793K×3），30.3M 次淘汰 ≈ 13 次全量周转，是"池子小 + 流量大"的固有现象。**43.2% 命中率才是有效指标**；DP=3 相比单实例 55% 的下降是缓存分片的固有代价，Prefix-Aware Router 已在缓解。
6. **KV 池满载（99.7% evictable）是健康状态**：缓存装满才可能命中，重点看命中率而不是"满不满"。

### 8.3 E2E 优化杠杆（按收益排序）

| 方向 | 预期收益 | 说明 |
|------|---------|------|
| 业务侧：`max_tokens` / `thinking_budget` / 关 thinking | E2E 随输出 token 线性下降 | 935 tok/req 是 E2E 89s 的根本原因 |
| 投机解码（EAGLE 等） | decode 大收益 | 唯一能显著压有效 decode 步数的系统侧手段；需验证 Qwen3.6 支持 |
| CUDA graph（驱动升级后） | ITL 10~20% | 减每步固定开销，不改变带宽上限 |

### 8.4 后续补充数据

- 会话总时长与聚合吞吐（tok/s）——评估容量的头条指标；
- Abort 原因分布（客户端断开 vs 超时 vs 错误）；
- stream=true/false 各自的 prompt 长度分布（解释 TTFT 3.75s vs 5.94s 的差异）。

## 9. 场景化请求参数建议（代码场景）

### 9.1 硬约束回顾

ITL 92ms 下，每 100 个输出 token ≈ 9.2s。**E2E<10s 只对短输出任务成立**；代码检视这类长输出任务要先调整预期，而不是压服务器配置。

### 9.2 tool call 提取上下文（目标 TTFT<1s / E2E<10s）

推荐配置（new API 侧）：

```json
{
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 0.0,
    "repetition_penalty": 1.0,
    "max_completion_tokens": 512,
    "max_tokens": 512
}
```

| 参数 | 建议值 | 理由 |
|------|--------|------|
| `enable_thinking` | `false` | 提取是确定性任务，thinking 是纯开销；模型模板不支持 thinking_budget，无法软限制 |
| `max_tokens` | 512（严格 <10s 目标用 256） | 工具调用 JSON 典型 50~160 token；512 覆盖长参数工具，256 保 E2E |
| `temperature` | 0.0 | 确定性 |
| `repetition_penalty` | 1.0 | |

> 注：Qwen3.6-27B 模板不支持 `thinking_budget`（实测无此字段，静默忽略），thinking 长度只能靠 `max_tokens` 硬截断（见 9.3）。

### 9.3 代码检视（质量-延迟权衡）

推荐配置（new API 侧，默认档）：

```json
{
    "chat_template_kwargs": {
        "enable_thinking": true
    },
    "temperature": 0.1,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
    "max_completion_tokens": 4096,
    "max_tokens": 4096
}
```

| 档位 | max_tokens | 单请求 decode | 适用 |
|------|-----------|--------------|------|
| **默认档（保答案）** | **4096** | ~82s+ | 常规代码检视，答案完整 |
| 折中档 | 3072 | ~66s+ | thinking 长时可能截断答案，需抽检 |
| 快速档（不建议） | 2048 | ~50s | **已验证：thinking 吃满后答案被截断** |

- **Qwen3.6-27B 模板不支持 `thinking_budget` / `budget_tokens`**（实测 tokenizer_config 中无此字段，设置会被静默忽略）——控制输出长度的唯一手段是 `max_tokens` 硬截断；
- 硬截断的代价：`max_tokens` 过低会截在 thinking 中途、答案丢失（2048 已实测复现）；没有"限思考预算"的软开关；
- 想真正砍 thinking 延迟，只有两条路：**关 thinking**（`enable_thinking: false` + 结构化 prompt 直接给审查结论，无 thinking token，E2E 减半以上，需质量 A/B）或**加容量/换 H20-H100**；
- 质量-延迟是同一杠杆两端，用"max_tokens vs bug 检出率"的 A/B 曲线决定档位，不要凭感觉。

### 9.4 通用配置（压低 TTFT）

- `--schedule-policy lpm`；
- system prompt / tool schema / 仓库上下文**固定且放请求开头**，变化内容（diff、当前文件）放最后，提高前缀命中率；
- 放量前预热（去掉 `--skip-server-warmup` 或先发几个代表性请求）；
- `context-length` 按实际最大输入收紧，不要一直挂 98304。

## 10. 投机解码（MTP/NEXTN）可行性核查

> 结论先行：MTP 是解 Decode 显存带宽瓶颈的正确方向，源码确认 Qwen3Next（Qwen3.6 系）存在原生 MTP 路径，但**对本部署不是开箱即用**，落地前必须通过 10.3 的四项核查。

### 10.1 源码核对结果

- `NEXTN` 是 SGLang 内置投机算法（`--speculative-algorithm`），内部解析为 EAGLE 系；`--speculative-eagle-topk 1` 即 NEXTN/MTP 模式（走拒绝采样）。
- [model_config.py](/home/atituiset/Projects/sglang/python/sglang/srt/configs/model_config.py:654) 中 `Qwen3NextForCausalLM` 在 draft 模式下切换为 `Qwen3NextForCausalLMMTP`，draft 仅 **1 个 hidden layer**，权重从同一 checkpoint 加载。
- 参数名纠错：正确写法是 `--speculative-algorithm NEXTN`，**不是** `--speculative-algo`；`--speculative-num-steps`、`--speculative-eagle-topk`、`--speculative-num-draft-tokens` 均存在。
- **投机 worker 全部基于 CUDA graph runner**（`eagle_draft_cuda_graph_runner`、`init_cuda_graphs`）——与必须的 `--disable-cuda-graph` 存在冲突风险，是最大未知数。

### 10.2 落地前四项核查（按重要性排）

| # | 核查项 | 状态 |
|---|--------|------|
| 1 | 部署版本（0.5.17）是否包含 Qwen3Next MTP 路径（以下核对基于当前主线源码，比线上版本新） | 待查 |
| 2 | FP8 checkpoint 是否保留 MTP 权重（量化转档通常丢弃，是最可能的拦路虎） | ✅ 已确认：模型目录存在 `mtp.safetensors`；张量命名待启动加载验证 |
| 3 | 关闭 CUDA graph 时投机解码能否运行（可能启动报错，或退化为 eager 低效模式） | 待测 |
| 4 | hybrid SSM（mamba）+ MTP + DP=3 组合兼容性（代码中有 `mamba_track_interval >= speculative_num_draft_tokens` 断言，路径有人维护，仍需实测） | 待测 |

核查命令：

```bash
# 1) 版本支持
python3.12 -m sglang.launch_server --help | grep -E "speculative-algorithm|speculative-num"
python3.12 -c "import sglang.srt.models.qwen3_next_mtp; print('MTP model class OK')"

# 2) MTP 权重与张量命名
ls /usr1/project/models/Qwen3.6-27B-FP8/ | grep -i mtp
python3.12 -c "
from safetensors import safe_open
f = safe_open('/usr1/project/models/Qwen3.6-27B-FP8/mtp.safetensors', framework='pt')
ks = list(f.keys())
print(len(ks), 'tensors'); print('\n'.join(ks[:3]))
"  # 预期 key 带 model.mtp.* 前缀
```

> 试点结果（ITL / TTFT / 输出一致性）验证后回填本节状态。

### 10.3 对常见说法（如 AI 生成的建议）的修正

| 说法 | 修正 |
|------|------|
| 提速 1.4~2.2x，ITL 45~60ms | 合理区间但别信上限；L40S + FP8 + 无 CUDA graph + 混合 SSM 按 1.3~1.8x 预期 |
| mem-fraction 从 0.85 降到 0.78 | 错误建议：draft 仅 1 层，显存开销几百 MB~1GB，不需要降 0.07；且 KV 池已 99.7% 满载，降了只会让命中率更低 |
| 并发降到 6~8/worker 给 MTP 留算力 | 无依据：先保持 12/worker 直接测，draft 1 层开销小 |
| 准确率无损 | 拒绝采样理论上保持分布，但 topk=1 是贪心验证路径，实际输出有差异，需抽检 |

### 10.4 试点方案（先小流量 A/B，不上生产）

```bash
# 检查项见 10.2；试点参数：
--speculative-algorithm NEXTN \
--speculative-num-steps 3 \
--speculative-eagle-topk 1 \
--speculative-num-draft-tokens 4
```

对比同一批请求的 ITL、TTFT 与输出一致性。若 checkpoint 无 MTP 权重（10.2-2 为空），则需重新准备带 MTP 的 FP8 checkpoint，并确认量化流程保留 MTP 层——这是投入最大的前置条件。

### 10.5 预期与提醒

- 即使 MTP 全通、ITL 从 92ms 压到 50~60ms，935 token 输出的 E2E 仍在 50 秒级——MTP 是系统侧最大优化，但 E2E<10s 的物理前提依然是限制输出长度（见第 9 节），两者要一起做。

### 10.6 上线实测与 TTFT 定位（MTP）

线上约 2000 请求实测（MTP 开启，6 卡 DP=3）：

| 指标 | 无 MTP（基线） | 有 MTP | 变化 |
|------|--------------|--------|------|
| E2E | 89.2s | 41.1s | **2.17x 提升** |
| TTFT | 3.75s | 10.63s | 变差 2.8x |
| 等效 ITL | 92ms | ≈33ms（按 935 tok/req 折算） | **2.8x 加速** |

结论：MTP 在 decode 侧效果显著（E2E 减半、等效 ITL ~33ms，优于 10.5 预估的 45~60ms），问题集中在 TTFT。

**TTFT 变差的三个候选原因（按可能性排序）**

1. **冷启动被摊进均值**：MTP 新增 draft extend / verify 内核 + `--skip-server-warmup` + prompt 长度多变 → triton 按 shape 逐个 JIT、flashinfer 按 shape 逐个 autotune，可能持续到前几百个请求。判据：TTFT 是否随时间下降。
2. **排队**：E2E 变短后同一批压测的到达节奏/并发变化，队列时间被算进 TTFT。看 `queue_time_seconds` 占比。
3. **prefill 结构性开销**：draft extend 若在 eager 模式下未吃上 chunked batching，长 prompt（平均 12.5K token）的 prefill 明显变慢；或 draft extend 破坏前缀缓存命中（`cache_hit_rate` 下降 → 重复 prefill）。

**定位命令**

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E "queue_time_seconds|cache_hit_rate|time_to_first_token_seconds|prefill"
```

| 现象 | 判定 | 修法 |
|------|------|------|
| `queue_time_seconds` 占 TTFT 大头 | 排队问题，非 MTP 的锅 | 调并发/到达节奏 |
| `cache_hit_rate` 明显低于基线 43.2% | 前缀命中被破坏 | 查 draft extend 是否绕过 radix cache |
| TTFT 高桶占比随时间下降 | 冷启动 | 预热 + keep-alive + `--disable-flashinfer-autotune` |
| prefill 段变长且稳定 | draft extend 结构性开销 | 版本实现问题，短期接受 tradeoff 或等新版优化 |

**决策点**：TTFT 影响 tool call 场景（短交互、要快），E2E 影响代码检视（长输出、能等）。MTP 当前 E2E 大胜、TTFT 小败；若 tool call 的 TTFT<1s 是硬指标，先区分冷启动 vs 结构性问题，再决定是否保留一条非 MTP 路径。

### 10.7 冷启动解决方案详解（预热）

> 适用前提：10.6 定位确认 TTFT 高桶占比随时间下降（冷启动）。MTP 比非 MTP 多了 draft/verify 内核，冷启动更贵，预热更重要。

**方案一：上线前预热（推荐）**

做法 A——去掉 `--skip-server-warmup`（若不再 OOM）：

- 优点：SGLang 启动阶段统一编译/预热，无需自建脚本；
- 前提：确认 warmup 不再 OOM。此前 96K 配置下 OOM 过，MTP 后显存占用更高，先小规模验证；若仍 OOM 用做法 B。

做法 B——预热脚本（可控，推荐）：

1. **等服务 ready**：轮询 `/v1/models` 直到返回 200（超时 600s）；
2. **发送覆盖实际负载形状的请求**：
   - 2~3 个短 prompt 非 thinking（tool call 形状，`max_tokens 32`）；
   - 2~3 个长 prompt（覆盖线上最长 prompt 的 50% 与 100% 长度，`max_tokens 16`，避免真生成）；
   - 1~2 个 thinking 请求（用生产采样参数：temperature 0.1 / top_p 0.95 / repetition_penalty 1.05）；
   - `stream=true/false` 各覆盖一遍；
   - 请求间间隔 5~10s，避免挤爆显存/排队；
3. **验证预热完成**：
   - 预热请求的 TTFT 显著下降并收敛；
   - `~/.cache/flashinfer` 下配置不再新增（文件数/时间戳稳定）；
   - 启动日志不再出现编译/autotune 相关输出；
4. 验证通过后再放量。

示例骨架：

```bash
#!/bin/bash
API=http://127.0.0.1:8000
MODEL=Qwen3.6-27B-FP8
# 1) 等服务 ready
for i in $(seq 1 120); do
  curl -sf -o /dev/null "$API/v1/models" && break
  sleep 5
done
# 2) 短 prompt 非 thinking
curl -sf "$API/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false},
  \"max_tokens\": 32
}" -o /dev/null
sleep 8
# 3) 长 prompt（用线上最长 prompt 长度的代表文本）
curl -sf "$API/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"<8000+ token 的代表性代码/上下文>\"}],
  \"chat_template_kwargs\": {\"enable_thinking\": false},
  \"max_tokens\": 16
}" -o /dev/null
sleep 8
# 4) thinking 请求（生产采样参数）
curl -sf "$API/v1/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"<代表性问题>\"}],
  \"temperature\": 0.1, \"top_p\": 0.95, \"repetition_penalty\": 1.05,
  \"max_tokens\": 16
}" -o /dev/null
```

**方案二：常驻 keep-alive**

- 每 30~60s 发一个轻量请求：短 prompt、非 thinking、`max_tokens 4`、`stream=false`；
- 作用：覆盖可能存在的懒加载/空闲后首请求开销；保持 autotune 缓存条目常热；兼做健康检查（非 200 即告警）；
- 成本：每请求毫秒级，可忽略；
- 注意：keep-alive **不能替代方案一**——它防的是"空闲后首请求慢"，不负责把内核编译完；
- 落地：`sglang_start.sh` 已内置 `--keep-alive`（默认 45s 间隔，`--keep-alive-interval` 可调）；K8s 中也可用 sidecar 容器实现；
- 部署：任意常驻进程/cron 均可（如 gateway 或独立 systemd timer）。

**方案三：autotune 缓存稳定后加 `--disable-flashinfer-autotune`**

- 流程：先按方案一预热并运行一段时间（覆盖线上主要 shape）→ 确认 `~/.cache/flashinfer` 不再增长 → **重启服务**并加 `--disable-flashinfer-autotune`；
- 原理：关掉后新 shape 不再触发 autotune（避免偶发卡顿），用启发式回退；
- 代价：若之后出现新 shape（如更长的 prompt 分块），可能用不到最优 kernel 配置——所以只在 shape 空间稳定后关；
- 该 flag 是启动参数，不能运行时切换，需要重启生效。

**与 10.6 的关系**：先定位（10.6），确认冷启动后再做本节方案；做完后回填 10.6 状态表，对比稳态 TTFT 是否回落。

### 10.8 生产数据全版对比与口径修正（MTP 低/高并发）

> 本节为线上实测数据（Qwen3.6-27B-FP8，6 卡 TP=2 DP=3，NEXTN steps=3 topk=1 draft=4），修正"有效 ITL"口径，并取代 10.6 的初步判断。

**关键指标对比**

| 指标 | 无 MTP (2294 req) | MTP 低并发 (692 req) | MTP 高并发 (2162 req) |
|------|------------------|---------------------|----------------------|
| ITL avg | 92.2ms | 34.8ms | 50.6ms |
| Accept Rate | - | 86.1% | 78.8% |
| Accept Len | - | 3.58 | 3.36 |
| TTFT (stream) | 3.75s | 1.35s | 8.76s |
| E2E | ~89s（实测 89.2s） | ~8s | ~41s（实测 41.1s） |
| Gen throughput | 112 tok/s | 335 tok/s | 723 tok/s |
| Cache hit | 43.2% | 59.9% | 37.6% |
| Abort | 2.2% | 0.3% | 3.3% |
| KV available | 3~4K | 97K~178K | 1.2~3.9K |
| KV/card | 793,189 | 746,595 | 746,595（draft 头占 ~46K，-5.9%） |

**有效 ITL 口径修正**

错误算法：`ITL avg ÷ accept_len`（50.6 / 3.36 = 15.1ms）。`ITL avg` 已经是每个输出 token 的实际流延迟——MTP 的收益已包含在内（一轮 verify ~170ms 出 3.36 个 token，平均到每个 token 就是 ~50ms），再除 accept_len 等于重复计算。

验证：

```text
648 tok/req × 50.6ms + TTFT 8.76s ≈ 41.5s  ← 与实测 E2E 41.1s 吻合
648 tok/req × 15.1ms + TTFT 8.76s ≈ 18.5s  ← 与实测不符
723 tok/s ÷ 36 并发 ≈ 20 tok/s/req ≈ 50ms/token  ← 一致
```

修正后真实收益：**每 token 流延迟 92.2 → 50.6ms（1.8x）；E2E ~90s → ~41s（2.2x）**。聚合吞吐 112→723 tok/s 含负载/并发差异，不能全部归因 MTP。

**低并发 vs 高并发：取舍而非 bug**

- 低并发：TTFT 1.35s、E2E ~8s、accept 86%、命中率 59.9%、abort 0.3%——**TTFT<1s / E2E<10s 目标在此 regime 基本达标**；
- 高并发：TTFT 8.76s、KV available 1.2~3.9K（紧张）、abort 3.3%、命中率 37.6%——瓶颈是 verify 与 prefill 争 GPU + KV 周转慢；
- 结论：MTP 高并发让位吞吐（723 tok/s），低并发让位延迟，两者不可兼得。

**建议**

| 问题 | 建议 |
|------|------|
| TTFT 高并发恶化 | steps 3→2 A/B（verify 工作量 -25%，accept len 预计 3.36→~2.8） |
| 目标冲突 | 按 SLA 分层：tool call（短交互）与代码检视（长输出）拆开或限并发 |
| Worker 1 热点（11 running vs 5/5） | 监控 prefix 分布，排查 prefix-aware 路由倾斜 |
| Abort 3.3% | 查原因分布（客户端超时 vs 服务端错误） |
| 冷启动 | 10.6 定位 + 10.7 预热方案仍未回填，先落地再测稳态 TTFT |

### 10.9 K8s 部署方案（两实例，无请求路由控制）【历史过程，已被 11.7 取代】

> **状态：已搁置（2026-08）**。双架构在无请求路由控制 + thinking 容量不足下实测更差（利用率 ~25% vs 统一 ~90%；thinking TTFT 随队列单调恶化至 ~100s）。当前生产方案回到 6 卡统一 + round_robin（见 11.7）；双架构仅在具备按类型分流能力和 thinking 容量 ≥4 卡后重启。

> 前提：pod 镜像为附录 C 固化的版本（含 sgl_stubs.so、load_utils 补丁、sglang-kernel 源码编译产物、flashinfer 0.6.13）。启动脚本：[sglang_start.sh](appendix/sglang_start.sh)（已适配 K8s：按 `CUDA_VISIBLE_DEVICES` 计数、空 `GPU_IDS` 不再禁用显卡、默认值按验证结论 0.85/98304/12）。

**为何是两实例而非三实例**：没有请求路由控制时，单卡轻量实例无法差异化——context 必须与其他实例一致（98304），否则长请求会被拒；而 98304 + TP=1 让它变成一个慢速通用实例，无意义。故放弃单卡实例，第 7 卡作热备/扩容预留。

**架构**

| 实例 | 卡数 | 架构 | 并发/worker | MTP | context |
|------|------|------|------------|-----|---------|
| tool call 主力 | 4 | TP=2 DP=2 | 5（可试 8） | NEXTN steps=2 draft=3 | 98304 |
| 代码检视 | 2 | TP=2 DP=1 | 12 | NEXTN steps=3 draft=4 | 98304 |
| 第 7 卡 | - | 热备/扩容预留 | - | - | - |

**启动命令**（pod 内端口统一 8000，不传 `--gpu-ids`，K8s 自动分配）

```bash
# 1) tool call 主力（4 卡 → 自动 TP2 DP2；round_robin 修复热点，warmup+keep-alive 防冷启动）
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --port 8000 \
    --max-running-requests 5 \
    --enable-speculative --speculative-num-steps 2 --speculative-num-draft-tokens 3 \
    --load-balance-method round_robin --warmup --keep-alive

# 2) 代码检视（2 卡 → 自动 TP2 DP1）
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --port 8000 \
    --max-running-requests 12 \
    --enable-speculative --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --load-balance-method round_robin --warmup --keep-alive \
    --priority-scheduling --proxy-port 8080
```

> 脚本默认已开启 priority 调度（`ENABLE_PRIORITY=true`），`--priority-scheduling` 可显式传；未带 priority 的请求默认 0（`--default-priority-value 0`），全 0 时等价 fcfs，无副作用。客户端走代理 8080，tool call 由代理注入 `priority=10`。

**注意事项**

- 两实例各自独立：KV 池/RadixTree/预热，context 一致 98304；
- 分流的真正前提是客户端能分别连接两个服务（如 agent 框架连 tool call 实例、检视应用连检视实例）；若所有请求都打同一个入口，拆分无效，退化为单实例 6 卡 + 优先级调度 + 网关限并发；
- 第 7 卡：热备；若必须全用 7 卡，唯一方案是检视改 3 卡 TP=3（需先验证 TP3 与 hybrid mamba + MTP 的兼容性）；
- tool call 实例并发先 5：`queue_time ≈ 0` 且 TTFT < 2s 再升到 8，TTFT 抬头即回落；
- 预热每个实例单独做（`--warmup` 或 10.7 预热脚本，挂在 initContainer/启动后脚本）；
- 横向扩容 = 加同配置 pod 副本 + 负载均衡，不要通过调高并发实现；
- 上线后按 10.6 定位表盯三个指标：`queue_time_seconds` / `cache_hit_rate` / `kv_available`；
- 请求侧参数（max_tokens / thinking_budget 等）按 9.2 / 9.3 场景取值。

## 11. 双架构生产数据与路由热点排查（2026-08）

### 11.1 两实例实测对比

| 指标 | tool call（4卡 TP2 DP2, steps2, 并发5） | thinking（2卡 TP2 DP1, steps3, 并发12） |
|------|------|------|
| 请求占比 | 278 (80%) | 69 (20%) |
| Gen/req | 160 tok | 2,546 tok |
| TTFT (stream) | 9.95s | 2.17s |
| ITL avg（每 token 实际流延迟） | 69.7ms | 32.2ms |
| Accept Rate | 97.6% | 60.8% |
| Accept Len | 2.96 | 2.83 |
| Cache hit | 30.8% | 39.1% |
| Running / Queue | 0+2 / 12 | 1 / 0 |

### 11.2 核心问题：DP Router 热点（致命）

tool call 实例 Worker 0 完全空转（0 running / 0 queued），Worker 1 满载（2 running / 12 queued）——相同 system prompt 的请求被**前缀粘滞路由**全塞到一个 worker，4 卡当 2 卡用，TTFT 9.95s 几乎全是排队。

**修复参数**：`--load-balance-method`（不是 `--dp-load-balancing`），choices：`auto` / `round_robin` / `follow_bootstrap_room` / `total_requests` / `total_tokens`。当前主线非 PD 默认 `auto`→`round_robin`，但 0.5.17 实测为前缀粘滞 → **显式加 `--load-balance-method round_robin`**。

**排查方法**：

```bash
python3.12 -m sglang.launch_server --help | grep load-balance   # 确认版本支持
# 看各 worker running/queue 是否均衡；热点特征是一个 worker 满载、其余空转
curl -s localhost:8000/metrics | grep -E "num_running|num_queue|gen_throughput"
```

round_robin 下每个 worker 首请求后各自缓存 system prompt，命中率不受影响；TTFT 因并行而下降。

### 11.3 口径修正

- "有效 ITL = ITL / accept_len" 是重复计算（同 10.8.2）：tool call 每 token **69.7ms**、thinking **32.2ms**；对应 E2E ≈ tool 21s / thinking 84s（非 13.7s / 30s）；
- accept rate 与 batch 大小无关：60.8% vs 82% 是任务分布/样本量（69 req）差异，不是"2 卡 batch 小"的锅；
- thinking non-stream TTFT 11.34s 是"等全部生成完才返回"的固有特性，非延迟问题；
- tool call **保留 MTP**（accept 97.6%，工作得极好）；问题是路由不是 MTP，关掉反而让 ITL 从 69.7ms 回到 ~92ms。

> **GLM 分析建议"tool call 不需要 MTP"，与实测不符**：其推理（短输出场景 verify 开销不划算）建立在输出 ≤50 token 且实例为 prefill/TTFT 瓶颈的前提上；实测 tool call 输出平均 160 token、accept 97.6%，两个前提都不满足。且其把 TTFT 9.95s 归因于 verify 争抢，实际是 DP Router 热点（排队）。验证方法：同一实例开/关 MTP 各跑 10 分钟，对比 ITL / E2E / TTFT / 吞吐，用数据定论。

### 11.4 关键洞察与行动

- **统一 6 卡方案的"均衡"可能是被混合流量掩盖的热点**：之前流量前缀多样所以看着均衡；tool call 专属实例前缀单一所以暴露。**回退统一方案时也显式加 round_robin**，TTFT 可能从 8.76~14.36s 明显下降；
- 行动顺序：回退统一 6 卡 + `round_robin` + MTP + 预热（`--warmup --keep-alive`）→ 盯 TTFT 与 GPU 利用率；
- 双架构待路由行为验证后再上（此时才需要客户端按类型分流）；
- thinking steps 保持 3（降 steps 只减 verify 开销，不会回升接受率）；
- 长期：PD 分离方向正确，但 PD + MTP/mamba 栈未验证，优先级最低。

### 11.5 脚本已内置

`sglang_start.sh` 新增 `--load-balance-method`（默认 `round_robin`），直接使用即可：

```bash
# 统一 6 卡回退验证（新方式）
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --port 8000 \
    --max-running-requests 12 \
    --enable-speculative --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --warmup --keep-alive
```

> 6 卡 pod 由 K8s 分配；若脚本自动推断不是 TP2 DP3（如检测到 7 卡），显式传 `--tp-size 2 --dp-size 3` 或用 `--gpu-ids`（裸机）限定 6 卡。

### 11.6 两实例最终启动命令（round_robin 修复后）【历史过程，已被 11.7 取代】

> **状态：已搁置（2026-08）**，保留作为"具备分流能力后的备选方案"。

> 已验证 `--load-balance-method` 在部署版本可用；脚本默认已是 `round_robin`。

```bash
# 1) tool call 主力（4 卡 pod → 自动 TP2 DP2）
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --port 8000 \
    --max-running-requests 5 \
    --enable-speculative --speculative-num-steps 2 --speculative-num-draft-tokens 3 \
    --load-balance-method round_robin --warmup --keep-alive

# 2) thinking / 代码检视（2 卡 pod → 自动 TP2 DP1）
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --port 8000 \
    --max-running-requests 12 \
    --enable-speculative --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --load-balance-method round_robin --warmup --keep-alive
```

验证点：启动日志出现 `load-balance: round_robin`；两个 worker 都有 running/queue（不再一个满载一个空转）；TTFT 明显下降、GPU 利用率回到 ~90%。tool call 并发先 5，`queue_time≈0` 且 TTFT<2s 后再试 8。

### 11.7 生产方案定版：回到 6 卡统一 + round_robin

**决策依据**：双架构总 decode 容量与统一方案相同（都是 3 个 TP2 worker），但没有路由控制导致隔离前提不成立，实测利用率 25%、thinking 队列单调恶化；统一方案数据稳定可预期，round_robin 同时修掉其隐藏的前缀粘滞热点。**生产选型优先稳定性。**

**生产配置（当前定版）**：

```bash
# 6 卡统一（K8s 6 卡 pod，或裸机 --gpu-ids 0-5）
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --port 8000 \
    --max-running-requests 12 \
    --enable-speculative --speculative-num-steps 3 --speculative-num-draft-tokens 4 \
    --load-balance-method round_robin --warmup --keep-alive
```

配套 API 侧参数（不可省）：

- thinking（代码检视）：`enable_thinking: true` + `max_tokens 4096`（Qwen3.6-27B 模板不支持 thinking_budget，见 9.3）；
- 非 thinking（tool call）：`enable_thinking: false` + `max_tokens 512`（见 9.2）。

**预期与后续**：

- round_robin 修掉前缀粘滞热点后，TTFT 应明显低于统一方案此前的 8.76~14.36s；
- 若 TTFT 仍高：为纯容量问题（thinking 长输出），走"非 thinking 检视质量 A/B"或换 H20/H100（见 11.8 硬件结论）；
- 双架构重启条件：具备按类型分流能力，且 thinking 容量 ≥4 卡（TP2 DP2）。

### 11.8 硬件适配结论（L40S × Qwen3.6-27B）

- L40S 跑 27B 的 decode 上限约 64 tok/s/卡（864GB/s ÷ 13.5GB/卡权重），MTP 后 ~120 tok/s/卡——**已接近软件极限，剩余瓶颈是显存带宽，不是配置**；
- 瓶颈是**显存带宽**（GPU 读自身板载显存），不是 PCIe：decode 每 token 读 13.5GB 权重（卡内），PCIe 只承担加载权重和 TP 激活值通信（KB 级/token），非瓶颈；
- 27B + 96K + thinking 长输出 + 高并发 的组合超出 L40S 物理能力；L40S 适合 ≤14B 模型或低并发/短输出；
- 出路：换 H20/H100（显存带宽 4~5x）、换小模型、或非 thinking 检视 + 结构化 prompt（质量 A/B 定论）。

### 11.9 TTFT 优化的非架构手段（优先调度 + 网关限并发 + round_robin）

> 目标：不碰 MTP、不动架构，把 tool call TTFT 从 7.19s 压向 3~4s，thinking TTFT 稳定不随队列恶化。

**三层机制**

| 层 | 手段 | 解决的问题 |
|----|------|-----------|
| 服务端 | `--schedule-policy priority` + `--enable-priority-scheduling` | tool call 排队时插到 thinking 前面 |
| 请求侧 | OpenAI 接口 `priority` 字段（tool call=10，thinking=0） | 同上（队列顺序） |
| 网关侧 | 按类型信号量限并发 + 超时 429 | thinking 占不满槽位，tool call 永远有位置 |

**限并发机制解读（8 + 12 为什么是"保护"而不是"浪费"）**

代理默认 `tool_call=16`、`thinking=12`，是**代理层全局并发上限**（不是 per-worker/per-GPU）；后端 6 卡 TP2 DP3 时 `max-running-requests=12/worker` 对应 **36 并发容量**。两层数字关系如下：

```
            ┌─ tool_call 闸门（16 个位置）─┐
客户端 ──► 代理                           ├──► SGLang（36 并发容量上限）
            └─ thinking  闸门（12 个位置）─┘
```

- **两个独立信号量**：thinking 请求永远只能占用最多 12 个位置，tool call 有自己的 16 个专用位置，thinking 再多也挤不到这 16 个；
- **thinking 是"长住户"**：一个 thinking 请求输出 2K+ token，占 GPU 数分钟、持续占 KV cache。不限的话 40 个 thinking 涌进来会占满 36 个槽位，tool call 即使 priority=10 插队也没用——GPU 全在跑长输出；限制 12 个后，GPU 永远有空槽位给 tool call 的 prefill 用，KV 也不会被 thinking 占死（保住 tool call 的 prefix 缓存命中）；
- **tool call 是"快进快出"**：输出 ~160 tok，2~3 秒完成释放位置，16 个位置轮转很快；
- **36 − 28 = 8 不是浪费**：槽位是容量上限不是工作岗位，GPU 打没打满看算力和显存带宽，不是请求数。28 个并发（含 12 个长输出）很可能已接近 L40S 饱和；硬塞满 36 会让每个请求变慢（实测高并发下 TTFT 1.35s→8.76s、abort 0.3%→3.3%）；
- **判断是否调大**：看 `nvidia-smi` GPU-Util。经常 90%+ → 现状合理；经常 30~40% → 请求量不足或配置过保守，可上调 tool_call（tool call 是主力场景时优先调它，thinking 保持 12 保护 tool call 延迟）。

**6 卡生产完整启动命令（默认值即生产配置，代理按需开启）**

`sglang_start.sh` 默认已内置生产配置：MTP 开 / keep-alive 开 / 预热开 / priority 开 / round_robin / mem=0.85 / context=98304 / max-running=12；**代理默认关**，`--proxy-port 8080` 开启后 tool_call=16 / thinking=12 为默认（传 `--proxy-tool-call-limit` / `--proxy-thinking-limit` 则用传入值）。生产推荐命令：

```bash
export LD_PRELOAD=/usr/local/lib/python3.12/site-packages/sgl_kernel/sgl_stubs.so
export TORCH_CUDA_ARCH_LIST="8.9"
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

bash sglang_start.sh \
    --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --port 8000 \
    --proxy-port 8080
```

默认值清单与覆盖开关：

| 配置项 | 默认值 | 覆盖开关 |
|--------|--------|----------|
| MTP 投机解码（NEXTN steps=3 draft=4） | 开 | `--no-speculative`（单卡验证不需要 MTP） |
| 前置代理 | 关 | `--proxy-port 8080` 开启；`--no-proxy` / `--proxy-port 0` 关闭 |
| tool call 并发上限（代理开启时） | 16 | `--proxy-tool-call-limit N` |
| thinking 并发上限（代理开启时） | 12 | `--proxy-thinking-limit N` |
| keep-alive | 开（45s） | `--no-keep-alive` |
| 启动预热 | 开 | `--skip-warmup` |
| priority 调度 | 开 | `--priority-scheduling` 仍可显式传 |
| per-worker 并发 | 12（TP2 DP3 → 容量 36） | `--max-running-requests N` |
| mem / context | 0.85 / 98304 | `--mem-fraction-static` / `--context-length` |
| 自适应限流 | 关 | `--adaptive-limit` 开启（需代理已开） |

> tool call 上限由启动参数固化（默认 16），运行中热调不持久化，pod 重启恢复启动默认值。
>
> 注意：**多实例拆分**（如 7 卡 4+2+1）时，只有显式开了代理的实例才占端口；若多实例都要代理，必须分别指定不同 `--proxy-port`，避免端口冲突。

**自适应限流（可选，替代人工盯曲线手调）**

`sglang_start.sh --adaptive-limit` 会额外拉起 [sglang_adaptive_limits.py](appendix/sglang_adaptive_limits.py) 控制器：定时读后端 `/metrics` 的 `sglang:num_queue_reqs` 与 `sglang:token_usage`，按规则自动调代理 `/admin/limits`（只调 tool_call；thinking 是保护阀默认不动，除非 `--adaptive-thinking` 显式开启）：

```bash
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --adaptive-limit \
    --adaptive-interval 15 \
    --adaptive-min-tool-call 4 \
    --adaptive-max-tool-call 24
```

| 规则 | 条件 | 动作 |
|------|------|------|
| 放宽 | 排队 > 6 且 token_usage < 0.92 | tool_call +4（后端有容量，放更多请求进） |
| 收紧 | 排队 < 2 且 KV 空闲（< 0.60）或已近上限 | tool_call −4（避免无谓占用 GPU/KV） |
| 收紧 | KV 吃紧（≥ 0.92）且仍有排队 | tool_call −4（防止驱逐风暴 / 请求超时） |
| 维持 | 其他情况 | 不动 |

- tool_call 波动区间默认 [4, 24]（`--adaptive-min/max-tool-call`），每次 ±4（`--step`），每 15s 评估一次（`--interval`）；
- 启动时先读代理当前 limits 作为基线（尊重人工热调值），读不到再用启动参数兜底；
- 后端 `/metrics` 连续不可达 12 次（约 3 分钟）自动退出，不影响代理与 SGLang；
- 控制器随脚本退出清理，`--kill-existing` 也会一并杀掉；
- 上线建议：先不加 `--adaptive-limit` 观察 1~2 天 baseline，再开启对比排队/TTFT 曲线，确认收敛行为符合预期；若想同时调 thinking，显式加 `--adaptive-thinking`（风险自担）。

启动后从 pod 外验证：

```bash
# 代理就绪
curl -s http://<service-ip>:8080/health
# 查询当前 limits（确认 16/12 生效）
curl -s http://<service-ip>:8080/admin/limits
# 运行中想再调（不持久化，重启恢复启动参数）
curl -X POST http://<service-ip>:8080/admin/limits \
  -H 'Content-Type: application/json' \
  -d '{"tool_call": 16, "thinking": 12}'
```

**服务端配置**（先 `--help | grep -iE "priority|schedule-policy"` 确认 0.5.17 支持）：

```bash
--schedule-policy priority \
--enable-priority-scheduling \
--default-priority-value 0
```

- 语义：priority 值越高越先调度（默认）；可选 `--disable-priority-preemption` 控制是否抢占；
- 注意：priority **只改队列顺序，不改 GPU 计算份额**，网关限并发不能省。
- 落地：`sglang_start.sh` 内置 `--priority-scheduling`，且**默认已开启**（自动切 `--schedule-policy priority` 并加 `--enable-priority-scheduling --default-priority-value 0`）；未传 priority 的请求默认 0，全 0 时等价 fcfs，无副作用。
- **只改启动脚本层的完整方案**：`sglang_start.sh` 加 `--proxy-port 8080` 会在脚本内同时拉起 [sglang_proxy.py](appendix/sglang_proxy.py) 前置代理（代理默认关）——客户端连代理端口，代理按类型限并发（tool call 16 / thinking 12，超时 429）、给 tool call 注入 `priority=10` 后转发到本机 SGLang；流式响应期间持续占用槽位。代理代码与脚本同目录，随镜像持久化。
- **限并发运行时可调（不用重启）**：`curl -X POST http://127.0.0.1:8080/admin/limits -H 'Content-Type: application/json' -d '{"tool_call": 12, "thinking": 16}'` 即时生效（重建信号量，在途请求不受影响）；**查询当前值**：`curl -s http://127.0.0.1:8080/admin/limits` → `{"limits": {"tool_call": ..., "thinking": ...}}`；启动默认值用脚本 `--proxy-tool-call-limit` / `--proxy-thinking-limit` 设置。调大后盯 TTFT 趋势，`num_queue_reqs` 上涨即回落。
- **K8s pod 场景（无法 exec 进 pod）**：代理监听 `0.0.0.0`，只要 K8s Service 暴露了 8080 端口，pod 外任意能访问 Service IP 的机器都能查/调：
  - 查询：`curl -s http://<service-ip>:8080/admin/limits`
  - 热调：`curl -X POST http://<service-ip>:8080/admin/limits -H 'Content-Type: application/json' -d '{"tool_call": 16, "thinking": 12}'`
  - 注意：热调**不持久化**，pod 重启后恢复启动默认值（`--proxy-tool-call-limit` / `--proxy-thinking-limit`）；若想把新值固化，需改启动参数后重建 pod；
  - 验收脚本已支持外部模式：`bash proxy_acceptance.sh --host <service-ip>`（backend 与 proxy 同 IP 时），无需进 pod。
- **其它接口透传**：代理只对 `/v1/chat/completions` 做限并发 + priority；`/health`、`/metrics`、`/v1/completions`、`/v1/embeddings` 等其它接口由兜底路由原样转发（含查询串与流式），健康检查和指标采集可直接走代理端口（8080）。
- **Anthropic 风格接口**：`/v1/messages` 按 `thinking.type == "enabled"` 分类并限并发（tool call / thinking 同信号量）；但 Anthropic 协议无 priority 字段，SGLang 的 anthropic 入口会拒绝未知字段，**无法注入 priority**——Anthropic 入口下 tool call 不能插队，只能靠限并发保底。
- **代理依赖验证**：`sglang_proxy.py` 只依赖 fastapi / uvicorn / httpx。fastapi 与 uvicorn 是 SGLang 的直接依赖（`python/pyproject.toml`），镜像内必有；httpx 为传递依赖，需实际验证：

```bash
# pod 内
kubectl exec <pod> -- python3.12 -c "import fastapi, uvicorn, httpx; print(fastapi.__version__, uvicorn.__version__, httpx.__version__)"
# 或 pip 列表
python3.12 -m pip list | grep -iE "fastapi|uvicorn|httpx"
```

判断：三个都有 → 代理直接用，无需安装；仅 httpx 缺失 → `pip install httpx` 后固化镜像；fastapi/uvicorn 缺失 → 镜像构建不完整，先查构建。
- **代理健壮性（已测试）**：后端不可达返回 502 且释放槽位（不会泄漏导致全量 429）；上游 4xx/5xx 原样透传（bytes，不走 JSONResponse 序列化）；并发满返回 429；流式期间持续占槽位、流结束释放。
- **两模式分类与错误格式（已测试）**：显式 `enable_thinking` 优先（false→tool call，true→thinking，即使带 tools）；chat 类错误体为 OpenAI 对象格式 `{"error":{"message","type","code"}}`，messages 类为 Anthropic 格式，避免 SDK 客户端解析炸裂。
- **热调信号量竞态（已修复并测试）**：`/admin/limits` 热换信号量时，在途请求的 release 会捕获**同一信号量对象**，不会释放到新对象导致超限；透传路由支持 OPTIONS/HEAD/PUT/DELETE（CORS 预检可用）；json 转发剥离客户端 content-type。
- **请求/响应头透传（已测试）**：chat/messages 路由转发 Authorization 等请求头（剥离 host/content-length/connection）；响应头（如 x-request-id）原样回传；流式与非流式均逐块透传，body 不做解析改写。
- **无缝使用检查清单（配合脚本联动）**：
  - 客户端 / K8s Service 指向代理端口（8080），就绪与存活探针用代理 `/health`（透传后端）；`/metrics` 走 `8080/metrics`；
  - SGLang 开 `--api-key` 不受影响：Authorization 请求头已透传；就绪探测改用 `/health`（免鉴权），避免 `/v1/models` 探测因鉴权失败；
  - `--kill-existing` 会连 `sglang_proxy.py` 一起清理，防止 kill -9 残留的孤儿代理占住 8080；
  - 代理启动后自检：120s 内 `/health` 未就绪打警告（不阻塞主服务）；
  - chat/messages 查询串（如 `?stream=true`）已透传；
  - 上线前做 diff 验证：同一请求直连 8000 与走 8080，响应（body+headers）应完全一致。
- **一次性能收脚本**：[proxy_acceptance.sh](appendix/proxy_acceptance.sh) 自动跑完无缝清单——body 一致性（归一化 id/created）、`/health` 与 `/metrics`、流式 SSE、tool call priority=10 注入、可选 429 上限测试（`--test-429`，瞬时影响生产需低峰）、代理进程/孤儿检查；输出 PASS/FAIL/SKIP 汇总，全绿即闭环。K8s 场景从 pod 外验收时用 `--host <IP>`（后端与代理同 IP）或 `--backend-host` / `--proxy-host` 分别指定。

**请求侧**：

```json
// tool call：加 priority 10
{ "priority": 10, ... }
// thinking：默认 0，可不传
```

**网关侧限并发**（超时直接 429，不让请求进服务器排队）：

```python
import asyncio
from contextlib import asynccontextmanager

LIMITS = {"tool_call": 8, "thinking": 12}   # tool call 保底槽位，thinking 封顶
sems = {k: asyncio.Semaphore(v) for k, v in LIMITS.items()}

@asynccontextmanager
async def guard(kind: str, timeout: float = 10.0):
    try:
        await asyncio.wait_for(sems[kind].acquire(), timeout)
    except asyncio.TimeoutError:
        raise   # 上层 catch 后返回 429/503，绝不进服务器排队
    try:
        yield
    finally:
        sems[kind].release()

# 处理处：
# kind = "tool_call" 或 "thinking"（按入口 path 或 chat_template_kwargs.enable_thinking 区分）
# async with guard(kind):
#     resp = await forward_to_sglang(payload)   # 转发时给 tool call 注入 priority=10
```

**预期效果**：round_robin 消 Worker 热点；priority=10 让 tool call 不被长 thinking 挡住；thinking 并发 ≤12 封顶后 TTFT 稳定（不再出现 100s 级队列）；tool call 并发 ≤8 防自爆。组合后 tool call TTFT 向 3~4s 收敛。

**验证 priority 是否生效**：

```bash
curl -s localhost:8000/metrics | grep num_queue_reqs
# 期望（有请求排队时）：
# sglang:num_queue_reqs{priority="10", ...} N   ← tool call 排队数
# sglang:num_queue_reqs{priority="0", ...}  M   ← thinking 排队数
```

- 出现 `priority="10"` 的 label → 请求侧字段生效，优先级调度在工作；
- 所有排队请求都只有 `priority="0"` → tool call 没带上字段，检查 API 调用处；
- 完全没有 priority label → 服务端未开 `--enable-priority-scheduling`（per-priority 统计依赖该开关）；
- 启动日志：脚本打印 `priority-scheduling: on/off`；
- 需要单条请求元数据时可开 `--log-requests`；
- 注：未设 `--default-priority-value` 时，未带 priority 的请求会显示为 `priority="None"`；脚本已默认设为 0。

### 11.10 上下文窗口扩展分析（96K → 256K）

> 当前生产基线见 0 章速查（2026-08：16454 req / abort 1.88% / TTFT 8.37s / E2E 33.6s）。

**Qwen3.6-27B 上下文 262144 可行性**

模型支持 262144（原生 max_position_embeddings），L40S 上**技术上可行，但并发要砍到约 1/3**：

| 项目 | 96K（现状） | 262144（目标） |
|------|------------|---------------|
| 每请求每卡 KV（TP2 分摊） | ~2GB（48K tokens/卡） | ~5.4GB（128K tokens/卡） |
| KV 池（0.85，约 16GB/卡） | 793K tokens/卡 | 793K tokens/卡（不变） |
| 满长度并发上限 | 12/worker（实测） | 约 3~4/worker |
| MTP draft 头占用 | ~46K tokens/卡 | 同左 |

推算依据：96K × 12/worker 时池子占用 ~99.7%（available 仅 2~4K），池子容量不变，256K 单请求占用为 96K 的 2.67 倍 → 12 ÷ 2.67 ≈ 4.5，留 MTP 与余量后 3~4/worker 较稳，DP3 总并发约 9~12。

**真正的代价（不只是并发）**

1. **TTFT 变长**：96K 单请求 prefill 约 1.35s，256K ≈ 3.5~4s+（无缓存），长请求一多互相拖，TTFT 破 10s 是常态；
2. **缓存命中率下降**：池子被 256K 占满后 evict 更频繁，43% 命中率会下滑，prefill 计算量反增；
3. **MTP 收益变小**：spec buffer 固定占 ~46K，长上下文下池子更紧，accept rate 可能下降。

**结论与建议**

- 先量实际 prompt 分布：代码检视若真实输入仅 20~50K，96K 上限完全够用（context-length 是上限不是常驻占用），**不要为参数好看牺牲并发**；
- 确认有 150K+ 真实需求再做实验，验证命令：

```bash
bash sglang_start.sh --model-path /usr1/project/models/Qwen3.6-27B-FP8 \
    --context-length 262144 \
    --max-running-requests 3 \
    --mem-fraction-static 0.88 \
    --no-proxy
```

- 盯三个数：启动日志有无 "reduced from the requested"（池子被钳）、TTFT（长 prefill 底线）、`nvidia-smi` 显存；OOM 则退回 mem=0.85；
- 生产若真上 256K，建议 PD 分离（长 prefill 单独处理）而非单实例硬扛。

### 11.11 后续观察点与行动阈值

基于 2026-08 基线（16454 req / abort 1.88% / TTFT 8.37s / E2E 33.6s），持续盯以下指标。监控平台接入见附录 G。

**观察矩阵（按优先级）**

| 优先级 | 指标 | 阈值 | 判定 | 行动 |
|--------|------|------|------|------|
| P0 | abort 率 | 单日 >2% 且趋势上升 | KV 紧张 / TTFT 超时 | 降 tool_call（`--proxy-tool-call-limit`）或 thinking；查 `token_usage` |
| P0 | `token_usage` | ≥0.92 持续 5m | KV 池吃紧，驱逐风暴前兆 | 降并发 / 收紧 context / 若 mem<0.85 可回 0.85 |
| P1 | TTFT p90（stream） | >10s 持续 10m | 排队或 prefill 争抢 | 看 `num_queue_reqs{priority="10"}` 占比；调 priority / 限流 / 扩副本 |
| P1 | `num_queue_reqs` | >10 持续 5m | 容量瓶颈 | 后端容量已到顶 → 考虑 HPA 扩副本（见附录 G 第 6 节） |
| P2 | `spec_accept_rate` | <65% 持续 15m | MTP 收益下降 | 检查 batch 干扰 / 降 steps |
| P2 | `kv_available_tokens` | 贴 0 且 evict 激增 | 缓存被挤爆 | 缓存命中率下滑 → 降并发或加池子 |

**每天看一遍的例行检查**

1. abort 率（目标 <2%）与 TTFT p90（目标 <10s）是否在阈值内；
2. 高峰时段 `token_usage` 峰值（记录当天的最高值，观察趋势是否逐日抬升——抬升说明 KV 需求在涨，需提前降并发）；
3. 代理日志 `[adaptive-limits]` 调整次数（若开启）：一天内调整过于频繁 = 参数震荡，调大 `--adaptive-interval`；
4. `nvidia-smi` GPU-Util 与温度（长期 90%+ 正常，100% 满负荷注意温度）。

**触发升级的动作清单**

| 现象 | 第一步 | 第二步 |
|------|--------|--------|
| abort 率连续两天 >2% | 热调 `tool_call` 降 4（`POST /admin/limits`） | 观察 1 天；未好转查 KV 与 TTFT |
| TTFT p90 >10s 且 queue 高 | 确认 priority 生效（`num_queue_reqs` 有 `priority="10"` label） | 降 thinking 并发（保 tool call） |
| token_usage 持续 >0.92 | 降 `max-running-requests` 或 mem 回 0.85 | 评估是否真需要 96K（见 11.10） |
| 一切正常但想压 TTFT | 确认 `--adaptive-limit` 开启且无震荡 | 按附录 G 6 节流程评估固化默认值 |

### 11.12 PD 分离可行性（6 卡 L40S）

**结论：6 卡做 PD 分离收益不大，大概率负收益。** PD 分离的典型收益场景是大规模部署（几十卡），6 卡拆完后每个角色都太小，且 L40S 无 NVLink，KV 传输走 PCIe。

**6 卡拆法对比**

| 方案 | P/D 分配 | D 组 KV 池 | 总并发 | 对比现状 |
|------|---------|-----------|--------|----------|
| 现状（统一） | — | 6 卡 × ~16GB ≈ 96GB | 36 | 基准 |
| 2P + 4D | 2 卡 prefill + 4 卡 decode | 4 卡 × ~16GB ≈ 64GB | 24 | 池子 −1/3，并发 −1/3 |
| 4P + 2D | 4 卡 prefill + 2 卡 decode | 2 卡 × ~16GB ≈ 32GB | 12 | 不可行 |

根因：PD 分离需要 P/D 两套实例**各自完整加载 27GB FP8 权重**，L40S 46GB 显存下权重吃掉一大半，拆完 D 组显存池反而缩小，总容量下降。

**收益与代价**

| 维度 | 影响 |
|------|------|
| TTFT | 会降（P 组专卡 prefill 不被 decode 拖累，8.37s → 可能 3~4s），是 PD 唯一大卖点 |
| KV 传输 | L40S 无 NVLink，PCIe 4.0 ~25GB/s 实际；96K 单请求 KV ≈ 1.5GB，传一次 ~60ms+，36 并发下传输总量可观 |
| MTP 兼容 | draft/verify 跨实例复杂，3.4x decode 加速可能打折，支持度需现验证 |
| 总吞吐 | 并发 36 → 24，E2E 不一定变好（只省 TTFT 几秒，decode 长输出固有省不掉） |
| 运维 | 两套实例 + bootstrap 服务 + KV 路由，故障面翻倍；RadixTree 前缀缓存跨实例命中复杂化 |

**判断依据**（见第 13 章 13.6 节"什么时候不该用"）：单机低负载、短输出场景、无独立网络——6 卡 L40S 全中。

**建议**

1. 当前 TTFT 8.37s 是 MTP 高并发的正常水位（对照实测 8.76s），不是"prefill 被 decode 拖到极限"，PD 是为那种极限场景设计的；
2. 想压 TTFT 的性价比排序：自适应限流（0 成本）→ tool call 专享低并发（低并发 MTP 实测 TTFT 1.35s）→ 扩容到 12+ 卡再考虑 PD（4P + 8D 才摊得开权重与池子）。
