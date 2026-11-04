# Session KV 实验脚本

`session_kv_bench.py` 对已运行的两个 SGLang Engine 发 HTTP 请求，比较普通 HiCache 按需恢复与独立 L3→L2 PREFETCH。脚本不启动或停止服务，不安装依赖，不修改服务配置。两个 Engine 使用同一权重 namespace、相同 layout、page size、模型、缓存容量与负载配置，共享同一个 Mooncake。A 产生 KV，B 恢复请求。

脚本与验证用例位于开发分支的 benchmark/session_kv_control 目录。

## 使用

环境中已存在的 Python：

```bash
export BENCH_PY=/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/.venv/bin/python
export BENCH_DIR=/home/youwei/bzh/project/TokenCake/RFC/sglang-rfc/benchmark/session_kv_control
```

用实际的 Engine A/B 地址替换下例 `51801`、`51803`。这两个端口只是调用示例，脚本不会建立服务。`51800` 是独立的静态参考服务，只读取其配置，不混入正式配对统计。

短验证：

```bash
"$BENCH_PY" "$BENCH_DIR/session_kv_bench.py" run \
  --engine-a http://127.0.0.1:51801 \
  --engine-b http://127.0.0.1:51803 \
  --reference-url http://127.0.0.1:51800 \
  --model /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --output /tmp/sglang-rfc-build/benchmark-smoke \
  --seed 1729 --prefix-lengths 2048 \
  --tool-waits 0.1 --lead-times 0.1 \
  --repetitions 1 --warmup-repetitions 1 --output-tokens 8 \
  --groups on_demand,prefetch \
  --negative unused,missing,cancel,route_change \
  --cold-mode flush
```

完整长度与等待时间矩阵：

```bash
"$BENCH_PY" "$BENCH_DIR/session_kv_bench.py" run \
  --engine-a http://127.0.0.1:51801 \
  --engine-b http://127.0.0.1:51803 \
  --model /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --output /tmp/sglang-rfc-build/benchmark-matrix \
  --seed 1729 --prefix-lengths 2048,8192,16384 \
  --tool-waits 0.05,1 --lead-times 0,0.05,0.5 \
  --repetitions 20 --warmup-repetitions 1 \
  --groups auto --cold-mode flush
```

输出目录内不能已有 `raw.jsonl`。默认 run ID 随机生成并存入 manifest；固定 seed 复现 token 材料和组内顺序，每次运行使用新的 salt 避免读到上次实验的 L3 内容。服务器也应显式使用相同的 `--random-seed 0`。访问凭证可通过 `--api-key-env SGLANG_API_KEY` 指定环境变量，凭证值不写入请求日志。

重新计算汇总与运行本地验证：

```bash
"$BENCH_PY" "$BENCH_DIR/session_kv_bench.py" summarize /tmp/sglang-rfc-build/benchmark-matrix
cd "$BENCH_DIR"
"$BENCH_PY" -m unittest -v test_session_kv_bench
```

## 对照与缓存证据

每个 `(prefix长度, 工具等待, 提前量, repetition)` 是一组配对。组内动作顺序由固定 seed 随机化。所有组使用相同 token 输入、greedy sampling 与输出长度。前缀来自确定性生成的普通英文词和 ledger 数字，按本地 tokenizer 截成准确长度；恢复请求附加同一工具结果文字。这里测量合成文本轨迹的缓存迁移，不代表真实工具或 RL batch 的完整吞吐。

每个 trial 使用独立原始 cache salt。控制 target 和推理请求使用同一 salt，namespace 编码交给 Engine。B 在开始时通过 `residency` 核验目标在 GPU 和 host 中均无副本。A 用 `max_new_tokens=1` 计算完整输入前缀，验证源请求 device/host/storage 命中全为零，再执行 GPU→storage `copy_to` 并等待 terminal、quiescent 和完整 `ready_scope`。提交返回 HTTP 200 不算发布完成。前缀 publication 失败时保留失败 trial，不进行缺乏 L3 前提的恢复对照。

工具等待从确认源 storage 发布后计时，所有组使用同一个等待时长。PREFETCH 在 `等待结束 − 提前量` 发起；提前量大于等待时间时从等待开始发起，记录有效提前量。恢复请求按预定时刻发送，不额外等待 PREFETCH 结束。lead=0 包含准备来不及的情况。无论成功、部分完成、缺失、拒绝或失败，原始控制状态与范围都保留。

`--cold-mode flush` 额外调用 B 的普通 `/flush_cache`，确认成功后刷新 capabilities/context/epoch，再做 residency 核验。当前源码路径 `Scheduler.flush_cache → UnifiedRadixCache.reset/_reset_full → HiCacheController.reset` 清本地树、GPU/host 池和队列，并重启 storage worker；该路径没有调用 `backend.clear`。独立的 HiCache clear 接口会删除 storage，脚本不调用它。源码核验基于本任务的 `sglang-rfc` 代码；换用其他版本时应核对其 reset 实现。默认 `--cold-mode salt` 依靠新 salt 与 residency 排除目标命中，保留旧 trial 的其他内容，适合观察连续实验中的自然容量压力。需要统一每次目标 Engine 起始占用时使用 `flush`。

源计算、L3 发布与工具等待分别记录。完整前缀 source publication 是所有组共同的实验前提；源写出前缀的时间不混入“恢复 TTFT”。完整周期仍包含这些时间。

自动动作组按运行时能力选择：

| 组名 | 工具等待期间的动作 | 恢复节点 |
| --- | --- | --- |
| `on_demand` | 无控制预取 | B，普通 HiCache L3 按需读取 |
| `prefetch` | B 独立 storage→host PREFETCH | B |
| `demote` | A GPU→host DEMOTE | B，普通按需读取 |
| `pin_demote` | A DEMOTE 后，对 A host 做有界 PIN | B，普通按需读取 |
| `pin_demote_prefetch` | 相同 A DEMOTE/PIN，加 B PREFETCH | B |

`demote/pin_demote` 主要验证等待期间的源保留与回收，不预设会降低 B 的 TTFT。PIN 完成恢复后显式 release，TTL 有限。PIN 校验独立于 transfer：要求 `public_result.claim` 为 active、TTL>0、`effective_protection=eviction_exclusion`，`actual_scope` 覆盖全部目标；`ready_scope` 可以为空。release 后用 `kind=claim` 查询 released/expired/invalidated 的终态及 tombstone，不能只凭 release 请求返回成功。优先读取顶层 `public_protocol.actions/paths`，使用 `sglang.prefetch` 的 `source/destination` 字段；旧服务只在 capabilities 明确给出对应 `copy_to` 路径时采用 `src/dst`。schema 中存在某动作不算支持。A/B 的 backend_namespace 必须一致；public capability 声明支持时，PREFETCH 显式发送该 namespace，legacy schema 不填该字段。

普通基线必须在真实反馈中显示 device=0、host/cpu=0、storage 至少覆盖整个前缀。命中 GPU 或 L2 的基线会被标为无效。已在请求发送前完成的 B PREFETCH 必须在恢复请求中显示 host/cpu 命中相同范围。完整 output IDs 和文本在条件之间比较，不一致时标为无效。源请求的确定性输出也会核对。

## 指标

`ttft_ms` 从发送流式恢复 HTTP 请求计时，到首个实际输出 token。空文本/空 prefill metadata 不算首 token；`output_ids` 或 `completion_tokens>0` 用于确认输出。读取使用 `urllib`/`HTTPResponse.readline`，不受 requests 默认 512 字节 `iter_lines` 块门槛影响。流末尾需有 `finish_reason`。`latency_ms` 覆盖整个恢复流。生成前，A/B 都按每种长度执行单独 warmup，使用独立 salt；warmup 存在 raw 记录中，排除出 trial 统计。

`complete_cycle_ms` 包含 A 初次计算、确认 L3 发布、完整工具等待和恢复生成；它截至响应完成。`quiescent_cycle_ms` 继续包含后台控制收尾、PIN release 和末尾观测。`initial_and_publication_ms`、`actual_tool_wait_ms` 和 PREFETCH 提交/轮询经过时间单独存储。控制经过时间包含排队和轮询分辨率，不能直接当作纯 I/O 时间。raw HTTP 记录提供每次 RPC 的墙钟经过时间与客户端线程 CPU；服务端 CPU 需要服务端遥测。

`cached_tokens_details` 中 `host` 与 `cpu` 统一记为 `cpu`；保留原始字段。分层 tokens 之和必须与 `cached_tokens` 相等，缺失或不一致时不猜测命中层。`repeated_prefill_tokens` 是已发布源前缀中没有通过任何缓存层复用的 token 数，不包含新增工具后缀。

`prepared_tokens` 使用 ready_scope 的 FULL token 区间并集，避免重复结果和 component 放大数量。预取在恢复请求发出前已就绪、且随后 host 命中同一冷目标时，`used_prepared_tokens` 为二者范围长度的交集，`used_prefetch_ratio` 为使用 tokens/准备 tokens。若预取和按需读取并发、仅能看到随后 host 命中，归因留空并标记 ambiguous。无恢复请求或路由离开 B 的准备范围计为 unused。逻辑 token 数不冒充物理传输字节；共享 job 物理 I/O、纯 I/O 耗时、host 峰值没有遥测时保持 `null`。`/metrics` 的有关样本会保存在 raw 中，供结合服务端计数器分析。

主延迟统计包含每条已经完成的恢复请求，包含控制失败与无效缓存路径，避免只选择成功预取。`valid`、失败原因与 `validated_*` 子集单独列出；无效路径无法证明收益。配对差值为 `候选组 − on_demand`，负值表示延迟较低。p50/p95/p99 使用线性插值；少量重复只适合功能验证，不能据此判断尾延迟稳定性。

## 负例与边界

`unused` 预取到 B 后不发送恢复请求；`missing` 查询未发布 salt，要求明确缺失且 ready_scope 为空；`cancel` 提交后请求取消并等到 quiescent；已完成再收到取消时记录 `cancel_raced_completion`，不能把这种情况当作成功打断 I/O 的证据；`route_change` 准备 B 后恢复到 A，明确记录 B 的未使用准备。取消超时会继续请求 drain 并等待第二个有界窗口；仍不 quiescent 时保留完整错误与引用。

`--router URL` 只读取 Router capabilities 并保存。当前 harness 固定恢复节点 B，尚未实现 B3 的 Router 就绪感知选路；缺少经核验的 worker attribution 与 Router session event 协议时，不将 HTTP 成功解释为完成 B3。静态参考服务也只采集配置。默认负载为每轮一个源请求和一个恢复请求；可选背景流量与 host 压力见下文。共享前缀比例尚未单独扫描。实验期间应避免未记录的其他流量占用 A/B。

真实 B3 选路使用独立入口 `router_bench.py --router-mode b2-b3`，协议和用法见 [Router 实验说明](ROUTER_BENCH.md)。原入口的 `--router` 仍保留能力采集用途。

## 本地轨迹输入

`--trace-jsonl FILE` 使用本地记录，替代合成长度矩阵。每行须有唯一 `id`、`token_prefix`（完整 u32 token 数组）或 `prefix_text` 二选一、`tool_suffix`（非空 token 数组或文本）、`tool_wait_s`。token 前缀必须符合实际 page 对齐；文本按本地 tokenizer 编码后同样检查，不自动截断原轨迹。可选 `lead_s` 覆盖命令行 lead，`model`/`tokenizer` 路径作为身份断言，必须与 `--model` 一致。

目录内的 `example-trace.jsonl` 是 64-token 合成 fixture，只供格式演示，不能标为真实应用数据。示例调用：

```bash
"$BENCH_PY" "$BENCH_DIR/session_kv_bench.py" run \
  --engine-a http://127.0.0.1:51801 --engine-b http://127.0.0.1:51803 \
  --model /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --trace-jsonl "$BENCH_DIR/example-trace.jsonl" \
  --output /tmp/sglang-rfc-build/trace-example \
  --lead-times 0,0.25,0.5 --repetitions 6 --cold-mode flush
```

每条 trial 保存源文件绝对路径、整个文件 SHA256、行号、原始行 SHA256、token prefix/suffix SHA256，以及本地模型/tokenizer 文件哈希。`declared_origin` 保留提交者标签，`origin_independently_verified` 恒为 false；导入 JSONL 不会证明其来自真实生产轨迹。记录自带 wait 时不再套用 `--tool-waits`，自带 lead 时不再套用 `--lead-times`。同长度的不同轨迹按各自 input_case 比较确定性输出，避免串行比错内容。

## 提前量实验与 EMA

固定矩阵继续使用 `--lead-times`。先用相同 wait 扫 lead、同时保留配对按需基线，可以观察准备来不及与占用过早的变化。`--lead-policy ema` 改为在每组配对开始前，从已完成的历史准备中选择同一 lead；不会看到本组未来结果。实际 lead 写入 trial，配对内各组使用相同选择。EMA 只用于参考策略实验，不自动承诺收益。

```bash
"$BENCH_PY" "$BENCH_DIR/session_kv_bench.py" run \
  --engine-a http://127.0.0.1:51801 --engine-b http://127.0.0.1:51803 \
  --model /home/youwei/bzh/model/Qwen/Qwen3-8B \
  --output /tmp/sglang-rfc-build/ema-experiment \
  --prefix-lengths 2048,8192,16384 --tool-waits 3 --repetitions 6 \
  --lead-policy ema --ema-alpha 0.25 --ema-safety-factor 1.5 \
  --ema-initial-ms 2000 \
  --cost-history-jsonl /tmp/sglang-rfc-build/bench-p2-matrix-v1/raw.jsonl \
  --groups on_demand,prefetch --cold-mode flush
```

成本桶按 namespace、backend_namespace、layout 和 token 范围长度区分。参考时间来自真实 operation 的 terminal/quiescent 完成观测，记录 `timing.completion_elapsed_ms`，不包含 ack。`completion_interval_ms` 给出上次非终态 RPC 发出到首个完成响应收到的保守区间，覆盖轮询不确定性；它仍包含排队、I/O 与 host 登记，纯 I/O 时间留空。物理字节没有遥测时不推断；可从 FULL 模型配置或 `--kv-bytes-per-token` 保存逻辑 KV 估计，用于字节/时间 EMA。

正式请求抵达时仍在途的准备进入 contended 成本桶，并标记 usage ambiguous，不计 unused。仅 uncontended 完成样本影响所需 lead；没有样本时用明确的 `--ema-initial-ms`，`sample_n=0`。本轮 16K 没有 uncontended 样本，导入该批历史仍会使用初始值。预测时间乘 safety factor 后与可用 wait 比较；不够时记录 `window_insufficient=true`，显式实验仍从等待开始提交。EMA 和 estimated bytes 写入 `prefetch-cost-ema.json`，历史文件路径与哈希写入 manifest。

summary 同时提供全部已完成恢复、验证通过、`prefetch_ready_before_resume` 和未提前就绪样本。ready 子集属于观察后的条件选择，只能辅助归因。配对 mean difference 的 95% bootstrap CI 对完整配对差值重采样，N、seed、方法和 4,000 次重采样均明确记录，失败样本与缺失的配对数保留。

## 可选背景负载与 host 压力

`--background-concurrency N` 在确认源发布后，向 B 运行 N 个封闭循环的普通推理客户端，直到 foreground 恢复结束。默认关闭。`--background-prefix-tokens` 与 `--background-output-tokens` 定义输入/输出长度；每个请求使用与 foreground 分离的新 salt。所有组使用相同配置。它会产生真实 GPU/host 负载，只有显式设置才运行。

吞吐以测量窗口内成功完成的请求和输出 tokens 除以窗口秒数计量。窗口结束后的在途请求继续 drain，分别记录完成数、失败数和 drain 时间，不算作窗口内吞吐。raw 保存每个背景请求及 TTFT，summary 保留失败和吞吐分布；不能把仅到达率当成完成吞吐。短窗口有边界偏差，应使用足够长的工具等待与重复。

`--host-pressure-prefixes N --host-pressure-tokens T` 在每轮计时前，将 N 个无关、独立 salt 的前缀生成并登记到 B host，通过 residency 核验实际范围。能力允许时用 DEMOTE 回收 GPU；仅支持 copy 时明确记录仍可能占 GPU。这些内容不 PIN，会承受普通驱逐，不以请求 tokens 冒充 host 字节峰值。压力准备耗时不计恢复周期，原始请求和观察范围保留。使用 `--cold-mode flush` 能使各 trial 的初始本地缓存状态更可比。

`--bandwidth-profile-json FILE` 只记录外部带宽配置、来源声明和文件哈希，例如网卡/协议/运营者测得带宽。脚本不运行 tc、不调整 Mooncake、不注入延迟；`applied_by_harness` 和 `independently_measured` 均为 false。实验脚本的配置记录不能替代带宽实测证据。

## 本次验证

2026-11-04，使用现有 `.venv/bin/python`：

- `python -W error::ResourceWarning -m unittest -v test_session_kv_bench`：8 项通过。覆盖计账、区间并集、SSE 换行/截断、保留失败样本、真实 SGLang SSE 记录重放，以及 mock HTTP 双 Engine 的 9 种动作组、无效 GPU 基线、public capabilities 和 legacy fallback。新增轨迹来源校验、EMA/配对 bootstrap、P4b claim、late attribution、取消 drain unknown，以及本地模拟的背景负载/host 压力整次实验。
- 对真实静态 SGLang `http://127.0.0.1:51800` 发了 1 次短请求：16 个 input tokens、3 个 output tokens，解析成功。服务器为 Qwen3-8B、A100、page64、Mooncake v0.3.14。此请求只验证 HTTP/SSE 解析，未作为性能实验。记录在 `real-parser-smoke.jsonl` 和 `real-parser-smoke-result.json`。
- 父任务已在 feature A/B 运行 `/tmp/sglang-rfc-build/bench-p2-matrix-v1` 的 36 条真实对照，36 条通过路径与输出校验；分析见 `p2-matrix-analysis.md`。2K 提前准备降低 TTFT，8K 效果不稳定，16K 在 0.5s 提前量下平均变慢。扩展的 trace/EMA/背景负载与 host 压力仅在 mock HTTP 中验证，本次未发起新的 GPU 实验。

交付文件：`manifest.json` 保存配置与版本，`schedule.json` 保存完整顺序，`raw.jsonl` 保存 HTTP/生成/warmup/trial 原始记录，`summary.json` 和 `summary.md` 提供可重新计算的统计。退出码 0 表示全部 trial 通过校验；2 表示存在无效/失败 trial；1 表示启动、能力或配置检查失败。
