# SGLang Perf Patch 使用指南

`python/sglang/srt/patches/batch_decode_scheduler/` 是一套 GPU 端到端性能基准工具，
用于把 SGLang 的 prefill / decode 延迟与 **RTP-LLM `grid_perf_test`** 和
**vLLM `batch_decode_scheduler` patch** 做**同口径对比**（三方可直接拼一张表）。

| 模块 | 作用 |
|---|---|
| `perf_test_harness.py` | 进程内驱动一个**真 `Scheduler`**，按 `get_next_batch_to_run → run_batch → process_batch_result` 单步推进并计时；提供 fake-KV、profile |
| `perf_test_runner.py` | CLI 入口：跑 `batch_size × seq_len` 网格，聚合输出表格 / CSV；支持 TP / PP / DP / EP |
| timeline 分析 | **复用 vLLM 侧** `vllm/patches/batch_decode_scheduler/perf_test_timeline.py`（吃 chrome trace，与引擎无关，见 §7） |

**计时口径**（与 RTP-LLM / vLLM patch 逐字对齐，全文适用）：

- 所有延迟为 **trimmed mean**：按轮排序、丢掉最小和最大、其余取平均（≥3 轮时），
  对齐 RTP `batch_perf_impl.run` 的 `measurements[1:-1]`，**不是 p50**。
  样本单位是**轮**（`--num-iters`），不是步
- `cost_ms` ≈ RTP `cost_time`；`prefill_ms` ≈ RTP `first_token_cost_time`；
  `per_token_ms = (cost − prefill) / num_decode_steps` ≈ RTP `decode_time_per_token`
  （分母口径与 RTP `output_len − 1` 的换算见下面「per_token 分母换算」，
  `--partial 1` 存在一处 off-by-one）
- prefill 网格里 `seq_len` 是输入长度；decode 网格里 `seq_len` 是 KV 长度（kv_len）
- 每步带相位断言：batch 被拆分 / prefill 被截断会**直接报错**而非产出错误数据
- **每轮只有 3 次 device synchronize**（`mark_batch_start` / `mark_lap` /
  `mark_batch_end`），测量循环内部零同步，decode 步间 overlap 完整保留。
  逐步计时是 `--per-step-timing` 的诊断路径，单列一节，**不可与 `cost_ms` 相加**

### per_token 分母换算（与 RTP 对齐时必看）

RTP 的 `decode_time_per_token = decode_time / (output_len − 1)`
（`dataclass.py:60-62`），本 patch 的分母是 `num_decode_steps`。对齐关系分两种：

| 模式 | 本 patch 摊的步 | RTP 摊的步 | 是否可直接对比 |
|---|---|---|---|
| `--partial 0`（PD） | 第 1 步是 prefill（计入 `prefill_ms`），其后 `num_decode_steps` 步纯 decode，÷ `num_decode_steps` | 首 token 计入 `first_token_cost_time`，其后 `output_len − 1` 步，÷ `output_len − 1` | **可比**：两边都把首 token 排除在 decode 平均外，各自"decode 总时间 ÷ 自己测的 decode 步数"，per-step 均值同口径 |
| `--partial 1`（fake-KV） | `prefill_ms = 0`，把**全部** `num_decode_steps` 步摊平，÷ `num_decode_steps` | 仍记 `first_token_cost_time`（首个 decode 步），只摊第 2..N 步，÷ `N − 1` | **有一处 off-by-one**：本 patch 多摊了（可能偏冷的）首个 decode 步。从 vLLM patch 继承的已知残留，**刻意不改**以保持与 vLLM byte-对齐；步数大时影响很小 |

- **步数对齐提示**：`--partial 0` 下若想让两边测的 decode 步数也相等，需
  RTP `--decode_test_length = num_decode_steps + 1`（RTP 测 `decode_test_length`
  个 token、首个算 prefill）。这只影响"测了多少步"，不影响上表的 per-step 可比性。
  当前 `BUILD` 的 `grid_perf_test` 用 `--decode_test_length 20`、本 patch 默认
  `num_decode_steps 20`，两边各测 19 / 20 个 decode 步，per_token 仍可比。

---

## 1. 环境准备

与 vLLM patch 共用 `/opt/conda310`，editable 安装，**不重装 torch**。

```bash
cd /home/liusongyue.lsy/sglang/python
/opt/conda310/bin/pip install -e . --no-deps      # --no-deps：不要动已装的 torch / transformers / flashinfer
/opt/conda310/bin/python3 -c "import sglang; print(sglang.__file__)"
```

**每次运行前必须**（否则 `import sgl_kernel` 报
`libnvrtc.so.13: cannot open shared object file`）：

```bash
export LD_LIBRARY_PATH=/home/liusongyue.lsy/.local/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
```

本机（H20 + torch 2.11.0+cu130）已知问题与处理：

| 问题 | 处理 |
|---|---|
| `sgl_kernel` 导入报 `libnvrtc.so.13` 找不到 | 手动：上面的 `LD_LIBRARY_PATH`（torch 带的 cu13 运行时没被预加载） |
| TP>1 时 SGLang JIT 编 `custom_all_reduce.cuh` 报 `namespace "std" has no member "bit_cast"` | 手动：加 `--disable-custom-all-reduce`。nvcc 12.9 + gcc 10.2.1 编不了 C++20 `std::bit_cast`，需 GCC ≥ 11；换 NCCL all-reduce 不影响口径 |
| 日志刷 `User lacks permission to set NUMA affinity` | 无害。NUMA 绑定失败会回落，但绑定本身影响数字，所以**保留**生产行为不去关它 |

### 通用注意

- 在 `/tmp` 等目录下运行，**不要**在 sglang 源码目录下运行（`sys.path` 冲突）
- `--model-path` 必须是本地有效路径；路径不存在时 transformers 会把它当 HF repo id
  报 `HFValidationError: Repo id must be in the form ...`

---

## 2. 快速开始

```bash
cd /tmp
export LD_LIBRARY_PATH=/home/liusongyue.lsy/.local/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

# Prefill：测首 token 延迟（--partial 2）
/opt/conda310/bin/python3 -u -m sglang.srt.patches.batch_decode_scheduler.perf_test_runner \
  --model-path /data0/models/Qwen3-0.6B \
  --partial 2 \
  --batch-sizes 1,4 --seq-lens 128,512 \
  --num-iters 3 --num-warmup-iters 1 \
  --disable-cuda-graph --mem-fraction-static 0.5 --context-length 2048

# Decode：PD 路径（--partial 0，真实 prefill 铺 KV 后跑 N 步 decode，两个指标都出）；
# 严格对齐 RTP 的 decode-only 用 --partial 1（fake-KV，见 §5）
/opt/conda310/bin/python3 -u -m sglang.srt.patches.batch_decode_scheduler.perf_test_runner \
  --model-path /data0/models/Qwen3-0.6B \
  --partial 0 \
  --batch-sizes 1,4 --seq-lens 128,512 \
  --num-iters 3 --num-decode-steps 20 --num-warmup-iters 1 \
  --disable-cuda-graph --mem-fraction-static 0.5 --context-length 2048 \
  --output /tmp/sgl_result.csv
```

输出为表格；`--output` 同时写 CSV（列名与 vLLM patch 完全一致，可直接拼表）。

## 3. 常用参数

| 参数 | 说明 |
|---|---|
| `--partial {0,1,2}` | RTP 同语义模式开关：0=PD（真实 prefill+decode，两个指标都出），1=只 decode（fake-KV，**默认**，见 §5），2=只 prefill |
| `--batch-sizes` / `--seq-lens` | 逗号分隔的网格；decode 时 seq-lens 是 kv_len；DP 模式下 batch-sizes 是**每 rank** 口径（见 §4.2） |
| `--num-iters` / `--num-warmup-iters` | 每格测量轮数 / 预热轮数（预热走**完全相同**的代码路径，结果丢弃） |
| `--num-decode-steps` | decode 模式每轮的 decode 步数 |
| `--disable-cuda-graph` | ≈ vLLM `--enforce-eager`。要看清 kernel 语义名时必开 |
| `--mem-fraction-static` | ≈ vLLM `--gpu-memory-utilization` |
| `--max-total-tokens` / `--context-length` | KV 池上限 / 模型最大长度；`context-length` 需 ≥ `max(seq_lens) + num_decode_steps` |
| `--tp-size` / `--pp-size` / `--dp-size` / `--ep-size` | 并行配置（见 §4） |
| `--enable-dp-attention` | MoE 跨 DP EP（见 §4.2） |
| `--disable-custom-all-reduce` | 本机 TP>1 必加，见 §1 |
| `--per-step-timing` | 额外跑一轮逐步计时诊断（见 §8） |
| `--enable-overlap` | 用 overlap 调度器（SGLang 生产默认）。**默认关**，见 §8 |
| `--profile` / `--profile-dir` | 抓 chrome trace（见 §6）；默认目录 `./profile_output` |
| `--dtype` / `--load-format` / `--trust-remote-code` / `--attention-backend` | 原样透传 ServerArgs |
| `--base-gpu-id` / `--random-seed` / `--log-level` | 原样透传；GPU 被占用时用 `--base-gpu-id` 挪开 |
| `--rank-timeout` | 等 rank 进程的秒数，默认 7200 |
| `--output` | CSV 输出路径 |

**被 patch 钉死、不可通过 CLI 覆盖的 ServerArgs**（改了就不再是同口径）：

| 参数 | 值 | 原因 |
|---|---|---|
| `chunked_prefill_size` | `-1` | prefill 不许被切 |
| `disable_radix_cache` | `True` | 否则同 prompt 命中前缀缓存，prefill 数字假 |
| `max_running_requests` | `max(batch_sizes)`（dp-attention 下 ×`dp_size`） | 精确凑批上界 |
| `max_prefill_tokens` | `max_bs × max_seq_len`（fake-KV 时收敛到 64Ki，见 §5） | 一步吃下整批 prefill |
| `skip_tokenizer_init` | `True` | 直接喂 token id |
| `disable_overlap_schedule` | `not --enable-overlap` | 走 `event_loop_normal` 语义 |
| `enable_metrics` | `False` | 去掉与测量无关的开销 |

**显存要点**：prefill token 预算 = `max(max_bs × max(seq_lens), 16384)`，它决定
staging buffer 和激活预留，所以真正撑爆显存的是 `bs × max(seq_lens)`，大 batch +
长 seq 时减 batch 或 seq，而不是压 `--context-length`。

---

## 4. 并行模式

CLI 的 `--tp-size` 是 **「每个 DP 副本内的 TP」**（vLLM / RTP 语义）；
SGLang 的 `server_args.tp_size` 在 dp-attention 下是 **world size**。runner 做一次
显式换算并在启动时把两套值都打出来：

```
CLI tp=2 dp=2 ep=4 pp=1 dp_attention=True
  -> ServerArgs tp_size=4 dp_size=2 ep_size=4 (attn_tp_size=2)
```

| CLI | ServerArgs | 派生 |
|---|---|---|
| `--tp-size 8` | `tp_size=8, dp_size=1` | `attn_tp_size=8` |
| `--tp-size 2 --dp-size 2 --enable-dp-attention --ep-size 4` | `tp_size=4, dp_size=2, ep_size=4` | `attn_tp_size=2` |
| `--tp-size 1 --dp-size 4 --enable-dp-attention --ep-size 4` | `tp_size=4, dp_size=4, ep_size=4` | `attn_tp_size=1`（纯 DP attention） |
| `--tp-size 2 --dp-size 2`（无 dp-attention） | `tp_size=2, dp_size=2` | 2 个独立 TP=2 副本 |

`--ep-size` 不整除 world `tp_size` 时**直接报错**，不会让 `moe_ep_rank` 算歪。

### 4.1 TP / PP

`--tp-size N` 即可，runner 复用官方 `Engine._launch_scheduler_processes`（它接受注入
的 worker 函数），rank 数学一行不自己算。本机记得加 `--disable-custom-all-reduce`。

```bash
... --tp-size 2 --disable-custom-all-reduce
```

### 4.2 DP（`--dp-size N`）

`--batch-sizes` 是**每 DP rank** 的 batch size（与 RTP-LLM `GridRunner` 同语义，
总请求数 = bs × dp_size）。两种模式：

- **独立 DP**（不加 `--enable-dp-attention`）：N 个互相独立的 TP 组，每组自己的
  nccl port，`base_gpu_id` 逐组递增
- **跨 DP EP**（`--enable-dp-attention`）：DP 从 `tp_size` 里切出来，全 rank 共用一个
  TP 通信组（同一个 `nccl_port` / `instance_id`）；跳步同步由 SGLang 自己的
  `maybe_prepare_mlp_sync_batch` 完成，patch 不手写 barrier

⚠️ **DP 路径不能走官方启动器**：`dp_size > 1` 时 SGLang 会起
`DataParallelController`，它在 `launch_tensor_parallel_group` 之后无条件进入
`event_loop()`（`while True` 的 ZMQ recv，永不返回）。bench 不走 ZMQ 请求路径，
controller 会挂死、worker 结果没人收尾。所以 `_fork_dp_workers` 自己 fork 一层，
只复用官方的 `_calculate_rank_ranges` / `_compute_parallelism_ranks` /
`compute_dp_attention_world_info` / `PortArgs.init_new` 做 rank 数学。
**这是本 patch 唯一"抄"SGLang 内部逻辑的地方，SGLang 升级后是重点 review 点。**

```bash
# MoE 跨 DP EP：Qwen3-30B-A3B-FP8，TP=2 × DP=2 EP=4（4×H20）
/opt/conda310/bin/python3 -u -m sglang.srt.patches.batch_decode_scheduler.perf_test_runner \
  --model-path /data0/models/Qwen3-30B-A3B-Instruct-2507-FP8 \
  --partial 1 --batch-sizes 4 --seq-lens 128 \
  --num-iters 3 --num-decode-steps 10 --num-warmup-iters 1 \
  --tp-size 2 --dp-size 2 --enable-dp-attention --ep-size 4 \
  --disable-cuda-graph --mem-fraction-static 0.6 --context-length 2048 \
  --disable-custom-all-reduce
```

注意：

- 与 RTP 对比时 batch_size 可直接对齐（两边都是 per-rank 口径）
- 每个 DP 副本取其**最小 tp_rank** 作代表；表格 / CSV 只打印**最小 dp_rank** 那个
  代表（RTP 是全 rank 平均）。其余 DP 代表与它逐格比 primary 指标，差 >10% 打
  WARNING（不影响退出码）
- 各 rank 的结果通过临时目录里的 `rank_dp{d}_tp{t}.json` 回传（跑完即删，不是产物）；
  想看全 rank 数字就直接看 stdout —— 每个 rank 的进度行都带
  `[DP{d} TP{t}]` 前缀（单卡时无前缀），例如
  `[DP1 TP3]   decode bs=4 seq_len=128: per_token=105.47ms cost=1054.72ms`
- 任一 rank 报错 / 缺结果 / 退出码非零 → 整体 `sys.exit(1)`。某个 rank 死掉时会
  立刻拆掉其余 rank（不然它们会永远卡在 NCCL 集合通信里），不等 `--rank-timeout`

---

## 5. fake-KV decode（`--partial 1`，对齐 RTP，实验功能）

跳过 prefill forward：**KV block 照常分配、`req_to_token` 照常写**
（`prepare_for_extend` 真的跑了），只是不执行 forward，然后伪造一个
`GenerationBatchResult` 喂给 `process_batch_result`。KV 内容为全零，
对齐 RTP `setIsContextStream(false)`。适用于真实 prefill 会 OOM 或太慢的
大 BS × 长 kv_len 场景。

跳过 `run_batch` 会跳过它本该做的 5 件事，harness 全部补齐：

1. deferred mamba clear/COW（正常在 forward 里执行 —— 不补会留上一个请求的
   recurrent state，**静默出错**，见下）
2. `forward_iter` / `launch_ts`（`_record_step_counters` 会解引用）
3. `resolve_forward_inputs`（消费 pinned prefill staging）
4. `next_token_ids` 必须是 **device tensor**（prefill 结果路径无条件 `.tolist()`）
5. `_relay_forward_payload`（不补的话下一个 decode 步会从
   `future_map.output_tokens_buf` 里 gather 到没写过的 slot）

**注册期的一致性保证**（否则 decode batch 变成 ragged，per-token 数字不可比）：

- 注册期间临时抬高 `max_prefill_tokens`，让整批**一轮注册完**（fake-KV 不跑 forward，
  这个预算没有物理成本）
- 拿到 `None` 或非 extend batch 时**只计 stall、不执行**，绝不偷偷给已注册请求推进
  一个 decode token；连续 16 次就报错退出
- 正式测量前硬断言：`waiting_queue` 为空、`chunked_req is None`、请求数 == bs、
  prompt 长度全批一致、`len(output_ids) == 1`、`req.seqlen` 全批一致

> `get_next_batch_to_run` 只在**下一次**调用时才把 extend `last_batch` 合进
> `running_batch`，所以断言必须查 `running_batch ∪ last_batch` 的并集。

**GDN / Mamba 混合注意力模型**：`prepare_for_extend` 会消费掉
`req.mamba_needs_clear` 并把 slot 号挪到 `batch.mamba_clear_indices`，真正清零发生在
forward 里。harness 镜像了 `ModelRunner` 的清零逻辑（清零 = "空前缀"，与 KV 全零同
语义，物理自洽）；radix cache 已关，理论上不会走 COW 分支，真走到就报错不静默。

```bash
/opt/conda310/bin/python3 -u -m sglang.srt.patches.batch_decode_scheduler.perf_test_runner \
  --model-path /data0/models/Qwen3-0.6B --partial 1 \
  --batch-sizes 1,4 --seq-lens 128,512 \
  --num-iters 3 --num-decode-steps 20 --num-warmup-iters 1 \
  --disable-cuda-graph --mem-fraction-static 0.5 --context-length 2048
```

限制：attention 读全零 KV，数值与真实 prefill 不同 —— **仅用于 kernel 计时对齐**，
不用于精度相关对比。已在 Qwen3-0.6B 上验证 `--partial 1` 与 `--partial 0` 的
`per_token_ms` 差 < 0.3%（见 §9）。

---

## 6. Profiling

`--profile` 在每个网格点额外跑一个专用 profiling pass：**先一轮 warmup，再一轮采集**，
decode 模式下 **prefill 在窗口外**，所以 trace 里恰好 `--num-decode-steps` 步 decode，
per-step 均值是准的。每个 rank 各 dump 一份。

```bash
... --partial 0 --batch-sizes 4 --seq-lens 128 --num-decode-steps 10 \
    --profile --profile-dir /tmp/sgl_traces
```

文件名：`sglang_<mode>_bs<B>_seq<S>_steps<N>[_dp<D>][_tp<T>].json`（未 gzip 的
chrome trace）。步数后缀在前是因为共用的 timeline 分析器用 `re.search` 取步数。

> 这里**不走** `SchedulerProfilerManager`：它会 gzip trace 并发
> `torch.distributed.barrier`，而共用分析器读的是纯 JSON。

SGLang 自带的语义 scope 会一并出现在 trace 里，可直接用：
`scheduler.get_next_batch_to_run` / `scheduler.run_batch` /
`scheduler.process_batch_result`（`utils/nvtx_utils.py` 的 `scheduler_nvtx_method`）
和 `step[DECODE bs=..]` / `step[EXTEND bs=.. toks=..]`。

nsys 场景仍需 `--cuda-graph-trace=node`，否则 CUDA Graph 内 kernel 全不可见。

---

## 7. Timeline 分析与 RTP 对比

**直接用 vLLM 侧的分析器**（吃 chrome trace，与引擎无关，已实测能原样解析 SGLang
的 trace）：

```bash
# 单个 trace 分类分解（steps 从文件名的 steps<N> 自动解析，可 --steps 覆盖）
python3 -m vllm.patches.batch_decode_scheduler.perf_test_timeline \
  /tmp/sgl_traces/sglang_decode_bs4_seq128_steps10.json

# SGLang ↔ RTP 逐组件每步 diff。
# ⚠️ RTP trace 文件名不带 steps<N>，必须 --steps-b 指定其 decode 步数，否则回落为 1。
# RTP 的 profiling 轮只采 min(decode_test_length, 3) 步，所以 decode trace 通常
# --steps-b 3，prefill 是 1
python3 -m vllm.patches.batch_decode_scheduler.perf_test_timeline \
  --compare sglang.json rtp.json --labels SGLang RTP-LLM --steps 10 --steps-b 3
```

### RTP-LLM 侧的跑法

```bash
$BAZEL --output_user_root=~/.cache/bazel_cuda13_cache \
  test //rtp_llm/test/perf_test:grid_perf_test \
  --config=cuda13 --jobs=200 --test_timeout=3600 --test_output=all
```

参数在 `rtp_llm/test/perf_test/BUILD` 的 `grid_perf_test` target 里改
（`--batch_size / --input_len / --partial / --decode_test_length / --tp_size /
--dp_size`）。timeline 在 bazel testlogs 的 `test.outputs/timelines/` 下。

---

## 8. 已知限制 / 与 vLLM patch 的差异

1. **没有 MoE Fake Balance**（本轮出局）。SGLang 有
   `SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS`，但排布是「全局专家号递增」且
   `topk_weights = 1/k`，而 RTP 是「先轮询 EP rank，再轮询 rank 内 local expert」
   且 weight `= 1.0`（并禁 `routed_scaling_factor`）。**两者不可互换**，所以没启用。
   MoE decode 延迟存在轮间方差 —— 与开了 Fake Balance 的 vLLM 数字对比时必须注明。
2. **没有 scope 对齐**：SGLang 的 `scheduler.*` / `step[...]` span 与 RTP 的
   `executor.model_forward` / `sampler_forward` / `gather_model_input` /
   `dispatch_output` 没有逐一对位，阶段级 CPU 墙钟不可直接比。GPU 归因仍以
   kernel 分类表为准（穿透 CUDA graph，是跨引擎唯一可靠的层）。
3. **DP 汇总只输出一个代表 rank**（同 vLLM patch；RTP 是全 rank 平均）。
4. **fake-KV 是实验功能**：KV 内容全零，仅用于 kernel 计时对齐。
5. **Overlap 默认关**，与 SGLang 生产默认（开）有偏差。`--enable-overlap` 时
   `result_queue` 延后一步 `process_batch_result`，prefill / decode 的拆分变得近似，
   只有 `cost_ms` 有意义 —— runner 会在表头打 WARNING。
6. **`_fork_dp_workers` 是唯一"抄"SGLang 内部逻辑的地方**，见 §4.2。
7. **与 vLLM patch 的数字口径有一处刻意不一致**：SGLang 侧测量循环**没有**逐步
   `synchronize()`，vLLM 侧当前还有（其 P0 未修）。本 patch 的
   `--per-step-timing` 就是那条逐步同步路径，故意隔离在单独一节里，
   `sum_step(ms)` **不可**与 `cost(ms)` 相加或对比。
   ⚠️ 但**别据此假设 vLLM 的数字一定偏高**：在本机 Qwen3-0.6B / eager 上实测
   逐步同步的代价 ≈ 0（见 §9.4），因为这个规模下步间本来就没有可破坏的重叠。
   逐步同步真正伤到的是「CUDA graph + CPU 调度能藏进 GPU 时间」的配置。
   两边对比前请按 §9.4 的方法在**目标配置**上实测这个差值，不要照搬结论。
8. **`--partial 1` 的 `prefill(ms)` 列结构性恒为 `0.00`**：fake-KV 不测首 token，
   `prefill_ms` 不赋值。decode 表照打 `prefill(ms)` 列，所以那格永远是 `0.00`
   （§9.3 的样例即如此），拼三方表时**别把它当"prefill 为 0"误读**，那一列在
   `--partial 1` 下无意义。RTP decode 模式仍会记 `first_token_cost_time`
   （推断 ≈ 首个 decode 步；此点仅从 Python 侧佐证，未读 RTP C++ 计时代码）。
9. **RTP prefill 网格与本 patch 覆盖面不同**（对比 `--partial 2` 时注意）：
   - RTP prefill 强制 `BS=1`（`grid_runner.py:34` `[1] if not is_decode`、
     `perf_config.py:186-187`），所以 `--partial 2 --batch-sizes 1,4,16` 里
     **bs=4/16 没有 RTP 对标行**，只有 bs=1 可三方拼表。
   - RTP decode 对放不下 KV 的格子是**静默跳过**
     （`batch_decode_test.py:141` `filter_bs_by_kvcache`）；本 patch 是
     `assert_batch` **硬断言报错**（更安全，但同一份 BUILD 参数下两边跑出的网格
     可能不是同一批格子）。

---

## 9. 已验证的参考结果

环境：Qwen3-0.6B / BF16 / 单张 H20 / `--disable-cuda-graph`（eager）/
`--mem-fraction-static 0.5` / `--num-iters 3` / `--num-decode-steps 10`。

### 9.1 `--partial` 三种模式一致性

| BS | SeqLen | prefill(ms) `--partial 2` | prefill(ms) `--partial 0` | per_token(ms) `--partial 1` | per_token(ms) `--partial 0` |
|---|---|---|---|---|---|
| 1 | 128 | 13.18 | 13.24 | 11.80 | 11.77 |
| 1 | 512 | 12.98 | 12.61 | 11.42 | 11.40 |
| 4 | 128 | 13.20 | 12.81 | 12.23 | 12.19 |
| 4 | 512 | 19.75 | 19.63 | 12.02 | 11.98 |

fake-KV（`--partial 1`）与真实 prefill（`--partial 0`）的 `per_token_ms` 差 < 0.3%，
prefill-only（`--partial 2`）与 PD 的 `prefill_ms` 同量级 —— 三条路径口径自洽。

### 9.2 交叉验证：与 SGLang 自带 `one_batch` 对齐

`one_batch` 绕过 Scheduler 直驱 ModelRunner，差值即**调度器每步开销**：

```bash
/opt/conda310/bin/python3 -m sglang.benchmark.one_batch \
  --model-path /data0/models/Qwen3-0.6B \
  --batch-size 4 --input-len 128 --output-len 21 \
  --disable-cuda-graph --mem-fraction-static 0.5 --disable-radix-cache
```

| 指标 | `one_batch` | 本 patch (bs4/seq128) | 差值 |
|---|---|---|---|
| decode / token | 11.89 ms | 12.19 ms | **+0.30 ms/step** |
| prefill | 12.56 ms | 12.81 ms | +0.25 ms |

0.30 ms/step 落在预期的 <1 ms 内，说明驱动路径没有多余同步。

### 9.3 并行

Qwen3-0.6B，decode per_token（`--partial 1`，bs4 / kv_len 128 / 10 步）。
每行 stdout 带 `[DP.. TP..]` 前缀，所以下面的 rank 归属是**日志直读**的，不是推测：

| 配置 | 各 rank 结果 | spread |
|---|---|---|
| `--tp-size 2 --disable-custom-all-reduce` | `[TP0]` 20.57 / `[TP1]` 20.61 ms | 0.2% |
| `--dp-size 2`（独立 DP） | `[DP0]` 12.38 / `[DP1]` 12.61 ms | 1.9% |
| `--tp-size 1 --dp-size 2 --enable-dp-attention` | `[DP0 TP0]` 25.83 / `[DP1 TP1]` 25.82 ms | 0.04% |

三组都无 spread WARNING（阈值 10%），进程全部正常退出，汇总表的
`per_token` 与最小 dp_rank / tp_rank 的那行**逐位一致**。

MoE 跨 DP EP（§4.2 的命令，Qwen3-30B-A3B-Instruct-2507-FP8 / 4×H20 /
TP=2 × DP=2 / EP=4 / eager / bs4 / kv_len 128 / 10 步）：

```
[DP1 TP2]   decode bs=4 seq_len=128: per_token=111.98ms cost=1119.83ms
[DP0 TP0]   decode bs=4 seq_len=128: per_token=111.99ms cost=1119.86ms
[DP0 TP1]   decode bs=4 seq_len=128: per_token=112.02ms cost=1120.20ms
[DP1 TP3]   decode bs=4 seq_len=128: per_token=112.04ms cost=1120.44ms
=== DP=2 (showing dp rank 0; batch_size is PER-DP-RANK, RTP-aligned — total = bs x 2) ===
decode      4     128   1119.86         0.00         111.99    112.46       3
```

4 个 rank 全部出数、极差 0.05%（远低于 10% 阈值，无 WARNING），
`ep_size=4` 校验通过，进程全部正常退出（`EXIT=0`）。

> ⚠️ **这组数只用于验证 EP 链路通 + rank 间一致，不要比绝对值。** 两点原因：
>
> 1. 绝对值偏高是本机环境：eager（`--disable-cuda-graph`）+ H20 上没有调优过的
>    FP8 block-quant / fused-MoE Triton config（日志刷 `Using default MoE kernel
>    config. Performance might be sub-optimal!`）。
> 2. **次间方差大**：同一条命令跑两次得到 105.4 ms 和 112.0 ms（差 6%），而
>    rank 间只差 0.05%。合成 prompt 的路由是确定的，所以这 6% 来自环境
>    （另一进程占着别的卡 / Triton 默认 config 的调度抖动），不是 rank 不均。
>    没做 Fake Balance（§8 第 1 条）时 MoE 的数字就是这个稳定性，跨引擎比对
>    MoE 前先把这一项解决掉。

### 9.4 逐步同步的代价（`--per-step-timing`）实测 ≈ 0

同一命令加 `--per-step-timing`（bs4 / seq128 / 10 步）：

```
=== Decode ===                cost=135.50   prefill=13.30   per_token=12.18
=== Per-step diagnostics ===  prefill_step=12.91  decode_step=12.14  sum_step=134.26
```

`sum_step` (134.26) 甚至比 `cost` (135.50) 略低 —— 也就是说**在这个配置下逐步
`synchronize()` 几乎不花钱**，计划里预期的 `sum_step > cost` 没有出现。

原因见 §9.5 的 kernel 分解：一个 decode 步 GPU 只忙 1.41 ms，其余 ~10.8 ms 是
eager 模式下的 kernel launch / Python 开销，本来就是串行的，步间没有可被同步
破坏的重叠。**结论**：不逐步同步在架构上仍然是对的（开 CUDA graph、或模型大到
GPU 时间能盖住 CPU 时间时它才有意义），但**在本机这个 shape 上它不是一个可观测
的改进** —— 所以别拿它去解释与 vLLM 的数字差异（§8 第 7 条）。

### 9.5 Timeline

vLLM 的 `perf_test_timeline.py` **未经修改**即可解析本 patch 的 trace，步数从
文件名自动解析（`context={'batch_size': 4, 'seq_len': 128, 'num_steps': 10}`）：

```
Category                 total(us)  per_step(us)  %kernel   count
Dense GEMM                 10594.6       1059.46    75.0%    1700
Other                       1110.2        111.02     7.9%     890
Elementwise/Copy            1102.6        110.26     7.8%     692
Attention                    907.1         90.71     6.4%     570
Activation                   350.6         35.06     2.5%     280
TOTAL GPU                  14130.0       1413.00
```

SGLang 自带的语义 scope 也都在（CPU 墙钟，profiler 开销已被计入，别当纯 GPU 读）：
`scheduler.run_batch` 16742 us/step、`step[DECODE bs=4]` 16386、
`scheduler.get_next_batch_to_run` 615、`scheduler.process_batch_result` 242。

**关键读数**：GPU 每步只忙 **1.41 ms**，而实测 per_token 是 12.18 ms —— 这个
shape（0.6B / bs4 / eager）是彻底的 **launch-bound**。`one_batch` 也是 11.89 ms
（§9.2），所以那 10.8 ms 是 SGLang eager 模式本身的开销，不是本 patch 的。
`Other` 占 7.9% 会触发分析器的 WARNING，属正常提示。

### 9.6 待补

**Qwen3-8B 三方同表**（RTP-LLM decode bs1/seq128 = 13.78 ms、
vLLM = 13.73 ms、vLLM prefill = 14.53 ms，见 `vllm/patches/RUN_GUIDE.md` §9）
尚未跑 —— 本机没有 Qwen3-8B。下载后按 §2 的 `--partial 1`、`bs ∈ {1,4,16}`、
`kv_len ∈ {128,1024}`、decode 20 步跑一遍，落在同一区间即视为三方口径对齐。

---

## 10. 单元测试

```bash
cd /tmp
export LD_LIBRARY_PATH=/home/liusongyue.lsy/.local/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
/opt/conda310/bin/python3 -m unittest discover \
  -s /home/liusongyue.lsy/sglang/test/registered/perf \
  -p "test_perf_test_runner.py" -v
```

纯 CPU、不加载模型（Scheduler 被 stub 掉、`ServerArgs` 只捕获 kwargs），已注册进
`base-a-test-cpu` suite。覆盖：trimmed-mean 聚合、ServerArgs 钉死项、TP/DP/EP 换算、
**每轮恰好 3 次同步**的不变量、批被拆 / prefill 被切 / decode 中途缩批的报错路径、
fake-KV 注册一致性断言、CSV 列布局、DP 失败归因。
