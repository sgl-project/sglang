# Research brief: is Lianmin right that SUM padding duplicates attention projection?

You are auditing a performance investigation. Be adversarial and neutral. Both the claim below and
my measurements may be wrong. Do not fabricate numbers. Cite `file:line` for every code claim.

## The claim to adjudicate (verbatim, from Lianmin Zheng / merrymercy on Slack, 2026-07-29)

> 这个可以删掉吗? (referring to https://github.com/sgl-project/sglang/pull/10414/changes)
> 这个对 prefill 性能影响挺严重的。prefill dp attention 都去跑 sum padding，attention projection 都被 duplicate 了
> ... 得把这个 condition 直接删了吧

So his position is:

1. PR #10414 makes prefill DP attention run **SUM padding**.
2. Under SUM padding, **attention projection gets duplicated**.
3. Therefore the condition should be deleted outright.

He is the author of the DP-attention code and of PR #30898 (which added the breakable-prefill-CUDA-graph
MAX_LEN force). Treat his claim as a strong prior, but verify it against source and data.

## What the code does (my current understanding — verify, do not trust)

- `python/sglang/srt/layers/dp_attention.py:84-122` — `DpPaddingMode` selection. PR #10414 added
  `is_extend_in_batch and dp_size > 1 -> SUM_LEN`, replacing an older comm-cost heuristic
  (`sum_len * 2 >= max_len * dp_size -> MAX_LEN`).
- `DpPaddingMode` is an `IntEnum` with `auto()`: MAX_LEN=1, SUM_LEN=2. **0 is not a valid value.**
- `python/sglang/srt/model_executor/forward_batch_info.py:1249-1261` — the prefill-CUDA-graph force:
  when `can_run_dp_breakable_cuda_graph and is_extend_in_batch and prefill_cg.bs and
  max(global_num_tokens) <= max(prefill_cg.bs)`, mode is unconditionally set to MAX_LEN.
- `forward_batch_info.py:1337-1360` — under MAX_LEN, idle DP ranks fabricate a full max-length dummy
  extend.
- `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:1177` — prefill graphs are
  **captured** with MAX_LEN, so a replayed graph executes MAX collectives no matter what mode the
  live batch is labeled with.
- Collectives: MAX_LEN -> `all_gather_into_tensor` + `reduce_scatter`; SUM_LEN -> `fill_(0)` +
  scatter-copy + full `all_reduce`.
- `default_prefill_backend()` in `python/sglang/srt/model_executor/cuda_graph_config.py` returns
  `BREAKABLE` on CUDA, i.e. **breakable prefill CG is the default**.

## THE CENTRAL QUESTION (answer this first, from source, before touching any data)

Lianmin says "attention projection 都被 duplicate 了". I need to know exactly what is duplicated and
whether the padding mode controls it. Concretely:

- Which tensors does each DP rank actually run the **attention projections** (qkv_proj / o_proj, or
  for MLA: q_a/q_b/kv_a/kv_b/o_proj) over — its own DP-local tokens, or the **gathered global**
  buffer?
- Does `dp_padding_mode` change *which rows* a rank computes, or only *how big the gathered buffer
  is* and *which collective is used*?
- Specifically evaluate this hypothesis, which would make Lianmin literally right and me wrong:
  **under SUM_LEN every rank computes the projections over the entire global (sum) buffer, i.e. 8×
  duplicated work at dp_size=8, whereas under MAX_LEN the reduce_scatter arrangement means each rank
  only computes its own slice.** Is that true? If it is false, say so with `file:line` and explain
  what he might actually be referring to.
- Read `layer_communicator.py` / `LayerScatterModes` / the `dp_gather`/`dp_scatter` call sites and
  every consumer of `dp_padding_mode`. Do not stop at `dp_attention.py`.
- Does the answer depend on model architecture (dense GQA like Qwen3-8B vs MLA like DeepSeek) or on
  `attn_tp_size` vs `tp_size`? Lianmin's production concern is likely DeepSeek-class MLA on large TP;
  my experiment was Qwen3-8B tp8/dp8. If the mechanism only bites on MLA or on
  `attn_tp_size < tp_size`, that alone reconciles everything — check it explicitly.

## My th3 measurements (verify against raw data; the report may be wrong)

Setup: latest main, branch `tom/adhoc-sum-pad-exp-th3`, Qwen3-8B, `--tp 8 --dp 8
--enable-dp-attention`, 8×H200, chunked prefill 8192 global (=1024/rank at dp8).
8 configs = 4 source variants (default / revert-#10414 / force-SUM / force-MAX)
× 2 graph settings (breakable / disabled). Two workloads: uniform (bs=64 isl=2048 osl=1) and
skew (bs=1 isl=16384 osl=1, i.e. 1 active rank + 7 idle per chunk).

My conclusions, in the order I trust them least:

1. **Skew: MAX_LEN is catastrophically worse.** GEMM summed over 8 ranks splits cleanly 4-vs-4:
   MAX-behaving configs 2188.59 / 2189.66 / 2190.24 / 2191.29 ms vs SUM-behaving 334.82 / 335.68 /
   336.15 / 336.27 ms — ratio 6.52×, <0.5% spread within each group. Per-rank: MAX -> all 8 ranks
   ~273 ms GEMM; SUM -> active rank 98 ms, other 7 ranks ~34 ms. 273/34 = 8.0, matching buffer 8192
   vs 1024. Crucially `max+disabled` (MAX eager, NO graph) sits in the MAX group, so the waste is the
   MAX_LEN mode, not the CUDA graph. **This directly contradicts "MAX is better".**
2. **Uniform: SUM is ~16% slower than MAX in eager.** Median TTFT: MAX eager 695-701 ms vs SUM eager
   808-816 ms, reproduced across two runs and four cells. I attributed this to *communication*:
   AllReduce 1664-1836 ms (MAX group) vs 3394-3710 ms (SUM group) = 1.85-2.23× ~= 2×; AllGather
   present only in MAX behavior (934-1013 ms); `FillFunctor` 112 ms only in SUM.
   **This agrees with Lianmin's direction but not his mechanism.**
3. **Reverting #10414 is a no-op in the default config**, because breakable is the CUDA default and
   its MAX force overrides the mode afterwards: `no10414+breakable` == `default+breakable` (uniform
   median 716.3 vs 715.5 ms, skew 19779 vs 19796, TTFT 827.0 vs 826.3). Only with
   `--cuda-graph-backend-prefill disabled` does the revert help, and there it wins both workloads.
4. Evidence split by idle-rank presence: with all ranks active, padding waste is 0.0-2.2% in every
   config (so MAX padding costs nothing on even batches); with any rank idle, waste is
   442.87% / 533.35% / 656.98% / 700.00% for MAX vs 0.00% for SUM.

**Your job on the data side:** the thing that would most cleanly settle the central question is the
**uniform GEMM totals per mode**, which my report did NOT tabulate (I only tabulated collectives for
uniform, and GEMM only for skew). If SUM duplicated projections 8× on uniform, uniform GEMM should
be ~8× larger for SUM than MAX and e2e would be far worse than 16%. Compute it. If instead uniform
GEMM is equal across modes, Lianmin's "duplicate" cannot be about uniform row counts — then say what
it IS about.

## Raw data and tools (all local, already downloaded, 128/128 traces, gzip-verified)

- GPU-only traces (8 configs × 16 traces = 2 workloads × 8 ranks each):
  `/Users/tom/domains/human/others/artifacts/sglang/2026-07-30-dp-padding-th3/profile/<config>/`
  where `<config>` in {default,no10414,sum,max}-{breakable,disabled}.
  `bs-64-il-2048-*` = uniform, `bs-1-il-16384-*` = skew. `-TP-N-DP-N` is the rank.
- e2e bench logs: `.../2026-07-30-dp-padding-th3/bench/` and `bench-repeat/` (independent repeat).
- Per-forward instrumentation logs (`[DPPAD]` / `[PCG]` / `[KPROBE]` lines):
  `.../2026-07-30-dp-padding-th3/evidence/<config>/server.log`.
- Aggregator: `scratch_sumpad_th3/analyze_traces.py` (typer;
  `uv run python scratch_sumpad_th3/analyze_traces.py <trace_dir>`). It buckets kernels into
  comm/attn/gemm/memcpy/elementwise/other, uses interval-union for busy time, and prints per-rank
  rows plus totals. **Check its bucket regexes before trusting them** — e.g. `sm\d+_` and `copy` are
  loose, and a misfiled kernel would move GEMM vs elementwise totals.
- Log summarizer: `scratch_sumpad_th3/summarize_logs.py`.
- My report: `/Users/tom/domains/human/context/notes/projects/sglang/2026-07-29-dp-prefill-sum-padding/agent-drafts/2026-07-30-report-th3-dp-padding.md`

Known caveats in my own data, which you should weigh: the skew SUM active rank is only ~53% GPU busy
(CPU-bound shape at bs=1), so the measured MAX penalty there is a lower bound; only the two extremes
of skew were sampled, so the crossover is unmeasured; `sum+breakable` is not a valid "SUM + graph"
datapoint because the graph body is MAX-captured.

## Deliverables

1. A verdict on the central question, with `file:line` evidence: is there a code path where SUM_LEN
   duplicates attention-projection compute? `REAL` / `NOT REAL` / `REAL ONLY UNDER <conditions>`.
2. If NOT REAL as stated, your best reconstruction of what Lianmin actually observed — he is the
   author and had a concrete production symptom. Candidates to test, not an exhaustive list: MLA vs
   dense, `attn_tp_size < tp_size`, very large dp_size, the SUM all_reduce being over
   `global_num_tokens` rather than local, moe/EP interaction, or the pre-#10414 heuristic having been
   load-adaptive in a way the current code is not.
3. Uniform GEMM totals per mode from the traces, with the numbers.
4. Which of my 4 conclusions survive, which die, and why. Name the ones you cannot decide from this
   data.
5. Should the condition be deleted, as he asks? If deleting it alone is wrong (because of the skew
   MAX blowup and/or because breakable's force makes it moot), state the smallest correct fix.
6. **A completeness audit of the th3 task spec below.** For every bullet, rule
   `DONE` / `PARTIAL` / `MISSING`, citing the artifact path or `file:line` that proves it. Be strict:
   "I ran something adjacent" is PARTIAL, not DONE. List the MISSING/PARTIAL items in the order I
   should fix them.

## th3 task spec, verbatim (translated bullets kept as written by the user)

Source: `AC8742 20260729 lianmin说的prefill sum pad问题` -> section `## (20260730 13:47) th3`.

- 前面的实验仅供参考，可能有坑！比如好像他用的代码都是错的
- 使用源码
    - 必须从今天最新的 main 开始做魔改，禁止用 stable branch 之类其他的
    - 在这个分支 `tom/adhoc-sum-pad-exp-th3` 做你需要的魔改
- 测量这几个场景
    - a. 打开 breakable cuda graph + **同时必须保证那个 batch 真的激活了它**
    - b. 关闭 breakable cuda graph
- 配合这几个源码
    - a. 默认模式
    - b. revert 10414 的情形
    - c. 强制都用 sum（如果逻辑上能跑）
    - d. 强制都用 max（如果逻辑上能跑）
- 均匀程度的压力
    - a. 各个 rank 尽可能均匀（e.g. `bench_one_batch_server isl=2k osl=1 bs=64 conc=64`）
    - b. 各个 rank 尽可能不均匀（e.g. `bench_one_batch_server isl=2k osl=1 bs=16 conc=1`）
- 测量方式
    - a. `bench_one_batch_server isl=2k osl=1 bs=64` 之类的方式（数字自己定）
    - b. profile，但**只开 gpu 不开 cpu**（避免 cpu 变慢好多倍导致 profile 很假）
- 测量信息
    - a. 端到端数值
    - b. profile 证据（理解端到端耗时的成因、理解是不是测错了）
    - c. 额外还需要 print 出每次实际决策使用的 dp padding mode，例如在 model runner forward 时 print
    - d. 额外还要一个小 jit cuda kernel，去朴素地 print 一下自己收到的 shape 之类的信息，从而在
      prefill cuda graph 之类的时候仍然能准确地展示出实际遇到的情形
- 要求
    - 所有实验都必须有原始结果（日志、profile 之类）放到 artifacts 目录让我审计
    - 所有源码（包括 adhoc 脚本）都必须在指定的 sgl branch 上改动，commit，push，禁止放到别处
    - 最终给一个 md 报告所有实验结果
    - 客观中立，新实验和旧实验都可能是错的
- 坑
    - 如果 prefill 一次跑的 token 数太少，会 bound 在 cpu 上而不是 gpu 上，则实际只在测 cpu 耗时，
      毫无意义不得相信。如果 profile 中存在一个 rank gpu 很闲（一般只有 1 个 rank 会表现出来），
      要小心是 bound 在 cpu 上
    - 可能用 qwen3 8b 这种中小模型 + 8gpu

Note on one spec bullet I deviated from: the spec's skew example is `bs=16 conc=1`; I used
`bs=1 isl=16384`. Judge whether that substitution is defensible or whether the specified shape must
also be run.
