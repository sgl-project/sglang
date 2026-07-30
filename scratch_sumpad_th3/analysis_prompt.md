# Task: explain contradictory e2e numbers in a DP-attention padding experiment

You are analyzing a completed measurement matrix in the sglang repo, branch
`tom/adhoc-sum-pad-exp-th3`. Work read-only. Be skeptical: the goal is to decide which
numbers are trustworthy and which are artifacts, not to rationalize a story.

## Setup

- Qwen3-8B, 8 GPUs, `--tp 8 --dp 8 --enable-dp-attention --disable-radix-cache`,
  `--mem-fraction-static 0.85`, default chunked prefill (8192 global → 1024 per DP rank).
- Two graph settings: `--cuda-graph-backend-prefill breakable` vs `disabled`.
- Four source variants, all selected by env var `SGLANG_DBG_DP_PAD_MODE`
  (see `python/sglang/srt/layers/dp_attention.py::DpPaddingMode.get_dp_padding_mode`
  and `python/sglang/srt/model_executor/forward_batch_info.py` around line 1240):
    - `default` — today's main.
    - `no10414` — skips the `is_extend_in_batch and dp_size > 1 → SUM_LEN` branch that
      PR #10414 added, falling back to the older communication-cost heuristic.
    - `sum` — force SUM_LEN, `max` — force MAX_LEN (both applied after the prefill-CUDA-graph
      force, so they win).
- Two workloads, both via `python -m sglang.bench_one_batch_server` ("bob") and
  `python -m sglang.bench_serving`:
    - uniform: bs=64, isl=2048, osl=1, concurrency 64 (all 8 DP ranks roughly even).
    - skew: bs=1, isl=16384, osl=1, concurrency 1 (one rank does everything, 7 idle).

## The measured table (input throughput tok/s, one run per cell)

| variant | graph | bob_uniform | bob_skew | serve_uniform | serve_uniform TTFT(ms) | serve_skew | serve_skew TTFT(ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| default | breakable | 177567 | 20156 | 160367 | 676.45 | 19796 | 826.33 |
| default | disabled | 150340 | 31275 | 153170 | 754.76 | 30830 | 530.13 |
| max | breakable | 166250 | 20220 | 174030 | 663.56 | 19786 | 826.70 |
| max | disabled | 169213 | 20367 | 180002 | 647.57 | 20048 | 815.93 |
| no10414 | breakable | 166248 | 20177 | 173488 | 665.80 | 19779 | 826.96 |
| no10414 | disabled | 170784 | 32516 | 176933 | 649.56 | 30920 | 528.59 |
| sum | breakable | 177225 | 32353 | 173242 | 670.10 | 30693 | 532.53 |
| sum | disabled | 150884 | 31660 | 153933 | 752.42 | 30999 | 527.12 |

## Instrumented per-forward evidence (same 8 configs, separate runs)

Each server forward prints `[DPPAD] ... mode= is_extend_in_batch= bcg_ok=
forced_max_by_prefill_cg= raw= final= real_local= padded_local= buffer=`; the breakable
prefill graph prints `[PCG] raw_num_tokens= bucket= ...` on replay; a JIT CUDA kernel
prints `[KPROBE] ... graph_rows= real_local_tokens= padded_local_tokens= dp_pad_mode=
dp_buffer_len= used_prefill_graph=` from inside the graph. `dp_pad_mode`: 1=MAX_LEN,
2=SUM_LEN, 0=capture-time (ignore those lines).

| config | extend-step modes | forced_max_by_prefill_cg | local pad waste | prefill graph replays |
| --- | --- | --- | --- | --- |
| default + breakable | MAX 344 / SUM 16 | 344 | 59.31% | 336 (bucket 1024×328, 8×8) |
| default + disabled | SUM 368 | 0 | 0.00% | 0 |
| max + breakable | MAX 360 | 344 | 60.77% | 336 |
| no10414 + breakable | MAX 368 / SUM 8 | 360 | 62.18% | 336 |
| no10414 + disabled | MAX 208 / SUM 168 | 0 | 4.77% | 0 |
| sum + breakable | SUM 352 | 344 | 0.00% | 192 |
| sum + disabled | SUM 352 | 0 | 0.00% | 0 |
| max + disabled | MAX 360 | 0 | ~61% | 0 |

## The contradictions you must resolve

1. **bob and bench_serving rank the uniform column differently.** bob says
   `default+breakable` is the fastest uniform cell (177567); bench_serving says it is the
   *slowest* of the non-SUM-eager cells (160367) while `max+disabled` is fastest (180002).
2. **`default+breakable` vs `max+breakable` should be nearly identical on uniform** — the
   evidence says both force MAX_LEN on essentially every extend step and both replay the
   prefill graph 336 times — yet bob reports 177567 vs 166250 (7%) and serving reports
   160367 vs 174030 (8%), *in opposite directions*.
3. **`sum+breakable` uniform (177225) ≈ `default+breakable` uniform (177567)**, but
   `sum+disabled` (150884) ≈ `default+disabled` (150340), 17% lower. Is the uniform gain
   really from the prefill CUDA graph, given `sum+breakable` only replays 192 times vs 336?
4. **On uniform, SUM_LEN eager (150k) looks 13% slower than MAX_LEN eager (170k)** — the
   opposite sign of the skew effect. Is that real, or an artifact (e.g. buffer sizing,
   allocator, allgatherv vs allgather, or the debug instrumentation itself)?
5. **Why does `default+breakable` still show 16 SUM_LEN extend steps and `no10414+breakable`
   8**, if the prefill-graph force is supposed to make every extend step MAX_LEN?
6. **Known instrumentation caveat**: the `sum`/`max` overrides are gated only on
   `sync_group_size > 1`, so eager decode/idle steps are forced too. osl=1 makes decode
   almost absent, but quantify whether this can explain any of the above.

## Where the raw data is

- `/Users/tom/domains/agent-default/artifacts/sglang/2026-07-30-dp-padding-th3/bench/<variant>-<graph>/`
  — `server.log`, `bob_uniform.log`, `bob_skew.log`, `serving_uniform.log`, `serving_skew.log`.
- `.../2026-07-30-dp-padding-th3/evidence/<variant>-<graph>/server.log` — the instrumented runs
  (`[DPPAD]` / `[PCG]` / `[KPROBE]` lines).
- Harness and parsers: `scratch_sumpad_th3/` in the repo (`run_one.sh`, `run_matrix.sh`,
  `summarize_logs.py`, `collect_bench.py`, `shape_probe.py`).
- Relevant source: `python/sglang/srt/layers/dp_attention.py`,
  `python/sglang/srt/model_executor/forward_batch_info.py`,
  `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`.

## What to deliver

A concise report with, for each of the 6 contradictions: the most likely explanation,
the specific evidence in the logs or source that supports it, and a verdict of
`REAL EFFECT` / `MEASUREMENT ARTIFACT` / `CANNOT TELL FROM THIS DATA`. Then:

- Which cells of the table should be trusted and which discarded.
- What the honest bottom-line answer is to "does the breakable-prefill-CUDA-graph MAX_LEN
  force cost real throughput, and on which workloads".
- Which additional measurement would settle whatever remains unresolved (a repeat bench
  matrix for noise is already running, so do not just say "repeat it" — say what to look at).

Do not fabricate numbers. If you cannot verify something from the logs or source, say so.
