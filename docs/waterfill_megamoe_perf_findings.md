# Waterfill + Mega-MoE Perf Findings

This note summarizes the current B200 evidence for why DeepEP Waterfill does
not show a stable end-to-end serving gain over regular shared-expert fusion in
the DSV4 Mega-MoE path.

## Setup

Common server setup:

- Model: `/home/scratch.xutingz_wwfo_2/model/DeepSeek-V4-Flash-Base`
- Repo used by jobs: `/home/scratch.xutingz_wwfo_2/sglang-wf-current`
- Backend: `--moe-a2a-backend megamoe`
- TP/EP: `tp=2`, Mega-MoE EP size follows TP size.
- Compared cases:
  - `fused`: `--enforce-shared-experts-fusion`
  - `fused_waterfill`: `--enforce-shared-experts-fusion --enable-deepep-waterfill`
- Timing instrumentation:
  - `SGLANG_MEGA_MOE_LOG_TIMING_INTERVAL`
  - `SGLANG_WATERFILL_LOG_STATS_INTERVAL`

The GitHub document referenced by the user,
`xutizhou/moe_load_balancer/docs/deepep-waterfill-lplb-blog-draft.md`, returned
404 for both the GitHub UI URL and raw URL during this investigation, so the
exact dataset settings from that document could not be inspected directly.

## Dataset Configuration Used

SGLang `bench_serving` has no built-in `mmlu` dataset selector. It supports a
`custom` dataset, implemented in `python/sglang/benchmark/datasets/custom.py`.
That loader expects JSONL rows with at least two conversation turns:

- first turn is the request prompt
- second turn is the completion used only to infer output length unless
  `--sharegpt-output-len` overrides it

For MMLU serving perf, `scripts/b200_megamoe_cap_eval_inner.sh` now supports:

```bash
PERF_DATASET_NAME=mmlu_custom
PERF_MMLU_NUM_EXAMPLES=2000
PERF_MMLU_PROMPT_MODE=raw
PERF_SHAREGPT_OUTPUT_LEN=4
```

This builds a JSONL file like:

```json
{"conversations":[{"from":"human","value":"<MMLU few-shot prompt>"},{"from":"gpt","value":" The answer is A."}]}
```

Then it runs:

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --dataset-name custom \
  --dataset-path "$PERF_DATASET_PATH" \
  --sharegpt-output-len "$PERF_SHAREGPT_OUTPUT_LEN" \
  --sharegpt-context-len "$MAX_PREFILL_TOKENS"
```

## Random 8K Prefill Result

Job: `2639863`

Workload:

- `PERF_DATASET_NAME=random`
- `PERF_NUM_PROMPTS=128`
- `PERF_INPUT_LEN=8192`
- `PERF_OUTPUT_LEN=1`
- `PERF_MEASURE_REPEATS=5`

End-to-end throughput:

| Case | Mean input tok/s | Mean req/s | Mean TTFT ms |
| --- | ---: | ---: | ---: |
| `fused` | 36061.30 | 9.1788 | 1662.77 |
| `fused_waterfill` | 36252.09 | 9.2274 | 1653.10 |

Speedup: `+0.53%`.

Mega-MoE timing, sampled layer calls:

| Metric | `fused` | `fused_waterfill` |
| --- | ---: | ---: |
| count ratio mean | 1.3987 | 1.0266 |
| count abs diff mean | 8291.85 | 323.60 |
| `pre_dispatch_to_fp8_fp4_ms` mean | 1.533350 | 1.504560 |

Waterfill clearly balances load, but the compute segment saves only
`0.02879 ms` per sampled MoE layer call, about `1.9%` of that segment. On a
43-layer DSV4 model this is roughly `1.2 ms` per large prefill batch, versus
about `1650 ms` request latency in this benchmark.

## MMLU Serving Workload

Jobs:

- `2640401`: `fused -> fused_waterfill`, 3 repeats
- `2640645`: `fused -> fused_waterfill`, 5 repeats
- `2653100`: `fused_waterfill -> fused`, 5 repeats

Workload:

- `PERF_DATASET_NAME=mmlu_custom`
- `PERF_MMLU_NUM_EXAMPLES=2000`
- `PERF_SHAREGPT_OUTPUT_LEN=4`
- `PERF_CONCURRENCY=64`
- `PERF_MEASURE_REPEATS=3` or `5`

### Individual Jobs

| Job | Order | `fused` mean input tok/s | `waterfill` mean input tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| `2640401` | fused then waterfill | 20026.93 | 21320.29 | +6.46% |
| `2640645` | fused then waterfill | 22152.89 | 21498.22 | -2.96% |
| `2653100` | waterfill then fused | 21527.82 | 21324.81 | -0.94% |

The single-job end-to-end result is not stable. The first MMLU job suggests a
large gain, but the next two do not reproduce it.

### Combined MMLU E2E Result

Across all three MMLU jobs:

| Aggregate | `fused` mean input tok/s | `waterfill` mean input tok/s | Speedup |
| --- | ---: | ---: | ---: |
| all runs | 21421.87 | 21390.46 | -0.15% |
| drop first run per case/job | 22415.09 | 22369.76 | -0.20% |

The measured order effect is also non-negligible. Using the post-first-run
measurements:

| Position in job | Mean input tok/s |
| --- | ---: |
| first case | 22307.10 |
| second case | 22477.74 |

Second case over first case: `+0.77%`.

This order/noise effect is already larger than the combined Waterfill e2e
effect.

### Combined MMLU Mega-MoE Timing

Across all three MMLU jobs:

| Metric | `fused` | `fused_waterfill` |
| --- | ---: | ---: |
| timing samples | 966 | 970 |
| median tokens per sampled call | 516 | 543 |
| count ratio mean | 1.5448 | 1.1470 |
| count abs diff mean | 2969.58 | 185.07 |
| `start_to_topk_ms` mean | 0.116856 | 0.124582 |
| `cast_to_pre_dispatch_ms` mean | 0.183099 | 0.184356 |
| `pre_dispatch_to_fp8_fp4_ms` mean | 0.807407 | 0.745459 |

Waterfill reduces the fused Mega-MoE compute segment by `0.061948 ms`, about
`7.7%` for this segment. But the end-to-end MMLU request has mean e2e latency
around `1333 ms`; the MoE segment savings are small in absolute time and are
not visible as a stable serving throughput gain.

## Follow-up Root Cause: Why `30% * 5-8%` Did Not Become Stable E2E Gain

A later paired profiler/debug pass showed that the simple `Mega-MoE share *
Mega-MoE speedup` estimate was too optimistic for two reasons:

1. The earlier `MEGA_MOE_TIMING` aggregate was a simple per-call average. It
   was not token-weighted and did not model the per-forward slow-rank critical
   path.
2. Waterfill adds its own count/expand kernels before Mega-MoE. Those kernels
   are small, but they are large enough to eat a meaningful part of the
   Mega-MoE core saving on MMLU-sized batches.

Important measurement correction: jobs with
`SGLANG_WATERFILL_LOG_STATS_INTERVAL=1` are useful for balance debugging but
must not be used as clean Mega-MoE core timing evidence. The stats path runs
extra per-layer accounting only for Waterfill cases and can inflate the setup
interval enough to make a no-remote-shared `remote_cost=16384` case look
artificially slow.

Clean log-ready diagnostic job `2695792` used
`SGLANG_WATERFILL_LOG_STATS_INTERVAL=0`, log-based server readiness, and a
DeepGEMM quiet wait. On the fixed MMLU seed/sample:

| Case | tput | token-weighted all-rank core | max-rank core mean | max-rank total mean | remote shared | max expert |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fused` | 1085 | 1.4078 | 1.3563 | 1.5032 | 0.0 | 5959.1 |
| `fused_waterfill` | 1089 | 1.4195 | 1.3886 | 1.5681 | 3231.9 | 3702.8 |
| `fused_waterfill_cost16384` | 1089 | 1.4060 | 1.3899 | 1.6016 | 0.0 | 5949.5 |
| `fused_waterfill_local` | 1089 | 1.3975 | 1.3714 | 1.4904 | 0.0 | 5950.9 |

The clean result shows that default Waterfill does reduce the largest expert
bucket, but it also creates remote shared fanout (`shared source edges` rises
from `2.00` to `3.91`). In this run, `remote_shared_entries` correlates with
max-rank core time at about `0.71`, which is the current best explanation for
why B200 Mega-MoE does not convert better balance into a clear wall-time gain.

Strategy sweep jobs `2695999` and `2696152` did not find a simple policy fix:

| Case | tput | max-rank pre mean | max-rank total mean | remote shared | max expert |
| --- | ---: | ---: | ---: | ---: | ---: |
| fused baseline | 1085 | 1.3563 | 1.5032 | 0.0 | 5959.1 |
| default Waterfill | 1089 | 1.3886 | 1.5680 | 3231.9 | 3702.8 |
| routed-only | 1087 | 1.3854 | 1.5587 | 3036.6 | 3769.2 |
| routed + cost4096 | 1088 | 1.3783 | 1.5351 | 368.4 | 5927.8 |
| routed + cost8192 | 1084 | 1.3741 | 1.5640 | 11.1 | 5959.1 |
| shared replicas 2 | 1024 | 5.6247 | 6.1365 | 3231.9 | 2705.8 |
| shared replicas 4 | 1011 | 5.3942 | 5.9029 | 3231.9 | 2521.1 |

Routed-rank restriction and static remote cost either keep enough shared remote
traffic to remain slow or remove the balance improvement. Extra shared replicas
reduce the expert bucket but introduce very large Mega-MoE outliers. The next
optimization should therefore model Mega-MoE grouped-kernel cost directly
instead of trying to reuse DeepEP's rank-count-only objective.

### Paired Clean Profiler

Job: `2655735`

Workload:

- `PERF_DATASET_NAME=mmlu_custom`
- `PERF_MMLU_NUM_EXAMPLES=128`
- `PERF_PREWARM_PROMPTS=128`
- `PERF_WARMUP_REQUESTS=128`
- Torch profiler GPU activities only
- `SGLANG_PROFILE_WITH_STACK=false`
- `SGLANG_PROFILE_RECORD_SHAPES=false`

The full prewarm matters. Without it, the waterfill profile window captured a
one-off DeepGEMM HC shape/autotune artifact:

- `sm100_tf32_hc_prenorm_gemm_impl<..., 13u, ...>` ran `16556` times
- the peer rank spent `~1.3 s` in `all_reduce_one_shot_kernel`

That artifact disappeared after prewarming the full MMLU prompt set used in
the profiled run.

Clean trace results:

| Metric | `fused` | `fused_waterfill` | Delta |
| --- | ---: | ---: | ---: |
| combined CUDA kernel sum | 3094.10 ms | 3101.04 ms | -0.22% speedup |
| max-rank kernel span | 2102.41 ms | 2052.38 ms | +2.44% speedup |
| Mega-MoE core | 1148.33 ms | 1101.10 ms | +4.29% speedup |
| Waterfill count | 0.00 ms | 13.99 ms | +13.99 ms |
| Waterfill expand | 0.00 ms | 8.51 ms | +8.51 ms |
| comm/allreduce | 391.03 ms | 409.76 ms | +18.73 ms |
| MHC/HC | 395.37 ms | 421.68 ms | +26.31 ms |

So the real steady-state Mega-MoE-core gain on this MMLU slice is about `4%`,
not a stable `8%`. Since Mega-MoE core is about `37%` of summed CUDA kernel
time in this workload, that gives only about `1.5%` gross total-kernel upside.
The Waterfill count/expand overhead is about `22.5 ms`, or `0.7%` of summed
CUDA kernel time, before considering normal comm/MHC noise.

### All-rank Timing Check

Job: `2656094`

This job enabled `SGLANG_MEGA_MOE_LOG_TIMING_INTERVAL=1` and
`SGLANG_MEGA_MOE_LOG_ALL_RANKS=1`. It is not a clean throughput run because
every MoE call synchronizes timing events, but it validates the timing
interpretation.

| Metric | `fused` | `fused_waterfill` | Delta |
| --- | ---: | ---: | ---: |
| simple mean `pre_dispatch_to_fp8_fp4_ms` | 0.6859 | 0.6320 | +7.9% |
| token-weighted mean | 1.1757 | 1.1125 | +5.4% |
| paired rank-max sum | 1684.43 ms | 1521.49 ms | +10.7% |

This confirms that Waterfill can reduce the Mega-MoE critical segment, but the
serving-level effect still depends on fixed overheads, Waterfill setup kernels,
and run-to-run batching variance.

### Full-prewarm 2000-example MMLU Perf

Jobs:

- `2656399`: `fused -> fused_waterfill`
- `2656838`: `fused_waterfill -> fused`

Workload:

- `PERF_MMLU_NUM_EXAMPLES=2000`
- `PERF_PREWARM_PROMPTS=256`
- `PERF_WARMUP_REQUESTS=128`
- `PERF_MEASURE_REPEATS=3`
- no profiler
- no Mega-MoE timing sync

Results:

| Job | Order | `fused` mean input tok/s | `fused_waterfill` mean input tok/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| `2656399` | fused then waterfill | 20979.38 | 20612.02 | -1.75% |
| `2656838` | waterfill then fused | 20959.92 | 21757.62 | +3.81% |
| combined | both orders | 20969.65 | 21184.82 | +1.03% |

The per-run variance is high:

| Case | n | mean input tok/s | median input tok/s | stdev |
| --- | ---: | ---: | ---: | ---: |
| `fused` | 6 | 20969.65 | 20055.90 | 1851.17 |
| `fused_waterfill` | 6 | 21184.82 | 21119.94 | 1151.78 |

This variance is larger than the expected net gain. The best interpretation is
that Waterfill is active and improves the Mega-MoE core, but the MMLU serving
benchmark at 2000 examples is too noisy to prove a stable 2-3% e2e gain.

## Code Path Interpretation

The relevant Mega-MoE path is in
`python/sglang/srt/layers/moe/mega_moe.py`:

1. `moe.topk(...)` computes routed plus fused shared expert ids.
2. `mega_moe_pre_dispatch(...)` packs tokens and topk metadata into the
   DeepGEMM Mega-MoE symmetric buffer.
3. `deep_gemm.fp8_fp4_mega_moe(...)` performs the grouped expert compute.

The logged interval `pre_dispatch_to_fp8_fp4_ms` covers the Mega-MoE compute
call after pre-dispatch. Waterfill affects the topk/shared assignment before
pre-dispatch. The measurements show that this improves rank balance and reduces
the Mega-MoE compute interval, so Waterfill is active and directionally
beneficial inside the MoE segment.

However, the absolute saved time is too small relative to the complete serving
request. In addition, the serving benchmark includes scheduler batching,
request interleaving, decode tokens, HTTP/client timing, and per-case server
restart effects. Those effects introduce throughput variation on the same order
as or larger than Waterfill's expected end-to-end gain.

## Why This Differs From Non-MegaMoE / Hopper Results

The earlier stable gains on non-MegaMoE or Hopper should not be treated as an
equivalent prediction for the current B200 Mega-MoE path. The optimization is
nominally the same Waterfill policy, but it lands in a different runtime
pipeline.

### Normal DeepEP Path

The normal DeepEP path in `DeepseekV2MoE.forward_deepep` is:

```text
gate/topk -> dispatcher.dispatch -> run_moe_core -> dispatcher.combine
```

Waterfill changes the shared-expert target rank before dispatch. Therefore, it
can reduce the critical-path rank load seen by:

- DeepEP dispatch
- per-rank expert GEMM input size
- combine
- any overlap window around dispatch/combine and shared experts

On that path, a load-balance improvement can convert directly into a visible
serving throughput improvement because the optimized section is a larger part
of the total MoE layer critical path.

### Mega-MoE Path

The B200 Mega-MoE path measured here is:

```text
gate/topk -> mega_moe_pre_dispatch -> deep_gemm.fp8_fp4_mega_moe -> TP allreduce
```

Waterfill still changes the topk/shared assignment, but in this path it mainly
reduces the `deep_gemm.fp8_fp4_mega_moe` interval. It does not remove the fixed
parts around the layer:

- gate/topk
- pre-dispatch packing/quantization
- result scaling/add
- TP allreduce
- attention, scheduler, HTTP/client timing, and decode outside the MoE layer

The measured data matches this interpretation:

- Random 8K: rank ratio improves from `1.3987` to `1.0266`, but
  `pre_dispatch_to_fp8_fp4_ms` improves by only `0.02879 ms`.
- MMLU: rank ratio improves from `1.5448` to `1.1470`, and
  `pre_dispatch_to_fp8_fp4_ms` improves by `0.061948 ms`.
- End-to-end MMLU still aggregates to about `-0.2%` because the fixed and noisy
  parts dominate the smaller sub-kernel gain.

### Why the `MoE span` Speedup Looks Lower Than H20

The `+2.37%`/`+2.44%` number is not the pure Mega-MoE core speedup. It is a
wider max-rank GPU-span measurement around the profiled Mega-MoE calls. In the
clean no-red B200 trace (`2655735`), dropping the first profiled step gives:

| Metric | fused | fused + Waterfill | Speedup |
| --- | ---: | ---: | ---: |
| max-rank full GPU span | `2003.44 ms` | `1955.74 ms` | `+2.44%` |
| max-rank MoE span | `645.86 ms` | `617.61 ms` | `+4.58%` |
| MoE share of full span | `32.24%` | `31.58%` | - |

That implies a simple full-span target of about `+1.43%`:

```text
MoE share 32.24% * MoE span speedup 4.58% ~= +1.48%
```

The observed full-span speedup (`+2.44%`) is already above that simple target.
So this trace does not show a broken Mega-MoE conversion. It shows that the
wide span is dominated by non-MoE and fixed work, while the narrower Mega-MoE
segment is improving.

This should not be compared directly with the H20 `+6.13%` serving result. The
H20 run was V3 `DeepEP normal`, EP8, DP attention, static logical-count
placement. Waterfill there acts on the broader DeepEP path:

```text
dispatch -> expert GEMM -> combine
```

The B200 run was V4 `Mega-MoE`, EP2, single-node. Waterfill there mainly
changes the fused Mega-MoE expert core. A later H20 MMLU profiler run with
shared-expert fusion enabled confirms that the H20 gain is broader than expert
GEMM alone:

| System | Backend/path | Strict evidence | Waterfill effect |
| --- | --- | --- | ---: |
| H20 V3 EP8 | DeepEP normal serving | same MMLU 2k tput run | `+6.13%` e2e |
| H20 V3 EP8 | DeepEP fusion profile | max-rank DeepEP MoE path | `+9.76%` MoE path |
| H20 V3 EP8 | DeepEP fusion profile | max-rank expert GEMM | `+4.27%` expert GEMM |
| B200 V4 EP2 | Mega-MoE no-red trace | max-rank MoE span | `+4.58%` MoE span |
| B200 V4 EP2 | Mega-MoE no-red trace | max-rank full GPU span | `+2.44%` full span |

The H20 profile path was
`/lustre/raplab/client/xutingz/workspace/bench/waterfill/h20_mmlu_profile_fusion_wf_20260617_120551`.
In that run, dispatch improved from `450.51 ms` to `410.97 ms` and combine
improved from `1232.40 ms` to `786.27 ms`, so the full MoE-path speedup is much
larger than the expert-GEMM-only speedup. This is the key reason the H20 e2e
gain is larger than the B200 Mega-MoE span number.

The remaining apples-to-apples gap is PR25391-style red32 placement. Earlier
B200 runs used `ep_num_redundant_experts=0` or trivial placement, while
PR25391 red32 used static placement from an MMLU logical-count file.

### Red32 Static Placement Bug in Mega-MoE

The first B200 red32 submissions were still not equivalent to PR25391. There
were two separate issues:

1. Setting `--ep-num-redundant-experts 32` alone allocated extra physical
   experts, but did not force routed topk IDs through the logical-to-physical
   dispatch table. `server_args.py` now defaults `ep_dispatch_algorithm` to
   `static` whenever redundant experts are requested.
2. `forward_mega_moe()` only built `ExpertLocationDispatchInfo` when
   `enable_eplb=True`. PR25391 red32 uses `enable_eplb=False` with
   `init_expert_location=<logical_count.pt>` and static dispatch, so Mega-MoE
   skipped the logical-to-physical remap even when the MMLU placement file was
   supplied.

The Mega-MoE path now calls:

```python
ExpertLocationDispatchInfo.init_new(layer_id=moe.layer_id)
```

unconditionally, matching the normal MoE paths. The helper returns `None` when
`ep_dispatch_algorithm` is unset, and otherwise carries the static or dynamic
dispatch table. This should make B200 Mega-MoE red32 actually use the same
logical-count placement mechanism as PR25391.

Pending verification jobs after this fix:

| Job | Purpose | Status |
| --- | --- | --- |
| `2678399` | short red32 MMLU 256c128, 1 warmup + 2 measure rounds, topk/remap logging | completed; diagnostic only |
| `2678908` | clean red32 MMLU 2000 examples, 2 warmup + 4 measure rounds, low logging | Slurm `Priority` pending |
| `2679024` | low-logging red32 MMLU profiler after warmup | Slurm `Priority` pending |
| `2679110` | force-local-shared red32 MMLU diagnostic | superseded by `2679486` |
| `2679486` | three-way red32 diagnostic: fused, Waterfill remote shared, Waterfill local shared | completed; timing diagnostic only |

### Red32 Mega-MoE Diagnostic: Balance Does Not Yet Become Steady-State Core Speedup

Short red32 job `2678399` confirms the static MMLU logical-count placement is
active in Mega-MoE:

- server args: `ep_num_redundant_experts=32`, `ep_dispatch_algorithm='static'`,
  `enable_eplb=False`
- expert-location map contains physical routed IDs above `255`, e.g. `286`
- Mega-MoE weight build shape is `(145, ...)` per rank: `144` physical routed
  experts plus `1` per-rank shared slot

This run had all-rank timing/topk/waterfill logging enabled, so its throughput
is not a valid perf number. It is still useful for internal timing diagnosis.
After removing two layer-0 JIT outliers above `10 ms`, the measurement-round
prefill groups show:

| Metric | fused red32 | fused + Waterfill red32 |
| --- | ---: | ---: |
| max-rank `fp8_fp4` mean | `1.4167 ms` | `1.4721 ms` |
| max-rank `fp8_fp4` p50 | `1.4430 ms` | `1.5032 ms` |
| max-rank `fp8_fp4` p95 | `1.5555 ms` | `1.5933 ms` |
| max rank-count mean | `28802` | `22931` |
| rank ratio mean | `2.05` | `1.15` |
| corr(`fp8_fp4`, max_count) | `0.895` | `0.792` |

So Waterfill is balancing the loads, but the steady-state Mega-MoE fused
expert kernel is not getting faster in this diagnostic. The most likely reason
is the fused shared-expert path: normal fused shared routes the appended shared
expert to the local per-rank shared slot, while Waterfill can route the shared
expert to a remote rank. On this B200 Mega-MoE path, that remote shared
placement appears to reduce max rank count but worsen the grouped/fused kernel
shape enough to erase the compute saving. Job `2679486` checks this directly
by comparing remote shared Waterfill with force-local shared Waterfill.

### Red32 Remote-vs-Local Shared Diagnostic

Job `2679486` reran the red32 MMLU timing diagnostic with three cases:

- `fused`: shared-expert fusion, no Waterfill
- `fused_waterfill`: Waterfill with remote shared assignment allowed
- `fused_waterfill_local`: Waterfill with
  `SGLANG_WATERFILL_FORCE_LOCAL_SHARED=1`

This job used all-rank timing on every MoE call and only two measured serving
rounds, so the throughput numbers are intentionally ignored. The timing data
is still useful because it compares the Mega-MoE critical segment directly.

Simple all-rank timing:

| Metric | `fused` | `fused_waterfill` | `fused_waterfill_local` |
| --- | ---: | ---: | ---: |
| token-weighted `pre_dispatch_to_fp8_fp4_ms` | `1.4298` | `1.4797` | `1.4229` |
| rank count ratio mean | `2.1382` | `1.2482` | `2.1152` |
| core speedup vs `fused` | baseline | `-3.37%` | `+0.49%` |

Max-rank prefill groups with at least 1024 tokens:

| Metric | `fused` | `fused_waterfill` | `fused_waterfill_local` |
| --- | ---: | ---: | ---: |
| max-rank `pre_dispatch_to_fp8_fp4_ms` mean | `1.4260` | `1.4786` | `1.4248` |
| rank count ratio mean | `2.2908` | `1.2142` | `2.2325` |
| mean core speedup vs `fused` | baseline | `-3.68%` | `+0.08%` |

This is the clearest answer to the low `MoE span` speedup question:

1. Remote shared Waterfill is not disabled. It improves balance strongly
   (`2.29 -> 1.21` on max-rank prefill groups).
2. That balance does not shorten the Mega-MoE core. The measured
   `pre_dispatch_to_fp8_fp4_ms` interval becomes about `3.7%` slower.
3. Forcing shared experts to stay local removes the core regression, but it
   also removes most of the balance improvement (`2.29 -> 2.23`), so the core
   is only flat against baseline.

The logged `pre_dispatch_to_fp8_fp4_ms` interval is inside the DeepGEMM
Mega-MoE core path; it excludes Waterfill count/expand and pre-dispatch setup.
Therefore the remote-shared regression is not just Waterfill bookkeeping
overhead. It means the current rank-count objective is not a sufficient cost
model for B200 Mega-MoE. Remote shared placement likely changes remote token
movement, grouped-GEMM shape, locality, or synchronization enough to offset the
load-balance gain.

There is still extra overhead around the core. In the same max-rank prefill
groups, the topk/count/expand side work increased from `321.86 ms` to
`432.23 ms` when remote shared Waterfill was enabled. That is secondary to the
core regression, but it further explains why the wider `MoE span` speedup is
low.

### Candidate-Restricted Static Waterfill Experiment

The next optimization hypothesis is that PR25391's static Waterfill objective
is too broad for the B200 Mega-MoE fused path. Normal DeepEP can route the
shared expert to any underloaded rank and then benefit in dispatch/combine. In
Mega-MoE, assigning a token's shared expert to a rank that none of its routed
experts already use may introduce an extra remote token/rank pair inside the
fused DeepGEMM path. That can make the core slower even though rank-count
balance improves.

The code now has a guarded experiment knob:

```bash
SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS=0
```

Default remains `1`, which preserves PR19290/PR25391 behavior. When set to
`0`, static Waterfill restricts the shared-expert target to the token's routed
expert ranks plus the source rank. This reuses the existing non-`ALLOW_ALL`
selection semantics in `deepep_waterfill.py`. For the B200 target EP2 setup,
the implementation uses a dedicated EP2 Triton expansion kernel instead of the
generic world-size candidate-mask loop, so the experiment does not add avoidable
Waterfill expand overhead.

Diagnostic logging also now records:

- `shared_remote`: number of tokens whose shared expert was assigned to a
  non-source rank
- `shared_remote_new_rank`: subset where the assigned shared rank was not
  already present in that token's routed TopK ranks

The B200 validation job for this hypothesis is:

| Job | Cases | Status |
| --- | --- | --- |
| `2680742` | intended `fused_waterfill`, `fused_waterfill_routed` | canceled; submitted with default sbatch env, so it started the wrong cases/model |
| `2680865` | `fused_waterfill`, `fused_waterfill_routed` | completed |

`fused_waterfill_routed` is the candidate-restricted case. Throughput from this
short job should still be treated as diagnostic only because it enables
all-rank timing and per-call Waterfill stats.

Job `2680865` eliminated the extra "new remote rank" assignment but did not
recover a Mega-MoE core speedup:

| Metric | remote shared | routed-rank-only shared |
| --- | ---: | ---: |
| `SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS` | `1` | `0` |
| token-weighted `pre_dispatch_to_fp8_fp4_ms` | `1.7766` | `1.7853` |
| token-weighted core speedup vs remote | baseline | `-0.49%` |
| all-call rank ratio mean | `1.3834` | `1.3758` |
| Waterfill after-ratio mean | `1.1457` | `1.1745` |
| `shared_remote_mean` | `5481.2` | `5340.2` |
| `shared_remote_new_rank_mean` | `349.4` | `0.0` |

The max-rank prefill view is also flat: `pre_dispatch_to_fp8_fp4_ms` mean is
`2.0289 ms` for remote shared and `2.0240 ms` for routed-rank-only shared, with
similar or slightly worse balance in the restricted case. This rejects the
simple hypothesis that "new remote token/rank pairs" are the main reason the
B200 red32 Mega-MoE core is slow. They are a real behavioral difference, but
removing them is not enough. The remaining issue is that the current
rank-count objective still does not match the actual DeepGEMM Mega-MoE cost:
remote shared token volume, grouped-GEMM shape, per-rank active expert shape,
locality, and synchronization are likely all part of the cost model.

### Local-Preference / Shape Diagnostic

Job `2681601` swept the static Waterfill local preference in the red32 B200
Mega-MoE setup:

- `fused_waterfill`: default local preference `11/10`
- `fused_waterfill_pref2`: `2/1`
- `fused_waterfill_pref4`: `4/1`
- `fused_waterfill_pref8`: `8/1`

This was a diagnostic run with all-rank per-call timing and Waterfill stats
enabled. It did not fix the MMLU sampling seed, so serving throughput from this
job is not a valid e2e comparison. The shape/timing logs are still useful.

All-call summary:

| Metric | `11/10` | `2/1` | `4/1` | `8/1` |
| --- | ---: | ---: | ---: | ---: |
| token-weighted `pre_dispatch_to_fp8_fp4_ms` | `1.7623` | `1.7568` | `1.7443` | `1.7239` |
| mean `pre_dispatch_to_fp8_fp4_ms` | `1.4548` | `1.4853` | `1.4451` | `1.4304` |
| Waterfill after-ratio mean | `1.1849` | `1.1643` | `1.1753` | `1.2102` |
| `remote_shared_entries` mean | `4669.2` | `4770.9` | `4464.7` | `4305.4` |
| `max_expert_tokens` mean | `4679.3` | `4789.5` | `4554.6` | `4516.0` |
| `p95_nonzero_expert_tokens` mean | `396.3` | `415.1` | `394.4` | `393.7` |

Max-rank prefill groups show the same direction:

| Metric | `11/10` | `8/1` |
| --- | ---: | ---: |
| max-rank `pre_dispatch_to_fp8_fp4_ms` mean | `2.0285` | `1.9481` |
| max-rank `pre_dispatch_to_fp8_fp4_ms` p50 | `2.0629` | `2.0152` |
| max-rank `pre_dispatch_to_fp8_fp4_ms` p95 | `2.1516` | `2.1085` |
| rank ratio mean | `1.2316` | `1.2699` |
| `remote_shared_entries` mean | `6217.6` | `5904.7` |
| `max_expert_tokens` mean | `6217.6` | `5918.6` |

Increasing local preference reduces the giant remote-shared bucket and recovers
some core time, but it also weakens rank balance. The best point in this short
sweep (`8/1`) is still only a `~4%` max-rank prefill core improvement against
default Waterfill, not a broader H20-like MoE-path improvement.

The strongest signal is the shape correlation. In max-rank prefill groups,
`pre_dispatch_to_fp8_fp4_ms` correlates with `max_expert_tokens` /
`remote_shared_entries`:

| Case | corr(core, `max_expert_tokens`) | corr(core, `shared_remote_new_rank`) |
| --- | ---: | ---: |
| `11/10` | `0.51` | `0.14` |
| `8/1` | `0.82` | `0.42` |

This refines the explanation: the problem is not mainly the existence of a new
remote rank per token. It is that Waterfill balances rank totals by creating a
large shared-expert token bucket, and the B200 Mega-MoE grouped kernel is
sensitive to that expert-level shape. Rank-count balance alone is the wrong
objective for this fused shared path.

Fixed-seed follow-up:

Job `2682308` repeated the default `11/10` vs `8/1` comparison with
`TPUT_SEED=20260617`. This run still enabled all-rank per-call timing, so its
serving throughput is intentionally ignored. It confirms the shape diagnosis
under identical request sampling:

| Metric | default `11/10` | `8/1` |
| --- | ---: | ---: |
| all-call token-weighted `pre_dispatch_to_fp8_fp4_ms` | `1.7314` | `1.7081` |
| all-call `remote_shared_entries` mean | `4756.8` | `4355.1` |
| Waterfill after-ratio mean | `1.1620` | `1.2142` |
| max-rank prefill `pre_dispatch_to_fp8_fp4_ms` mean | `1.9668` | `1.9327` |
| max-rank prefill rank-ratio mean | `1.2333` | `1.2753` |
| max-rank prefill `remote_shared_entries` mean | `6078.5` | `6046.3` |
| max-rank prefill `max_expert_tokens` mean | `6078.5` | `6048.8` |

So the stronger local preference improves the max-rank prefill Mega-MoE core
by only `~1.8%` in the fixed-seed run and worsens rank balance. More
importantly, the prefill critical path still has a `~6K` shared-expert bucket.
The fused shared path is therefore still dominated by expert-level shape, not
by the rank-total balance objective.

### Architecture / Workload Implication

The prior 4% gain likely came from a workload/backend combination where the
DeepEP MoE dispatch/compute/combine critical path was a large enough fraction
of the full request. In the current B200 Mega-MoE measurements:

1. Mega-MoE's fused DeepGEMM path already makes the expert compute segment
   relatively small.
2. In the no-red trace, Waterfill improved the Mega-MoE segment by a few
   percent, but the absolute saving was tens of microseconds per sampled MoE
   layer call.
3. In the red32 remote-shared Mega-MoE path, Waterfill improves rank balance
   but does not improve the fused core time.
4. Restricting shared experts to routed ranks removes `shared_remote_new_rank`
   assignments, but the core remains flat, so the problem is broader than just
   extra remote token/rank pairs.
5. A stronger local preference can reduce the remote-shared expert bucket and
   recover some core time, but it trades off balance and does not create the
   H20-style dispatch/combine savings.
6. The serving-level variance from request batching and run order is larger
   than the expected e2e effect.

So the missing 4% does not contradict the old result. It shows that the old
result depended on the non-MegaMoE/Hopper runtime bottleneck, while the current
B200 Mega-MoE runtime has moved the bottleneck elsewhere.

## Conclusion

Waterfill is functionally active:

- rank load ratio improves substantially
- red32 static logical-to-physical placement is now actually used by Mega-MoE

Waterfill does not show a stable end-to-end throughput improvement over regular
shared-expert fusion in these serving benchmarks because:

1. The optimized segment is a small part of the full request latency.
2. In red32 Mega-MoE, better balance has not yet converted into steady-state
   fused-kernel speedup; the current suspect is a mismatch between the
   Waterfill rank-count objective and Mega-MoE's real grouped-kernel cost.
3. End-to-end serving variance from batching/order/node state is comparable to
   or larger than the expected gain:
   - combined MMLU Waterfill effect: about `-0.2%`
   - measured case-order effect: about `+0.77%`

Therefore, the correct interpretation is not "Waterfill is disabled" or
"Waterfill is functionally wrong"; it is "Waterfill balances Mega-MoE rank
load, but on B200 Mega-MoE red32 the fused shared-expert path does not yet turn
that better balance into a stable core or serving-level speedup."

## Shared-Replica Follow-Up

The fixed-seed diagnostics point to an expert-level bucket problem. In the
per-rank fused shared layout, `num_fused_shared_experts=1` previously meant both
one logical shared TopK column and one physical shared expert slot per EP rank.
Waterfill can choose the target rank, but all shared tokens assigned to that
rank still accumulate in a single physical shared expert bucket. On B200
Mega-MoE red32, the max-rank prefill critical path remained dominated by a
`~6K` shared bucket even when rank counts improved.

The new experimental knob is:

```text
SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK=<N>
```

Default `N=1` is the existing layout. With `N>1`, the logical shared expert is
still a single TopK column and has the same weight, but each EP rank owns `N`
physical shared slots. The checkpoint shared expert weights are copied into all
replicas, and Waterfill hashes each selected shared token across replicas on
the chosen rank. This changes the B200 red32 layout as follows:

| replicas/rank | total experts | experts/rank | rank0 shared ids | rank1 shared ids |
| ---: | ---: | ---: | --- | --- |
| `1` | `290` | `145` | `144` | `289` |
| `2` | `292` | `146` | `144,145` | `290,291` |
| `4` | `296` | `148` | `144..147` | `292..295` |

The intended metric is whether `max_expert_tokens` drops roughly with the
replica count and whether the max-rank `pre_dispatch_to_fp8_fp4_ms` MoE span
then moves closer to the H20-style MoE improvement. This directly tests a
narrow explanation for the previous low `+2.37%` MoE span speedup: if a single
critical shared bucket is the limiter, splitting it should expose the expected
compute speedup. The smoke result below shows that this narrow explanation is
not sufficient.

B200 validation jobs:

- Job `2683472`: `fused`, `fused_waterfill`, `fused_waterfill_rep2`,
  `fused_waterfill_rep4`
- Job `2683760`: shorter multi-partition smoke with `fused_waterfill` and
  `fused_waterfill_rep2`
- Dataset/placement: same fixed-seed MMLU 2k source with red32 static
  logical-count placement
- Logging: all-rank `MEGA_MOE_TIMING` and `WATERFILL_STATS`
- Current status: `2683760` completed the rep1/rep2 smoke; `2683472` remains
  pending for the full rep1/rep2/rep4 comparison.

### Rep2 Smoke Result

Job `2683760` completed the `fused_waterfill` and `fused_waterfill_rep2` smoke.
The rep2 case is functionally active:

- local expert weight shape changed from `145` to `146` experts per rank
- `MEGA_MOE_TIMING` reports `shared_replicas_per_rank=2`
- prefill `max_expert_tokens` dropped from roughly one full shared bucket to
  about half that size

The raw summary is not a valid steady-state timing comparison because the first
rep2 requests triggered new DeepGEMM JIT/warmup outliers:

| Case | `pre_dispatch` outlier examples |
| --- | --- |
| rep1 | layer 0 first requests around `6-7 ms` |
| rep2 | layer 0 first requests around `2137-2398 ms` |

After excluding paired prefill groups with rank-max
`pre_dispatch_to_fp8_fp4_ms >= 5`, the steady-state result is:

| Metric | rep1 | rep2 | Change |
| --- | ---: | ---: | ---: |
| paired prefill groups | `257` | `257` | same |
| rank-max `max_expert_tokens` mean | `6138.36` | `3403.76` | `0.55x` |
| rank-max `max_local_expert_tokens` mean | `2599.48` | `2585.18` | `0.99x` |
| rank-max `active_local_experts` mean | `134.57` | `134.91` | `+0.25%` |
| rank-max `p95_nonzero_expert_tokens` mean | `478.04` | `499.39` | `+4.5%` |
| global incoming count ratio mean | `1.0587` | `1.0587` | unchanged |
| global incoming count max mean | `44223.85` | `44223.85` | unchanged |
| rank-max `pre_dispatch_to_fp8_fp4_ms` mean | `1.4596` | `1.4862` | `-1.8%` |

This means the original shared-bucket hypothesis was incomplete. Rep2 splits
the per-source global shared bucket, but the Mega-MoE kernel does not get less
critical-path work on the destination rank. The rank-local grouped-GEMM shape
that better tracks the kernel, especially `max_local_expert_tokens`, barely
changes, and active local expert count slightly increases. Splitting one
shared expert into two physical expert groups on the same destination rank is
therefore not equivalent to reducing the slow rank's compute span.

### Global Count Recheck

The earlier red32 result emphasized per-source rank-count ratio:

```text
fused: 2.29
fused + Waterfill: 1.21
```

That metric was too pessimistic for fused and too optimistic for Waterfill
because Mega-MoE's slow rank is determined by incoming work summed across TP
source ranks. Recomputing job `2679486` with paired global counts gives:

| Metric | fused | fused + Waterfill |
| --- | ---: | ---: |
| global incoming count ratio mean | `1.0859` | `1.0647` |
| global incoming count p95 | `1.2347` | `1.1737` |
| global incoming count max mean | `45461.53` | `45454.91` |
| global incoming count diff mean | `3487.56` | `2696.47` |
| rank-max `pre_dispatch_to_fp8_fp4_ms` mean | `1.4160` | `1.4752` |

Waterfill mostly raises the underloaded rank and reduces the count difference;
it does not lower the slow-rank incoming max in this TP2 B200 workload. That is
the main reason the observed MoE span speedup is only `+2.37%` in the cleaner
profile, and why it remains much lower than H20's normal DeepEP MoE result.

### Dynamic Global-Objective Check

Job `2687128` tested a dynamic/global Waterfill variant after fixing its target
semantics. The first dynamic attempt was invalid because it added
`local_tokens_per_rank` to the initial rank load, effectively treating the
shared tokens as already assigned before Waterfill selected them. The corrected
dynamic path all-reduces only routed counts, then uses
`global_routed + total_shared_tokens` to derive the final target.

The run compared `fused_waterfill` and `fused_waterfill_dynamic` on the same
fixed-seed MMLU source and red32 static placement. Throughput is only a debug
signal because every MoE call is timing-synchronized.

| Metric | static Waterfill | dynamic/global Waterfill | Change |
| --- | ---: | ---: | ---: |
| prefill global incoming count ratio mean | `1.0582` | `1.0014` | nearly flat |
| prefill global incoming count max mean | `42845.22` | `41673.45` | `-2.7%` |
| prefill rank-max `pre_dispatch_to_fp8_fp4_ms` sum | `817.95 ms` | `795.63 ms` | `+2.81%` |
| prefill rank-max `topk`/setup sum | `389.30 ms` | `517.06 ms` | `-24.7%` |
| prefill rank-max total logged sum | `1208.59 ms` | `1312.90 ms` | `-7.95%` |
| prefill rank-max `max_expert_tokens` mean | `5949.71` | `3984.88` | `0.67x` |
| all-call token-weighted `pre_dispatch_to_fp8_fp4_ms` | `1.5113` | `1.4452` | `+4.57%` |

This isolates the conversion limit. Once Waterfill optimizes the true global
incoming rank load, B200 Mega-MoE does recover a small core speedup. But the
slow-rank core speedup is still only about `3%` on the prefill span, and the
extra dynamic all-reduce/topk setup is much larger than the core saving in this
implementation. Compared with H20, the B200 fused path is missing the big
normal-DeepEP dispatch/combine reductions; its core-only gain is instead close
to H20's expert-GEMM-only `+4.27%`.

### Static Dispatch-Map Rank-Load Recheck

Job `2688000` reran the same fixed-seed MMLU red32 workload after changing the
static Waterfill metadata to estimate rank load from the actual static
logical-to-physical dispatch map, rather than from a replica-split proxy. The
job compared:

- `fused_waterfill`: static dispatch-map rank-load target
- `fused_waterfill_dynamic`: dynamic/global all-reduce target

The new static target is active; the server logged per-layer loads such as
layer 29 `[20913, 19706]` and layer 40 `[20005, 21152]`, which are not the
near-perfectly balanced replica-split estimates used before.

Prefill rank-max groups after excluding DeepGEMM warmup outliers:

| Metric | old static/local `2687128` | static map-load `2688000` | dynamic/global `2688000` |
| --- | ---: | ---: | ---: |
| global incoming count ratio mean | `1.0582` | `1.0767` | `1.0014` |
| global incoming count max mean | `42845.22` | `43208.60` | `41730.37` |
| rank-max `pre_dispatch_to_fp8_fp4_ms` sum | `817.95 ms` | `778.37 ms` | `767.66 ms` |
| rank-max `topk`/setup sum | `389.30 ms` | `397.57 ms` | `517.17 ms` |
| rank-max total logged sum | `1208.59 ms` | `1176.04 ms` | `1286.63 ms` |
| `max_expert_tokens` mean | `5949.71` | `3702.85` | `3990.55` |
| `max_local_expert_tokens` mean | `2483.63` | `3348.68` | `3979.80` |
| `remote_shared_entries` mean | `5949.71` | `3231.93` | `3596.78` |

This confirms the static rank-load bug was real and worth fixing: the static
map-load target improves the slow-rank Mega-MoE core by about `4.5%` over the
old static/local target and cuts the gap to the dynamic/global target from
`2.81%` to `1.40%`.

However, it does not change the main B200/H20 explanation. Compared with the
plain fused baseline from job `2687589`, the Waterfill variants still do not
reduce the slow-rank incoming count max; static map-load is `43208.60` versus
fused `43193.80`, while dynamic/global lowers it only to `41730.37`. At the
same time, Waterfill introduces remote shared work and multi-rank full-token
fanout inside Mega-MoE. The strongest correlations with
`pre_dispatch_to_fp8_fp4_ms` are `remote_routed_entries`,
`tokens_multi_full_rank`, and `max_expert_tokens`, not just rank-count ratio.
So the low B200 MoE span speedup is primarily a Mega-MoE cost-model issue:
the rank-count objective improves balance, but the fused Mega-MoE path is
also sensitive to remote shared placement and grouped-kernel shape.

### Locality-Penalty Recheck

Job `2688305` reran the fixed-seed MMLU red32 workload with the static
dispatch-map rank-load target and swept a stronger static local preference:
default `11/10`, `8/1`, `16/1`, and `32/1`. This directly tests whether the
low B200 MoE span speedup is mainly caused by remote shared placement.

Serving throughput is measured with one warmup round and one measured round
over the same 64 MMLU prompts, concurrency 64, `max_tokens=1`; all-rank
Mega-MoE timing is enabled on every call, so throughput is a diagnostic signal
rather than a clean production number. The max-rank prefill analyzer excludes
DeepGEMM warmup outliers with `pre_dispatch_to_fp8_fp4_ms >= 5`.

| Case | tok/s | token-weighted `pre_dispatch_to_fp8_fp4_ms` | prefill rank-max core sum | Speedup vs fused | global incoming max | rank ratio | remote shared | topk/setup sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fused | `1086` | `1.3762` | `742.87 ms` | baseline | `43193.80` | `2.2344` | `0.00` | `79.02 ms` |
| Waterfill `11/10` | `1078` | `1.4317` | `782.50 ms` | `-5.06%` | `43149.22` | `1.6236` | `3228.68` | `397.18 ms` |
| Waterfill `8/1` | `1063` | `1.4290` | `780.04 ms` | `-4.77%` | `43207.10` | `2.0271` | `1113.81` | `387.93 ms` |
| Waterfill `16/1` | `1083` | `1.4245` | `785.40 ms` | `-5.41%` | `43158.37` | `2.0993` | `764.76` | `554.63 ms` |
| Waterfill `32/1` | `1082` | `1.4213` | `772.04 ms` | `-3.78%` | `43241.38` | `2.1458` | `566.69` | `380.39 ms` |

The strongest local preference reduces remote shared entries by about `5.7x`
relative to default Waterfill (`3228.68 -> 566.69`) and recovers only `1.35%`
of the default Waterfill max-rank prefill core time (`782.50 -> 772.04 ms`).
It still remains slower than plain fused (`742.87 ms`).

This rejects the simple "remote shared alone" explanation. The global incoming
max stays essentially flat around `43.2K` in every case, so none of these
static Waterfill variants reduces the slow rank's total incoming work. Strong
local preference mostly trades away balance, while the fused Mega-MoE core
still sees similar routed remote work and a less favorable grouped-kernel
shape. The remaining gap to H20 is therefore not a disabled Waterfill path; it
is that the B200 Mega-MoE fused path needs a cost model closer to its real
critical path than rank-count balancing plus a local bias.

### Remote-Cost, One-Way, and Local-Replica Diagnostics

Job `2696444` swept middle remote costs on the fixed-seed MMLU red32 B200
Mega-MoE setup. None opened a useful window:

| Case | tok/s | max-rank core mean | Core speedup vs fused | max expert | remote shared | shared source edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused | `1082` | `1.3620 ms` | baseline | `5959.08` | `0.00` | `2.00` |
| Waterfill | `1089` | `1.3961 ms` | `-2.44%` | `3705.47` | `3235.40` | `3.91` |
| cost512 | `1087` | `1.3961 ms` | `-2.44%` | `3878.85` | `3015.34` | `3.91` |
| cost1024 | `1089` | `1.3897 ms` | `-1.99%` | `4105.85` | `2770.22` | `3.91` |
| cost2048 | `1089` | `1.3856 ms` | `-1.70%` | `4756.99` | `2105.39` | `3.91` |

This shows a scalar remote penalty mainly trades away balance; it does not make
the DeepGEMM Mega-MoE core faster.

I then tested one-way shared placement in jobs `2696876` and `2697005`, plus a
local-only shared-replica diagnostic in `2697089`.

| Case | tok/s | max-rank topk/setup | max-rank core | Core speedup vs fused | max-rank total | max expert | remote shared | shared source edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fused | `1087` | `0.1410 ms` | `1.3705 ms` | baseline | `1.5161 ms` | `5959.08` | `0.00` | `2.00` |
| Waterfill | `1090` | `0.4364 ms` | `1.4002 ms` | `-2.12%` | `1.8332 ms` | `3701.38` | `3232.15` | `3.90` |
| one-way shared | `1089` | `0.2126 ms` | `1.4215 ms` | `-3.59%` | `1.6318 ms` | `5947.97` | `3134.09` | `2.90` |
| local shared rep2 | `1089` | `0.1159 ms` | `1.4179 ms` | `-3.34%` | `1.5405 ms` | `3288.42` | `0.00` | `2.00` |

The one-way experiment reduces shared source edges but loses the shared-bucket
split that produced Waterfill's `max_expert_tokens` reduction. The local-rep2
experiment reduces the shared bucket without remote shared fanout and even has
lower setup time than fused, but the DeepGEMM Mega-MoE core is still slower.

That narrows the bottleneck further: B200 Mega-MoE is not failing because
remote shared load balancing is disabled, and not because shared source fanout
alone is too high. In this setup, Waterfill does not reduce the part that
dominates the fused Mega-MoE core: the routed remote pattern / active-expert
shape handled inside DeepGEMM. Shared-expert reshaping can move setup and
bucket metrics, but it does not reduce the max-rank core span.

### Destination-Recv Block Diagnostic

Job `2697220` added per-call `expert_counts` logging and recomputed the B200
red32 fixed-seed MMLU comparison from the destination/recv-rank perspective.
This is the metric that should track Mega-MoE's slow rank: sum the expert
counts from both TP/source ranks, split by destination rank, then estimate the
local grouped-GEMM block count with `ceil(expert_count / block_m)`.

The run used the same red32 static MMLU placement and compared `fused` with
default `fused_waterfill`. It still enabled all-rank timing on every MoE call,
so throughput is diagnostic only. The prefill groups below have at least 1024
tokens.

| Metric | fused | fused + Waterfill |
| --- | ---: | ---: |
| paired prefill groups | `559` | `559` |
| debug tput round | `1078 tok/s` | `1084 tok/s` |
| max-rank core mean | `1.4087 ms` | `1.3932 ms` |
| max-rank core sum | `787.44 ms` | `778.80 ms` |
| core speedup | baseline | `+1.11%` |
| max recv-rank tokens | `43122.98` | `43137.73` |
| recv token ratio | `1.0759` | `1.0765` |
| max recv blocks, `block_m=128` | `410.30` | `410.41` |
| recv block ratio, `block_m=128` | `1.0596` | `1.0599` |
| max recv blocks, `block_m=64` | `742.59` | `742.77` |
| max recv blocks, `block_m=256` | `248.33` | `248.33` |
| recv active local experts | `135.16` | `135.16` |
| source-local max expert tokens | `5949.54` | `3338.11` |
| source-local local blocks, `block_m=128` | `302.75` | `278.94` |
| remote shared entries | `0.00` | `3232.15` |
| shared source edges | `2.00` | `3.90` |
| max-rank topk/setup sum | `81.57 ms` | `216.89 ms` |
| max-rank total logged sum | `869.61 ms` | `991.59 ms` |

This is the cleanest explanation so far for the low B200 Mega-MoE speedup.
The old source-local metrics move a lot: Waterfill cuts the apparent
source-local max expert bucket from `5949.54` to `3338.11` and reduces
source-local `block_m=128` blocks by about `7.9%`. But after aggregating the
two source ranks into the actual destination/recv-rank work, the slow rank is
unchanged: max recv tokens are slightly higher, max recv `block_m=128` blocks
are slightly higher, and active local expert count is identical.

Therefore, default static Waterfill is improving the wrong balance metric for
this B200 Mega-MoE shared-fusion path. It makes each source rank look more
balanced, but the two source ranks' shared movements largely cancel out at the
destination-rank aggregate that determines the Mega-MoE wait time. The small
`+1.11%` core gain in this diagnostic is not a real rank-work reduction, and
the added topk/setup work is much larger than that saving.

The per-destination split confirms the same point:

| Metric | fused | fused + Waterfill |
| --- | ---: | ---: |
| recv rank0 tokens mean | `41548.81` | `41400.76` |
| recv rank1 tokens mean | `41744.73` | `41892.77` |
| recv rank0 `block_m=128` blocks mean | `398.14` | `397.05` |
| recv rank1 `block_m=128` blocks mean | `399.46` | `400.68` |
| groups where rank0 is max | `281` | `261` |
| groups where rank1 is max | `278` | `298` |
| mean token diff between ranks | `2952.42` | `2981.92` |
| mean block diff between ranks | `23.01` | `23.08` |

Waterfill shifts a small amount of work from rank0 to rank1, but rank1 was
already equally likely to be the max. The result is no improvement in the
destination max and a tiny worsening in recv-rank imbalance. That explains why
the Mega-MoE span does not scale with the source-local balance improvement.

### Source-Aware Static Load and Remote-Cost Fix

The next diagnostic changed the static Waterfill objective for B200 Mega-MoE.
Instead of using one offline `(layer, rank)` load vector for every source rank,
the loader now derives `(layer, source_rank, destination_rank)` load from the
static dispatch map. At runtime each source rank builds:

```text
rank_load = current_local_routed_counts + offline_other_source_rank_load
```

This matches the destination/recv-rank objective better than the old
source-local view.

Job `2697498` compared the old static Waterfill with the new source-aware
path on the fixed MMLU 2k slice, using
`/home/scratch.xutingz_wwfo_2/model/DeepSeek-V4-Flash`.

| Case | tput | token-weighted core | source-aware | remote shared | max local expert |
| --- | ---: | ---: | --- | ---: | ---: |
| fused | `1084` | `1.4165 ms` | no | `0.0` | `4606.4` |
| old Waterfill | `1086` | `1.4269 ms` | no | `2355.7` | `2496.2` |
| source-aware Waterfill | `1086` | `1.4272 ms` | yes | `3045.1` | `2269.4` |

The source-aware objective did improve the destination metrics:

| Prefill metric | fused | source-aware Waterfill |
| --- | ---: | ---: |
| recv-rank max tokens | `43122.98` | `42174.35` |
| recv-rank max `block_m=128` blocks | `410.30` | `403.13` |
| global incoming ratio | `1.0759` | `1.0278` |

But the prefill Mega-MoE core span still did not improve:
`pre_sum_speedup_pct = -0.40%`. The reason is that B200 Mega-MoE pays for
remote shared placement inside the fused DeepGEMM path. A per-prefill
regression on this run showed:

- `recv_b128` has the expected positive correlation with core time.
- `remote_shared_entries` and `shared_remote_new_rank` are also positively
  correlated with core time.
- The block reduction from source-aware was too small to offset the additional
  remote shared traffic.

I then swept source-aware static Waterfill with an explicit remote shared cost
penalty. Job `2697718` used the same MMLU source and ran:

```bash
CASE_ORDER=fused:fused_waterfill_cost4096:fused_waterfill_cost8192:fused_waterfill_cost16384
SGLANG_WATERFILL_SOURCE_AWARE_STATIC_LOAD=1
```

The names in this run predate the explicit `source_cost*` aliases; all three
cost cases below had source-aware static load enabled.

| Case | tput | token-weighted core | remote cost | remote shared | max local expert |
| --- | ---: | ---: | ---: | ---: | ---: |
| fused | `1075` | `1.4149 ms` | `0` | `0.0` | `4606.4` |
| source-aware cost4096 | `1083` | `1.3881 ms` | `4096` | `2045.4` | `2963.2` |
| source-aware cost8192 | `1083` | `1.3940 ms` | `8192` | `218.6` | `4456.9` |
| source-aware cost16384 | `1083` | `1.3964 ms` | `16384` | `0.0` | `4606.4` |

Analyzer results on prefill groups with at least 1024 tokens:

| Case | core span speedup | recv-rank max blocks | global ratio | remote shared | topk/setup change |
| --- | ---: | ---: | ---: | ---: | ---: |
| source-aware cost4096 | `+3.80%` | `402.09` | `1.0216` | `3455.5` | `-65.7%` |
| source-aware cost8192 | `+3.19%` | `408.78` | `1.0657` | `534.4` | `-50.2%` |
| source-aware cost16384 | `+2.93%` | `410.30` | `1.0759` | `0.0` | `-50.7%` |

`remote_cost=4096` is the best point in this sweep. It reduces the true
destination max block work while not sending as much shared traffic as the
unpenalized source-aware objective. The measured debug tput improved from
`1075` to `1083` tok/s (`+0.74%`) and the Mega-MoE core span improved by
`+3.80%`.

Important caveat: the Waterfill topk/setup path is still expensive in these
all-rank timing runs. The analyzer's logged `total_sum` remains negative
because it includes Waterfill plan/materialization overhead and heavy timing
instrumentation. The fix recovers the expected Mega-MoE core/span direction;
the remaining work is to reduce the topk/setup overhead enough for the span
gain to convert into a larger end-to-end gain.

### Profile Correction: Cost4096 Does Not Reliably Speed Up Mega-MoE Span

The `+3.80%` source-aware `remote_cost=4096` result above came from all-rank
log timing. Follow-up clean serving and profiler runs show that this is not a
stable profiled max-rank Mega-MoE span gain.

Clean serving job `2698077` used the fixed MMLU 2k slice, disabled Waterfill
and Mega-MoE debug logging, used `TPUT_WARMUP_ROUNDS=2`, and measured five
serving rounds:

| Case | Round tput samples | Mean | Trimmed mean | vs fused trimmed |
| --- | --- | ---: | ---: | ---: |
| fused | `35814, 4949, 7412, 4173, 5483` | `11566.2` | `5948.0` | baseline |
| old Waterfill | `36279, 4848, 7448, 4194, 5492` | `11652.2` | `5929.3` | `-0.31%` |
| source-aware cost4096 | `36148, 4907, 7447, 4198, 5493` | `11638.6` | `5949.0` | `+0.02%` |

The first serving round is a large outlier in every case, so the trimmed mean
is the useful steady-state signal. It shows the source-aware cost4096 case is
effectively tied with fused.

CPU+GPU profile job `2698337` captured six prefill profile steps:

| Metric | fused | source-aware cost4096 | Change |
| --- | ---: | ---: | ---: |
| serving round tput | `4284` | `4340` | `+1.31%` |
| EXTEND max-rank Mega-MoE impl | `365.49 ms` | `362.50 ms` | `+0.83%` |
| EXTEND max-rank Mega-MoE span | `369.97 ms` | `366.90 ms` | `+0.84%` |
| EXTEND full trace span | `1903.53 ms` | `1895.01 ms` | `+0.45%` |
| DECODE max-rank Mega-MoE span | `32.17 ms` | `21.62 ms` | `+48.83%` |

GPU-only profile job `2698497` captured the full measured prefill window:

| Metric | fused | source-aware cost4096 | Change |
| --- | ---: | ---: | ---: |
| serving round tput | `13239` | `13801` | `+4.25%` |
| EXTEND max-rank Mega-MoE impl | `814.13 ms` | `811.48 ms` | `+0.33%` |
| EXTEND max-rank Mega-MoE span | `824.24 ms` | `821.62 ms` | `+0.32%` |
| EXTEND full trace span | `2318.25 ms` | `2256.49 ms` | `+2.74%` |
| EXTEND GPU active time | `2152.91 ms` | `2154.62 ms` | `-0.08%` |
| EXTEND MoE-span target e2e gain | baseline | `+0.12%` | small |
| DECODE max-rank Mega-MoE span | `30.15 ms` | `24.07 ms` | `+25.27%` |

The GPU-only profile is the strongest correction. The single measured serving
round looks `+4.25%` faster and the full trace span looks `+2.74%` faster, but
the GPU active time is flat/slightly worse and the EXTEND max-rank Mega-MoE
span only improves by `+0.32%`. That means the observed one-round serving
speedup is mostly not coming from the Mega-MoE fused compute path.

The likely reason is that B200 Mega-MoE uses a fixed padded capacity for the
fused DeepGEMM path. The serving config passes
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320`, and the trace
kernel template shows a padded `8448u` shape. Rank-count and recv-count
balancing can reduce some bookkeeping and small-path imbalance, but it does
not proportionally shrink the main fused Mega-MoE kernel shape. The next
optimization target should be a Mega-MoE-specific objective based on padded
DeepGEMM blocks/shapes, or a way to reduce the padded-cap work itself, rather
than treating rank token count as the sole cost.
