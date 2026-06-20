# Waterfill H20 vs B200 MegaMoE Comparison

## Scope

This note compares the Waterfill behavior seen on H20 normal DeepEP MoE with
the B200 DSV4 MegaMoE path. The goal is to explain why H20 shows a clear
end-to-end gain while B200 MegaMoE only shows a small or noisy serving gain.

## H20 Normal DeepEP MoE

Run:

- Host: `10.6.131.5`
- Model: `/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3`
- Dataset: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/mmlu_bench_2k.json`
- Distribution: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/ep8_logical_count.pt`
- Backend: `--moe-a2a-backend deepep --deepep-mode normal`
- Parallelism: `tp=8`, `dp=8`, single-node EP8
- Output: `/lustre/raplab/client/xutingz/workspace/bench/waterfill/h20_node0_v3_ep8_tput_20260616_165024`

Longer serving throughput run:

| Case | Measurement rounds | Mean tok/s | Trimmed mean tok/s |
| --- | --- | ---: | ---: |
| baseline | `18117, 18035, 18140, 18084` | 18094 | 18101 |
| waterfill | `19296, 19068, 19126, 19322` | 19203 | 19211 |

Mean speedup: `+6.13%`.

Historical 2-node EP16 reproduction docs on the same H20 setup report
approximately `+3%` to `+4.4%` for the PR19290 Method B workload. The single
node EP8 result is higher, but directionally consistent: Waterfill produces a
stable visible e2e gain on normal DeepEP MoE.

Short MMLU profiler run with shared-expert fusion enabled:

- Output:
  `/lustre/raplab/client/xutingz/workspace/bench/waterfill/h20_mmlu_profile_fusion_wf_20260617_120551`
- Workload: same MMLU 2k source, 256 profiled prompts, concurrency 128,
  profile window 8 steps.

| Metric | fusion | fusion + Waterfill | Speedup |
| --- | ---: | ---: | ---: |
| input tok/s | `10324.57` | `10923.70` | `+5.80%` |
| max-rank DeepEP MoE path | `6093.11 ms` | `5551.53 ms` | `+9.76%` |
| max-rank expert GEMM | `4652.76 ms` | `4462.37 ms` | `+4.27%` |
| max-rank dispatch | `450.51 ms` | `410.97 ms` | `+9.62%` |
| max-rank combine | `1232.40 ms` | `786.27 ms` | `+56.74%` |
| max-rank Waterfill overhead | `0.00 ms` | `9.59 ms` | overhead |

Balance evidence from H20 rank-load stats:

| Metric | Before / routed counts | Waterfill target/static load |
| --- | ---: | ---: |
| median max/min ratio | `2.37` to `2.57` | `1.05` to `1.06` |
| p95 max/min ratio | `4.50` to `5.19` | `1.62` |
| median CV | `0.235` to `0.255` | `0.018` to `0.019` |
| p95 CV | `0.404` to `0.437` | `0.140` |

The H20 data shows both large balance improvement and visible e2e improvement.
The profiler split is important: the expert GEMM speedup is only `+4.27%`,
while the whole DeepEP MoE path is `+9.76%` because dispatch and combine also
shrink. This is the main reason H20 can show a larger end-to-end gain than the
B200 Mega-MoE core-only numbers.

## B200 DSV4 MegaMoE

Setup:

- Model: `/home/scratch.xutingz_wwfo_2/model/DeepSeek-V4-Flash-Base`
- Backend: `--moe-a2a-backend megamoe`
- Parallelism: `tp=2`
- Cases:
  - `fused`: `--enforce-shared-experts-fusion`
  - `fused_waterfill`: `--enforce-shared-experts-fusion --enable-deepep-waterfill`

Serving e2e results:

| Workload | fused | fused + Waterfill | Speedup |
| --- | ---: | ---: | ---: |
| random 8K prefill, job `2639863` | 36061.30 tok/s | 36252.09 tok/s | `+0.53%` |
| MMLU combined, jobs `2640401/2640645/2653100` | 21421.87 tok/s | 21390.46 tok/s | `-0.15%` |
| MMLU full-prewarm combined, jobs `2656399/2656838` | 20969.65 tok/s | 21184.82 tok/s | `+1.03%` |

MegaMoE segment and balance:

| Metric | fused | fused + Waterfill | Change |
| --- | ---: | ---: | ---: |
| random 8K count ratio mean | 1.3987 | 1.0266 | much flatter |
| random 8K count abs diff mean | 8291.85 | 323.60 | much flatter |
| random 8K `pre_dispatch_to_fp8_fp4_ms` mean | 1.533350 | 1.504560 | `+1.9%` |
| MMLU count ratio mean | 1.5448 | 1.1470 | flatter |
| MMLU count abs diff mean | 2969.58 | 185.07 | much flatter |
| MMLU `pre_dispatch_to_fp8_fp4_ms` mean | 0.807407 | 0.745459 | `+7.7%` |

Clean paired profiler, job `2655735`:

| Metric | fused | fused + Waterfill | Delta |
| --- | ---: | ---: | ---: |
| combined CUDA kernel sum | 3094.10 ms | 3101.04 ms | `-0.22%` speedup |
| max-rank kernel span | 2102.41 ms | 2052.38 ms | `+2.44%` speedup |
| MegaMoE core | 1148.33 ms | 1101.10 ms | `+4.29%` speedup |
| Waterfill count | 0.00 ms | 13.99 ms | overhead |
| Waterfill expand | 0.00 ms | 8.51 ms | overhead |
| comm/allreduce | 391.03 ms | 409.76 ms | +18.73 ms |
| MHC/HC | 395.37 ms | 421.68 ms | +26.31 ms |

All-rank timing check, job `2656094`:

| Metric | fused | fused + Waterfill | Delta |
| --- | ---: | ---: | ---: |
| simple mean `pre_dispatch_to_fp8_fp4_ms` | 0.6859 | 0.6320 | `+7.9%` |
| token-weighted mean | 1.1757 | 1.1125 | `+5.4%` |
| paired rank-max sum | 1684.43 ms | 1521.49 ms | `+10.7%` |

Red32 static-placement diagnostic, job `2679486`:

- Cases: `fused`, `fused_waterfill`, `fused_waterfill_local`
- Placement: `ep_num_redundant_experts=32`, `ep_dispatch_algorithm=static`,
  MMLU logical-count placement file
- Timing: all-rank Mega-MoE timing on every MoE call
- Throughput: intentionally ignored because this was a short high-logging run

Simple all-rank timing:

| Metric | fused | Waterfill remote shared | Waterfill local shared |
| --- | ---: | ---: | ---: |
| token-weighted `pre_dispatch_to_fp8_fp4_ms` | 1.4298 | 1.4797 | 1.4229 |
| rank count ratio mean | 2.1382 | 1.2482 | 2.1152 |
| Mega-MoE core speedup vs fused | baseline | `-3.37%` | `+0.49%` |

Max-rank prefill groups with at least 1024 tokens:

| Metric | fused | Waterfill remote shared | Waterfill local shared |
| --- | ---: | ---: | ---: |
| max-rank `pre_dispatch_to_fp8_fp4_ms` mean | 1.4260 | 1.4786 | 1.4248 |
| rank count ratio mean | 2.2908 | 1.2142 | 2.2325 |
| Mega-MoE core speedup vs fused | baseline | `-3.68%` | `+0.08%` |

This red32 run refines the earlier no-red conclusion. Waterfill is still
active and balances the remote-shared case, but for B200 Mega-MoE that balance
does not reduce the critical DeepGEMM fused-core interval. Forcing shared
experts to remain local removes the core slowdown, but also removes almost all
of the balance improvement.

Follow-up experiment:

The current B200-specific hypothesis is that remote shared assignment is only
profitable for Mega-MoE when the target rank is already present in the token's
routed TopK ranks. A new guarded knob keeps PR25391's default behavior unless
explicitly changed:

```bash
SGLANG_WATERFILL_STATIC_ALLOW_ALL_RANKS=0
```

With this set, static Waterfill restricts shared targets to routed ranks plus
the source rank. The B200 EP2 implementation uses a specialized Triton expand
kernel for this restricted case. The short diagnostic job `2680865` compares
`fused_waterfill` and `fused_waterfill_routed`, and logs `shared_remote` plus
`shared_remote_new_rank` to test whether extra remote token/rank pairs explain
the Mega-MoE core regression. An earlier submission, `2680742`, was canceled
because it inherited the sbatch script defaults instead of the intended
environment.

Job `2680865` completed and rejects the simple version of that hypothesis:

| Metric | Waterfill remote shared | Waterfill routed-rank-only |
| --- | ---: | ---: |
| token-weighted `pre_dispatch_to_fp8_fp4_ms` | `1.7766` | `1.7853` |
| all-call rank ratio mean | `1.3834` | `1.3758` |
| Waterfill after-ratio mean | `1.1457` | `1.1745` |
| `shared_remote_new_rank_mean` | `349.4` | `0.0` |

Restricting shared assignment to routed ranks successfully removes new remote
rank pairs, but it does not make the Mega-MoE core faster. The likely issue is
broader than `shared_remote_new_rank`: the Waterfill rank-count objective is
not yet modeling Mega-MoE's real grouped-kernel cost, including remote shared
token volume, active expert shape, locality, and synchronization.

Local-preference sweep:

Job `2681601` then swept static Waterfill's local preference on B200 red32
Mega-MoE (`11/10`, `2/1`, `4/1`, `8/1`). This job enabled all-rank per-call
timing and did not fix the request sampling seed, so serving throughput is only
diagnostic. The shape/timing trend is clear:

| Metric | default `11/10` | `8/1` |
| --- | ---: | ---: |
| all-call token-weighted `pre_dispatch_to_fp8_fp4_ms` | `1.7623` | `1.7239` |
| max-rank prefill `pre_dispatch_to_fp8_fp4_ms` mean | `2.0285` | `1.9481` |
| max-rank prefill rank-ratio mean | `1.2316` | `1.2699` |
| max-rank prefill `remote_shared_entries` mean | `6217.6` | `5904.7` |
| max-rank prefill `max_expert_tokens` mean | `6217.6` | `5918.6` |

The core time improves only when the giant shared-expert bucket shrinks, and
the improvement trades off against worse rank balance. In max-rank prefill
groups, corr(`pre_dispatch_to_fp8_fp4_ms`, `max_expert_tokens`) rises from
`0.51` at default to `0.82` at `8/1`, while the earlier
`shared_remote_new_rank` hypothesis was much weaker. This means the B200 issue
is expert-level grouped-kernel shape, not merely "new remote rank" fanout.

Fixed-seed confirmation:

Job `2682308` repeated default `11/10` and `8/1` with `TPUT_SEED=20260617`.
The run enabled all-rank timing on every MoE call, so throughput is only a
debug signal. The max-rank prefill result is:

| Metric | default `11/10` | `8/1` |
| --- | ---: | ---: |
| `pre_dispatch_to_fp8_fp4_ms` mean | `1.9668` | `1.9327` |
| rank-ratio mean | `1.2333` | `1.2753` |
| `remote_shared_entries` mean | `6078.5` | `6046.3` |
| `max_expert_tokens` mean | `6078.5` | `6048.8` |

The fixed-seed run only recovers `~1.8%` of max-rank prefill core time while
making balance worse. The critical B200 shape remains the `~6K`
shared-expert bucket, which is why this path does not convert rank balance
into H20-like MoE span speedup.

## B200 Cap-Bucket Rerun, DeepSeek-V4 Flash MMLU

After adding Mega-MoE cap buckets, the first useful failure was an EP-rank cap
divergence: one rank could select the `4096` bucket while another fell back to
`8320` due to a local free-HBM guard, leading to a DeepGEMM NVLink barrier
timeout. The current code fixes this by all-reducing the token cap requirement
and the "needs new bucket" decision across the Mega-MoE EP group. The free-HBM
guard now applies only when a smaller bucket needs to be created; if the bucket
already exists, all ranks keep using it.

Common setup for the rerun:

- Model: `/home/scratch.xutingz_wwfo_2/model/DeepSeek-V4-Flash`
- Dataset: `/home/scratch.xutingz_wwfo_2/bench/waterfill/mmlu_bench_2k.json`
- Backend: `--moe-a2a-backend megamoe`, `tp=2`
- Shared expert fusion enabled in both cases
- Cap buckets: `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS=4096:8320`
- Compared cases:
  - `fused`
  - `fused_waterfill_source_cost4096`

Correctness of the bucket path:

- Job `2699359` completed a small fused smoke test with synchronized `cap=4096`
  bucket creation and no NVLink timeout.
- Job `2699457` then completed a paired profile run. Both `fused` and
  `fused_waterfill_source_cost4096` created `cap=4096 experts=258 topk=7` on
  TP0 and TP1, with no fallback, CUDA error, or barrier timeout.

Paired profiler job `2699457`, `sample=128`, `concurrency=64`,
`PROFILE_STAGES=prefill`:

| Metric | fused | Waterfill source-cost4096 | Speedup |
| --- | ---: | ---: | ---: |
| profile round tput | 1665 tok/s | 1674 tok/s | `+0.54%` |
| EXTEND max-rank Mega-MoE span | 275.596 ms | 258.177 ms | `+6.75%` |
| EXTEND max-rank Mega-MoE impl | 243.551 ms | 236.531 ms | `+2.97%` |
| main `8448` shape span | 239.394 ms | 231.745 ms | `+3.30%` |
| EXTEND full GPU span | 894.620 ms | 837.109 ms | `+6.87%` |

The full-span speedup is inflated by the first profiled step:

| Step set | fused full span | Waterfill full span | fused MoE span | Waterfill MoE span | Target from MoE delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| all 5 EXTEND steps | 894.620 ms | 837.109 ms | 275.596 ms | 258.177 ms | `+1.99%` |
| steady 4 `8448` steps only | 685.335 ms | 673.265 ms | 239.394 ms | 231.745 ms | `+1.13%` |

So the clean expectation from the trace is not `30% * 5-8%`. For the steady
main shape, Waterfill gives about `3.3%` Mega-MoE span speedup, and the
trace-derived e2e target is only about `+1.1%`.

No-profile MMLU throughput:

| Job | Sample | Concurrency | Rounds | fused | Waterfill source-cost4096 | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `2699632` | 128 | 64 | 3 | 3802.3 tok/s | 3801.3 tok/s | `-0.03%` |
| `2699741` | 512 | 64 | 5 | 7330.4 tok/s | 7348.8 tok/s | `+0.25%` |
| `2699741` trimmed | 512 | 64 | 5 | 7353.7 tok/s | 7373.7 tok/s | `+0.27%` |
| `2699846` | 512 | 256 | 3 | 7015.3 tok/s | 7023.3 tok/s | `+0.11%` |

Increasing client concurrency from 64 to 256 did not materially change the
round trajectory, so the missing e2e gain is not a simple client-side
concurrency cap.

Diagnostic timing job `2699995`, `sample=256`, `concurrency=256`,
`SGLANG_MEGA_MOE_LOG_TIMING_INTERVAL=50`,
`SGLANG_WATERFILL_LOG_STATS_INTERVAL=50`:

| Metric | fused | Waterfill source-cost4096 | Change |
| --- | ---: | ---: | ---: |
| tput | 6928 tok/s | 6949 tok/s | `+0.30%` |
| token-weighted `pre_dispatch_to_fp8_fp4_ms` | 1.4232 ms | 1.3858 ms | `+2.69%` |
| count ratio mean | 1.4383 | 1.3297 | `-7.55%` |
| count ratio p95 | 1.8581 | 1.6250 | `-12.54%` |
| max local expert tokens mean | 5334.8 | 4804.2 | `-9.95%` |
| max local 64-blocks mean | 83.90 | 75.59 | `-9.90%` |
| remote shared entries mean | 0.0 | 829.1 | +829.1 |
| shared remote new-rank mean | 0.0 | 20.5 | +20.5 |

This confirms Waterfill is working and makes Mega-MoE's local expert shape
smaller. But the internal Mega-MoE timing gain in the clean no-profile serving
path is about `2.7%`, not `5-8%`. Multiplying the observed e2e gain by this
internal timing gain implies the effective request-wall Mega-MoE share is only
around `10-12%` for this MMLU serving workload. The earlier `~30%` share was a
prefill GPU-window metric, not a full client-observed serving-wall metric.

Updated clean three-way MMLU throughput:

Job `2704268` reran the same MMLU 2k source with the clean serving setup:

- `sample=1000`, `concurrency=128`, `warmup=4`, `measure=8`
- `MEM_FRACTION_STATIC=0.90` to avoid decode OOM on the current B200 nodes
- `MEGA_MOE_CAP_BUCKETS=4096:8320`
- `SGLANG_WATERFILL_LOG_STATS_INTERVAL=0`
- `SGLANG_MEGA_MOE_LOG_TIMING_INTERVAL=0`
- same job and node for all three cases: `umbriel-b200-092`

All three cases created the synchronized `4096` runtime Mega-MoE bucket on both
TP ranks, so the cap-bucket path was active:

```text
Using DeepGEMM Mega-MoE cap buckets: [4096, 8320].
Creating DeepGEMM Mega-MoE symmetric buffer cap=4096 experts=258 topk=7 hidden=4096 intermediate=2048.
```

| Case | Measurement rounds | Mean tok/s | Trimmed mean tok/s | Speedup vs fused |
| --- | --- | ---: | ---: | ---: |
| fused | `14663, 35420, 35258, 35286, 17963, 35331, 19044, 19321` | 26535.75 | 27033.83 | baseline |
| fused + Waterfill | `14773, 35868, 35615, 35730, 18112, 35788, 19172, 19487` | 26818.13 | 27317.33 | `+1.05%` |
| fused + Waterfill + fused pre-dispatch | `14752, 35735, 35601, 35665, 18095, 35777, 19158, 19485` | 26783.50 | 27289.83 | `+0.95%` |

The ordinary Waterfill path is the current best B200 MMLU setting. It is
`+1.05%` trimmed / `+1.06%` mean over fused baseline, and it is `+0.10%`
trimmed faster than the experimental fused pre-dispatch path. This is now
aligned with the PR25391 MMLU-scale expectation; the earlier `0.1-0.3%`
serving results were from smaller/debug runs and are no longer the best
estimate of clean MMLU behavior.

Current conclusion for B200 Mega-MoE:

- The cap-bucket bug is fixed; `4096` buckets work without EP divergence.
- Waterfill improves rank/expert balance and produces a real Mega-MoE internal
  speedup.
- Clean B200 MMLU e2e throughput now shows the expected `~+1%` gain when using
  ordinary Waterfill with synchronized cap buckets.
- The fused Waterfill pre-dispatch path is correctness-valid but not faster in
  serving throughput, so it should remain experimental/off by default.

Clean log-ready MMLU diagnostic:

Job `2695792` reran the B200 red32 static MMLU 2k slice with:

- `SGLANG_WATERFILL_LOG_STATS_INTERVAL=0`
- `SGLANG_WATERFILL_REUSE_TOPK_BUFFER=1`
- log-based server readiness plus DeepGEMM quiet wait, so `/health` polling did
  not interact with DeepGEMM JIT startup
- all-rank Mega-MoE timing enabled for diagnosis, so serving throughput is only
  directional

This run corrects an earlier measurement issue: enabling Waterfill stats every
layer inflated the materialized Waterfill setup path and made even
`remote_cost=16384` look much slower than baseline. With stats disabled,
`remote_cost=16384` matches the baseline token-weighted Mega-MoE span.

Max-rank prefill groups with at least 1024 tokens:

| Metric | fused | default Waterfill | `remote_cost=16384` | force-local shared |
| --- | ---: | ---: | ---: | ---: |
| serving tput, 64-prompt debug run | 1085 | 1089 | 1089 | 1089 |
| rank-global count ratio | 1.0760 | 1.0767 | 1.0759 | 1.0759 |
| max expert tokens | 5959.1 | 3702.8 | 5949.5 | 5950.9 |
| max local expert tokens | 5959.1 | 3348.7 | 5949.5 | 5950.9 |
| remote shared entries | 0.0 | 3231.9 | 0.0 | 0.0 |
| shared source edges | 2.00 | 3.91 | 2.00 | 2.00 |
| max-rank `pre_dispatch_to_fp8_fp4_ms` mean | 1.3563 | 1.3886 | 1.3899 | 1.3714 |
| max-rank total timing mean | 1.5032 | 1.5681 | 1.6016 | 1.4904 |

The clean result changes the diagnosis:

- The old `remote_cost=16384` slowdown was mostly measurement contamination
  from `WATERFILL_STATS=1`, not proof that the materialized TopK path alone is
  inherently slow.
- Default Waterfill still improves the largest expert bucket, but it creates
  about `3.2K` remote shared entries and nearly doubles the shared source-edge
  fanout. In this clean run, `remote_shared_entries` correlates with max-rank
  pre span at about `0.71`.
- Force-local shared removes the fanout and gives the best max-rank total
  timing, but it also removes the expert-bucket balancing. Therefore the
  remaining B200 issue is not Waterfill correctness; it is that Mega-MoE's
  remote shared path does not turn lower expert counts into lower wall-clock
  span.

Strategy sweep follow-up:

Jobs `2695999` and `2696152` tested routed-rank-only targets, explicit remote
cost, and extra shared replicas. The first sweep hit a port-limit failure when
the derived gRPC port exceeded `65535`; the completed cases are still valid.

| Case | tput | max-rank pre mean | max-rank total mean | remote shared | max expert | Note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fused baseline | 1085 | 1.3563 | 1.5032 | 0.0 | 5959.1 | reference from job `2695792` |
| default Waterfill | 1089 | 1.3886 | 1.5680 | 3231.9 | 3702.8 | balances expert bucket but slows span |
| routed-only | 1087 | 1.3854 | 1.5587 | 3036.6 | 3769.2 | removing new-rank fanout is not enough |
| routed + cost4096 | 1088 | 1.3783 | 1.5351 | 368.4 | 5927.8 | loses most balance, still slower |
| routed + cost8192 | 1084 | 1.3741 | 1.5640 | 11.1 | 5959.1 | almost no remote shared, still slower |
| shared replicas 2 | 1024 | 5.6247 | 6.1365 | 3231.9 | 2705.8 | large outliers; not viable |
| shared replicas 4 | 1011 | 5.3942 | 5.9029 | 3231.9 | 2521.1 | large outliers; not viable |

The routed/cost sweep shows that simply charging or forbidding remote shared
does not recover a useful speedup. Once the cost is high enough to avoid remote
shared, the Waterfill result is essentially baseline placement plus extra
setup. The replica sweep reduces the max expert bucket but introduces very
large Mega-MoE outliers and hurts serving throughput, so it is not a practical
fix in this path.

Historical red32 conclusion: on B200 Mega-MoE with redundant static placement,
the Waterfill objective needs a Mega-MoE-specific cost model that accounts for
grouped-kernel shape and shared remote fanout. Pure rank-count balance,
routed-rank restriction, static remote cost, and extra shared replicas did not
reproduce the H20 DeepEP gain in those debug runs. This red32 diagnosis is
separate from the later clean no-red MMLU result, where ordinary Waterfill now
shows the expected `~+1%` e2e gain.

## Interpretation

Waterfill is active on both systems. It improves load balance on H20 and B200
MegaMoE. The difference is where, and whether, the balanced load converts into
time savings.

The latest B200 no-red MMLU result (`2704268`) now confirms a clean serving
gain of `+1.05%` with ordinary Waterfill. The remaining H20-vs-B200 difference
is therefore not whether B200 works at all, but why B200 Mega-MoE converts the
balance improvement into a PR25391-scale `~+1%` serving gain instead of the
larger H20 DeepEP-path gain.

On H20 normal DeepEP MoE, Waterfill affects the larger critical path:

```text
gate/topk -> dispatcher.dispatch -> run_moe_core -> dispatcher.combine
```

The rank-load improvement reduces dispatch, expert compute, combine, and their
synchronization/overlap windows. In the H20 MMLU profiler, the expert GEMM
improved by `+4.27%`, but the full max-rank DeepEP MoE path improved by
`+9.76%` because combine dropped from `1232.40 ms` to `786.27 ms`.

On B200 MegaMoE, the measured path is narrower:

```text
gate/topk -> mega_moe_pre_dispatch -> deep_gemm.fp8_fp4_mega_moe -> TP allreduce
```

In the earlier no-red trace, Waterfill reduced the
`deep_gemm.fp8_fp4_mega_moe` interval by a few percent. In the red32
remote-shared run, however, the same rank-count balance objective did not make
the DeepGEMM Mega-MoE core faster. It improved count ratio from `2.29` to
`1.21`, but max-rank core mean regressed from `1.4260 ms` to `1.4786 ms`.

The B200 path also does not have a separate DeepEP combine stage to shrink, and
it does not remove fixed gate/topk, pre-dispatch packing, result handling, TP
allreduce, attention, scheduler, HTTP/client, and decode overheads.

The B200 profiler makes the accounting explicit:

- MegaMoE core is `1148.33 ms` out of `3094.10 ms`, about `37%` of summed CUDA
  kernel time.
- The clean no-red MegaMoE-core saving is `4.29%`, so the gross total-kernel
  upside is only about `1.6%`.
- The red32 remote-shared Mega-MoE core did not reproduce that saving; it was
  about `3.7%` slower on max-rank prefill groups despite much better balance.
- Restricting shared targets to routed ranks removed
  `shared_remote_new_rank` but remained flat/slightly slower, so the core
  regression is not explained solely by extra remote token/rank pairs.
- Waterfill count plus expand adds `22.5 ms`, about `0.7%` of the summed CUDA
  kernel time.
- comm/allreduce and MHC/HC moved by `18.7 ms` and `26.3 ms` in the same
  profile window, which is already comparable to the net Waterfill saving.
- Serving repeat variance and order effects are around `1%` or higher in the
  smaller/debug runs, so those runs understated the expected e2e gain. The
  larger clean paired MMLU run `2704268` recovers the expected `+1.05%`.

Therefore, the B200 MegaMoE result does not look like a disabled or broken
Waterfill path. It now looks like two separate effects:

1. On no-red / clean MMLU placement, Waterfill improves the Mega-MoE
   sub-kernel and converts into the expected `~+1%` serving e2e gain. The
   optimized segment is still much narrower than H20's full DeepEP
   dispatch/compute/combine path, so the gain is PR25391-scale rather than
   H20's larger MoE-path gain.
2. On red32 B200 Mega-MoE, the balance objective itself is not enough: better
   rank counts do not currently become a faster fused DeepGEMM Mega-MoE core,
   even after removing shared assignments to new remote ranks. The newer
   local-preference sweep shows that the fused core is instead strongly tied
   to the large shared-expert token bucket.

Follow-up change tested: decouple the logical shared TopK column from the
physical shared expert slots with
`SGLANG_WATERFILL_SHARED_REPLICAS_PER_RANK`. This keeps the shared expert
mathematically identical by loading the same checkpoint weight into each
replica, but lets Waterfill split the selected rank's shared tokens across
multiple physical shared buckets.

Job `2683760` ran a fixed-seed smoke comparing `fused_waterfill` and
`fused_waterfill_rep2` on the same MMLU 2k source and red32 static placement.
The first rep2 requests hit one-time DeepGEMM JIT outliers (`pre_dispatch` up
to `~2.4 s`), so the steady-state comparison below excludes paired prefill
groups with rank-max `pre_dispatch_to_fp8_fp4_ms >= 5`.

| Metric | rep1 | rep2 | Change |
| --- | ---: | ---: | ---: |
| `shared_replicas_per_rank` | `1` | `2` | active |
| prefill rank-max `max_expert_tokens` mean | `6138.36` | `3403.76` | `0.55x` |
| prefill rank-max `max_local_expert_tokens` mean | `2599.48` | `2585.18` | `0.99x` |
| prefill rank-max `active_local_experts` mean | `134.57` | `134.91` | `+0.25%` |
| prefill global incoming count ratio mean | `1.0587` | `1.0587` | unchanged |
| steady prefill rank-max `pre_dispatch_to_fp8_fp4_ms` | `1.4596` | `1.4862` | `-1.8%` |

This rejects the narrow "single giant shared bucket" explanation. Rep2 does
split the per-source global shared bucket, but it does not reduce the incoming
rank max, and it barely changes the rank-local grouped-GEMM shape. It also
slightly increases the number of active experts. For this Mega-MoE path,
splitting work into more expert groups on the same destination rank is not the
same as reducing the slow rank's total local work.

Recomputed global count evidence also explains why the earlier `+2.37%` MoE
span speedup was low. In red32 job `2679486`, the per-source rank ratio looked
large (`2.29 -> 1.21`), but after summing counts across TP/source ranks, the
slow-rank incoming max was already flat:

| Metric | fused | fused + Waterfill |
| --- | ---: | ---: |
| prefill global incoming count ratio mean | `1.0859` | `1.0647` |
| prefill global incoming count max mean | `45461.53` | `45454.91` |
| prefill rank-max `pre_dispatch_to_fp8_fp4_ms` | `1.4160` | `1.4752` |

Waterfill improves the ratio mostly by lifting the underloaded rank; it does
not materially reduce the slow rank's total incoming Mega-MoE work on this
B200 TP2 workload. That is why it cannot reproduce the H20 DeepEP result, where
Waterfill reduced the full dispatch/compute/combine critical path.

Dynamic global-objective diagnostic, job `2687128`, fixes that specific
static-objective issue by all-reducing routed counts first and then balancing
the shared expert against the true global incoming load. The debug run used the
same fixed-seed MMLU source and red32 static placement, comparing static
`fused_waterfill` with `fused_waterfill_dynamic`.

| Metric | static Waterfill | dynamic/global Waterfill | Change |
| --- | ---: | ---: | ---: |
| prefill global incoming count ratio mean | `1.0582` | `1.0014` | nearly flat |
| prefill global incoming count max mean | `42845.22` | `41673.45` | `-2.7%` |
| prefill rank-max `pre_dispatch_to_fp8_fp4_ms` sum | `817.95 ms` | `795.63 ms` | `+2.81%` |
| prefill rank-max `topk`/setup sum | `389.30 ms` | `517.06 ms` | `-24.7%` |
| prefill rank-max total logged sum | `1208.59 ms` | `1312.90 ms` | `-7.95%` |
| prefill rank-max `max_expert_tokens` mean | `5949.71` | `3984.88` | `0.67x` |
| all-call token-weighted `pre_dispatch_to_fp8_fp4_ms` | `1.5113` | `1.4452` | `+4.57%` |

This resolves the immediate question about the low B200 MoE span speedup: even
when the global incoming rank load is almost perfectly balanced, the slow-rank
Mega-MoE core improves by only a few percent and the dynamic all-reduce/setup
cost is larger than that saving in this debug path. The B200 fused Mega-MoE
core gain is comparable to H20's expert-GEMM-only gain (`+4.27%`), but H20's
reported MoE-path gain is much larger because normal DeepEP also shrinks
dispatch and combine, especially combine.

Latest fixed-seed local-preference recheck, job `2688305`, strengthens the
same conclusion. It used the corrected static dispatch-map rank-load target and
swept stronger local preferences (`8/1`, `16/1`, `32/1`) to see whether
reducing remote shared placement converts balance into B200 Mega-MoE speed.
The result is no:

| Case | tok/s | prefill rank-max `pre_dispatch_to_fp8_fp4_ms` sum | Speedup vs fused | global incoming max | rank ratio | remote shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fused | `1086` | `742.87 ms` | baseline | `43193.80` | `2.2344` | `0.00` |
| Waterfill `11/10` | `1078` | `782.50 ms` | `-5.06%` | `43149.22` | `1.6236` | `3228.68` |
| Waterfill `8/1` | `1063` | `780.04 ms` | `-4.77%` | `43207.10` | `2.0271` | `1113.81` |
| Waterfill `16/1` | `1083` | `785.40 ms` | `-5.41%` | `43158.37` | `2.0993` | `764.76` |
| Waterfill `32/1` | `1082` | `772.04 ms` | `-3.78%` | `43241.38` | `2.1458` | `566.69` |

The local-pref sweep reduces remote shared entries, but it does not reduce the
slow-rank global incoming max; it only changes placement shape and relaxes the
rank-count balance. That is why the B200 red32 Mega-MoE path remains below the
H20 result: H20's `+9.76%` was a full DeepEP MoE-path gain
(dispatch + expert GEMM + combine), while B200's fused Mega-MoE path mostly
exposes the core grouped-GEMM/packing critical path and needs a
Mega-MoE-specific cost objective.

Latest destination/recv-rank block diagnostic, job `2697220`, tightens the
B200 explanation. The previous B200 diagnostics included source-local metrics
that made Waterfill look more useful than it is for Mega-MoE. This run logs
full per-expert counts on both TP/source ranks, sums them into the
destination/recv-rank view, and estimates local grouped-GEMM block work.

| Metric | B200 fused | B200 fused + Waterfill | Change |
| --- | ---: | ---: | ---: |
| source-local max expert tokens | `5949.54` | `3338.11` | lower |
| source-local `block_m=128` blocks | `302.75` | `278.94` | lower |
| recv-rank max tokens | `43122.98` | `43137.73` | flat/slightly higher |
| recv-rank max `block_m=128` blocks | `410.30` | `410.41` | flat/slightly higher |
| recv-rank block ratio, `block_m=128` | `1.0596` | `1.0599` | flat |
| recv active local experts | `135.16` | `135.16` | unchanged |
| max-rank core sum | `787.44 ms` | `778.80 ms` | `+1.11%` |
| max-rank topk/setup sum | `81.57 ms` | `216.89 ms` | overhead |

So the direct answer is: B200 Mega-MoE is not seeing the H20-like gain because
static Waterfill does not reduce the actual slow recv-rank grouped-kernel
block work. It mostly redistributes shared tokens in a way that improves
per-source/local-looking balance while leaving the destination aggregate
unchanged. H20 normal DeepEP still gets a clear gain because its Waterfill
benefit applies to the broader dispatch + expert GEMM + combine critical path;
B200 Mega-MoE only exposes the fused grouped-kernel span, and the current
rank-count objective is not aligned with that span.

## Latest B200 Update: Source-Aware + Remote-Cost Objective

A follow-up B200 run fixed the first objective mismatch by computing static
rank load per source rank:

```text
rank_load = current_local_routed_counts + offline_other_source_rank_load
```

This changes B200 Mega-MoE Waterfill from a source-local balance objective to a
destination/recv-rank objective. By itself it improved recv-rank balance but
still did not speed up the fused Mega-MoE core, because it increased remote
shared traffic.

Job `2697498`, fixed MMLU 2k slice:

| Case | tput | token-weighted Mega-MoE core | recv max blocks | recv ratio | remote shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| fused | `1084` | `1.4165 ms` | `410.30` | `1.0759` | `0.0` |
| old Waterfill | `1086` | `1.4269 ms` | `410.41` | `1.0765` | `3232.1` |
| source-aware Waterfill | `1086` | `1.4272 ms` | `403.13` | `1.0278` | `4466.5` |

The source-aware path lowered the slow recv-rank block count, but the extra
remote shared traffic offset the compute gain. A per-prefill regression showed
both `remote_shared_entries` and `shared_remote_new_rank` correlate positively
with B200 Mega-MoE core time.

Job `2697718` then swept a remote shared cost penalty while keeping
source-aware static load enabled:

| Case | tput | token-weighted core | prefill core-span speedup | recv max blocks | remote shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| fused | `1075` | `1.4149 ms` | baseline | `410.30` | `0.0` |
| source-aware cost4096 | `1083` | `1.3881 ms` | `+3.80%` | `402.09` | `3455.5` |
| source-aware cost8192 | `1083` | `1.3940 ms` | `+3.19%` | `408.78` | `534.4` |
| source-aware cost16384 | `1083` | `1.3964 ms` | `+2.93%` | `410.30` | `0.0` |

At this point, the all-rank log timing appeared to change the B200 conclusion:

- The earlier "Waterfill cannot make Mega-MoE faster" result appeared to be
  caused by the wrong static objective plus no remote shared penalty.
- With a Mega-MoE-aware objective, the B200 fused core span now moves in the
  expected direction: `+3.80%` on prefill span for `remote_cost=4096`.
- The end-to-end gain is still much smaller than H20 (`+0.74%` in this short
  run) because B200 Mega-MoE still pays Waterfill topk/materialization overhead
  and does not get H20 DeepEP's large dispatch/combine-path reduction.

So, before the clean profiler correction below, H20 and B200 appeared
consistent at the expert/core level: H20 expert GEMM speedup was `+4.27%`, and
B200 Mega-MoE source-aware cost4096 appeared to give about `+3.8%` prefill
core-span speedup. The remaining suspected gap was path composition: normal
DeepEP benefits in dispatch, expert GEMM, and combine, while B200 Mega-MoE
mainly benefits in the fused core and must still pay the Waterfill setup path.

## Correction From Clean B200 Profiles

Later clean serving/profile jobs revised the B200 conclusion. The `+3.8%`
number from job `2697718` was an all-rank log-timing signal and did not
reproduce as a stable profiled max-rank Mega-MoE span gain.

Clean serving job `2698077`, using the same fixed MMLU 2k slice and no
Waterfill/Mega-MoE debug logging:

| Case | Trimmed serving tput | vs fused |
| --- | ---: | ---: |
| fused | `5948.0` | baseline |
| old Waterfill | `5929.3` | `-0.31%` |
| source-aware cost4096 | `5949.0` | `+0.02%` |

GPU-only profile job `2698497` captured the full measured prefill window:

| Metric | fused | source-aware cost4096 | Change |
| --- | ---: | ---: | ---: |
| serving round tput | `13239` | `13801` | `+4.25%` |
| EXTEND max-rank Mega-MoE span | `824.24 ms` | `821.62 ms` | `+0.32%` |
| EXTEND full trace span | `2318.25 ms` | `2256.49 ms` | `+2.74%` |
| EXTEND GPU active time | `2152.91 ms` | `2154.62 ms` | `-0.08%` |

The latest evidence is that a single B200 serving round can look faster, but
the trace does not attribute that speedup to the Mega-MoE fused kernel. The
full trace span improved mostly through non-MoE idle/scheduling gaps, while
GPU active time was flat and the max-rank Mega-MoE span improved only
`+0.32%`.

So the H20 and B200 paths are not yet consistent at the full MoE-path level.
H20 normal DeepEP benefits across dispatch, expert GEMM, and combine. B200
Mega-MoE is dominated by a fixed-cap padded fused DeepGEMM path
(`MEGA_MOE_CAP=8320`, trace padded shape `8448u`), so rank-count balancing does
not automatically reduce the main kernel work in this source-aware
cost-penalty configuration.

## Latest B200 Update: Cap Buckets + Default Waterfill

The latest rerun changes the B200 conclusion again. After fixing Mega-MoE cap
bucket selection so all EP ranks synchronize the selected token cap and bucket
creation decision, the best B200 MMLU result is the default static Waterfill
policy, not the source-aware `remote_cost=4096` policy.

Common setup:

- Model: `/home/scratch.xutingz_wwfo_2/model/DeepSeek-V4-Flash`
- Dataset: `/home/scratch.xutingz_wwfo_2/bench/waterfill/mmlu_bench_2k.json`
- Backend: `--moe-a2a-backend megamoe`, `tp=2`
- Shared-expert fusion enabled in all cases
- Cap buckets:
  `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK_BUCKETS=4096:8320`

Debug/timing sweep, job `2700110`, `sample=256`, `concurrency=256`:

| Case | tput | token-weighted Mega-MoE core | Core speedup | rank ratio mean | p95 ratio | max local expert tokens | remote shared |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fused | `6927` | `1.4035 ms` | baseline | `1.4383` | `1.8581` | `5334.8` | `0.0` |
| default Waterfill | `6994` | `1.3520 ms` | `+3.81%` | `1.1191` | `1.4069` | `3287.6` | `2582.3` |
| one-way remote shared | `6981` | `1.3754 ms` | `+2.04%` | `1.1807` | `1.4069` | `3908.3` | `1981.7` |
| source-aware cost4096 | `6987` | `1.3616 ms` | `+3.08%` | `1.3297` | `1.6250` | `4804.2` | `829.1` |

This run explains why the previous B200 Mega-MoE span number was too low:
`source_cost4096` is too conservative on this no-red MMLU workload. It reduces
remote shared traffic, but leaves a much larger max local expert bucket. The
default Waterfill policy gives the best core timing because it most strongly
reduces the slow bucket.

No-logging serving confirmation, job `2700287`, `sample=512`,
`concurrency=256`, five rounds:

| Case | Rounds | Mean tok/s | Trimmed tok/s |
| --- | --- | ---: | ---: |
| fused | `10474, 5562, 6477, 9008, 6699` | `7644.0` | `7394.7` |
| default Waterfill | `10537, 5596, 6502, 9050, 6727` | `7682.4` | `7426.3` |

Mean serving speedup is `+0.50%`; trimmed speedup is `+0.43%`. Every paired
round is slightly faster with Waterfill, but the round-to-round serving
variance is much larger than the gain.

Profile trace job `2700382`, `sample=128`, `concurrency=64`, default
Waterfill:

| Metric | fused | default Waterfill | Change |
| --- | ---: | ---: | ---: |
| serving round tput | `4066` | `4092` | `+0.64%` |
| EXTEND max-rank Mega-MoE impl | `252.160 ms` | `241.669 ms` | `+4.34%` |
| EXTEND max-rank Mega-MoE span | `268.174 ms` | `257.146 ms` | `+4.29%` |
| EXTEND full trace span | `1412.631 ms` | `1394.923 ms` | `+1.27%` |
| EXTEND GPU active time | `1265.611 ms` | `1245.292 ms` | `+1.63%` |
| MoE span share in EXTEND trace | `18.98%` | `18.43%` | lower |

The trace-derived e2e target from the MoE span delta is `+0.787%`, and the
observed EXTEND full-span speedup is `+1.269%`. Decode is neutral/slightly
negative (`span_speedup=-0.39%`), which is expected because this optimization
is a prefill/MoE-bucket effect.

Revised B200 conclusion:

- B200 Mega-MoE is no longer a "no benefit" case. With cap-bucket
  synchronization and default Waterfill, the max-rank Mega-MoE span improves
  by `+4.29%`, which is essentially the same scale as H20's expert-GEMM-only
  gain (`+4.27%`).
- H20 still shows a much larger full MoE-path gain (`+9.76%`) because normal
  DeepEP also shrinks dispatch and combine, especially combine. B200
  Mega-MoE mostly exposes the fused core/packing span and has no comparable
  separate combine stage to shrink.
- The B200 no-logging serving gain is only `+0.43%` to `+0.50%` because the
  improved Mega-MoE interval is about `18%` to `19%` of the profiled EXTEND GPU
  window and only around `10%` to `12%` of the full client-observed serving
  wall in this MMLU workload.
- The earlier `+2.37%` low-span result was a measurement/configuration
  artifact of using the more conservative source/cost objective and the older
  cap-bucket path. The current best B200 policy is default Waterfill with
  synchronized cap buckets.

FP4 activation dispatch sanity check:

Jobs `2702839` and `2702950` tested
`SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1` as a possible B200 Mega-MoE
path difference. This is not a usable optimization in the current DSV4 Flash
server path.

| Job | Setup | Result |
| --- | --- | --- |
| `2702839` | `MEGA_MOE_CAP_BUCKETS=4096:8320`, no-waterfill first | crashed on first forward |
| `2702950` | `MEGA_MOE_CAP_BUCKETS=` empty, no-waterfill only | crashed on first forward |

Both failures have the same stack:

```text
deep_gemm.fp8_fp4_mega_moe(...)
tvm.error.InternalError: CUDA driver error ... CUDA_ERROR_INVALID_VALUE
```

This rules out the narrow "FP4 activation dispatch fixes the missing gain"
hypothesis. The failure also shows the crash is not caused by dynamic creation
of the smaller `cap=4096` bucket, because the no-bucket run fails in the same
DeepGEMM entrypoint. The current viable B200 path remains FP8 activation
pre-dispatch into `deep_gemm.fp8_fp4_mega_moe`, with synchronized cap buckets.

## PR25391 No-Red MMLU Recheck On B200

PR25391 reports several different numbers. The `+2.60%` no-red and `+4.31%`
red32 figures are from the H20 FP8 perf sanity workload. The MMLU line in that
PR is much smaller: no-red MMLU 1000 examples improves from `892.9` to `902.6`
output tok/s, or about `+1.09%`.

To match that MMLU-specific target more closely, job `2700515` reran B200
Mega-MoE with:

- Dataset: the fixed MMLU 2k slice,
  `/home/scratch.xutingz_wwfo_2/bench/waterfill/mmlu_bench_2k.json`
- `sample=1000`, `concurrency=128`
- `warmup=4`, `measure=8`
- no redundant experts, no static placement
- shared expert fusion enabled in both cases
- `MEGA_MOE_CAP_BUCKETS=4096:8320`
- Waterfill stats and Mega-MoE timing logs disabled

Results:

| Case | Measured rounds tok/s | Mean tok/s | Trimmed tok/s |
| --- | --- | ---: | ---: |
| fused | `19695, 19556, 19703, 12028, 12902, 8772, 19685, 19689` | `16503.75` | `17259.17` |
| fused + Waterfill | `19875, 19815, 19846, 12082, 12966, 8798, 19825, 19848` | `16631.88` | `17397.00` |

Speedup:

- Mean: `+0.776%`
- Trimmed mean: `+0.799%`
- Paired round speedups:
  `+0.914%, +1.324%, +0.726%, +0.449%, +0.496%, +0.296%, +0.711%, +0.808%`

This is a stable positive result: every paired measured round is faster with
Waterfill. It is also much closer to PR25391's MMLU-specific `+1.09%` number
than to the PR's H20 perf sanity `+2.60%` or red32 `+4.31%` numbers. Therefore,
the current B200 no-red MMLU result should not be treated as a missing 4% e2e
gain. A 4% target only makes sense under the corresponding red32/static or perf
sanity workload.

The remaining actionable gap is not "B200 no-red MMLU should be 4% e2e". It is
to run the matching red32/static placement or PR perf-sanity workload on B200
and compare against the corresponding PR25391 target.

## B200 Red32 Static MMLU Recheck

Job `2700837` reran the same MMLU 1000-example throughput test with redundant
experts and the historical static placement file:

- `EP_NUM_REDUNDANT_EXPERTS=32`
- `EP_DISPATCH_ALGORITHM=static`
- `INIT_EXPERT_LOCATION=/home/scratch.xutingz_wwfo_2/bench/waterfill/v4_mmlu_expert_dist/v4_ep16_mmlu_stat_approx_layerctx_20260514_171736_logical_count.pt`
- `CASE_ORDER=fused:fused_waterfill`
- `sample=1000`, `concurrency=128`, `warmup=4`, `measure=8`
- `MEGA_MOE_CAP_BUCKETS=4096:8320`

Results:

| Case | Measured rounds tok/s | Mean tok/s | Trimmed tok/s |
| --- | --- | ---: | ---: |
| red32 static fused | `35038, 34672, 34857, 16376, 18042, 10877, 34810, 34845` | `27439.63` | `28933.67` |
| red32 static fused + Waterfill | `35083, 34855, 34889, 16401, 18059, 10880, 34883, 34931` | `27497.63` | `29003.00` |

Speedup:

- Mean: `+0.211%`
- Trimmed mean: `+0.240%`
- Paired round speedups:
  `+0.128%, +0.528%, +0.092%, +0.153%, +0.094%, +0.028%, +0.210%, +0.247%`

This result is also stable positive across all paired rounds, but the
incremental Waterfill gain is smaller than no-red MMLU. The main reason is that
red32 static placement already removes much of the MMLU load imbalance before
Waterfill runs: baseline red32 static fused trimmed throughput is `28933.67`
tok/s, compared with no-red fused trimmed `17259.17` tok/s in job `2700515`.
At that point Waterfill only has a small residual imbalance to improve.

The PR25391 red32 `+4.31%` target therefore still should not be applied to this
MMLU throughput harness directly. On this MMLU slice, red32 static placement is
the dominant optimization, and Waterfill is an incremental `+0.24%` on top of
that. To chase PR25391's red32 `+4.31%`, the next required comparison is the
same FP8 perf sanity workload used in the PR, not this MMLU client workload.

## PR25391 Random Sanity Recheck On B200

The first B200 PR-style random sanity submission, job `2701096`, is invalid for
cap-bucket analysis. It used `MAX_RUNNING_REQUESTS=2048`, leaving only
`3.66 GB` local free HBM on one rank, so the `4096` bucket creation path was
skipped by the free-HBM guard. That run was canceled and is not used below.

The valid reruns use:

- Model: `/home/scratch.xutingz_wwfo_2/model/DeepSeek-V4-Flash`
- Workload: random `8192 -> 1`, `512` prompts, concurrency `128`
- `MAX_RUNNING_REQUESTS=256`
- `MEGA_MOE_CAP_BUCKETS=4096:8320`
- shared expert fusion enabled in both cases
- Waterfill/Mega-MoE debug timing disabled for throughput

No-red random sanity, job `2701169`:

| Case | Rounds tok/s | Mean tok/s | Trimmed tok/s | Drop-first tok/s |
| --- | --- | ---: | ---: | ---: |
| fused | `34810, 36152, 36135, 36135, 36179, 36153, 36156, 36183` | `35987.7` | `36151.5` | `36156.0` |
| fused + Waterfill | `35312, 36330, 36351, 36374, 36383, 36325, 36379, 36325` | `36222.3` | `36347.2` | `36352.4` |

Speedup:

- Mean: `+0.652%`
- Trimmed: `+0.541%`
- Drop-first: `+0.543%`
- Every paired round is positive.
- Both cases created synchronized `cap=4096` and `cap=8320` buffers; no
  cap-bucket skip or scheduler exception was observed.

Red32/static random sanity, job `2701344`:

| Case | Rounds tok/s | Mean tok/s | Trimmed tok/s | Drop-first tok/s |
| --- | --- | ---: | ---: | ---: |
| red32 static fused | `33425, 35878, 35904, 35967, 35929, 35921, 35935, 35965` | `35615.4` | `35922.0` | `35928.4` |
| red32 static fused + Waterfill | `34911, 35927, 35928, 35949, 35958, 35997, 35998, 35982` | `35831.2` | `35956.8` | `35962.6` |

Speedup:

- Mean: `+0.606%`, inflated by the cold first round.
- Trimmed: `+0.097%`
- Drop-first: `+0.095%`
- One paired steady round is slightly negative.
- Both cases created synchronized `cap=4096` and `cap=8320` buffers; no
  cap-bucket skip or scheduler exception was observed.

These B200 numbers are much smaller than PR25391's H20 random sanity numbers
(`+2.60%` no-red and `+4.31%` red32). They are also consistent with the MMLU
finding: on B200 Mega-MoE, Waterfill improves the rank/expert distribution but
does not automatically move the main 8K source-token prefill into a smaller
DeepGEMM cap/shape.

Important correction on cap interpretation: seeing a `cap=4096` buffer in the
server log does not mean the main random 8K prefill used that cap. The
Mega-MoE pre-dispatch buffer is shaped as `[padded_max, hidden]`, and the
kernel wrapper enforces `num_tokens <= padded_max`. The cap selector is
therefore constrained by the source-rank token count before Waterfill, not by
the post-Waterfill destination rank load. For 8K source batches, the main
prefill shape still needs the `8320` bucket; `4096` creation comes from smaller
warmup/decode/chunked shapes. Making Waterfill choose cap from the post-balance
max-rank load would require a deeper pre-dispatch/buffer-layout redesign, not
just a different threshold.

## B200 Random Sanity Timing Explanation

Jobs `2701609` and `2701611` reran the same random `8192 -> 1` workload with
all-rank `MEGA_MOE_TIMING` and Waterfill stats enabled. Throughput from these
jobs is intentionally ignored because per-layer logging adds large topk/setup
overhead. The useful signal is the paired max-rank
`pre_dispatch_to_fp8_fp4_ms` span.

No-red timing job `2701609`:

| Metric, prefill groups >=1024 tokens | fused | fused + Waterfill | Change |
| --- | ---: | ---: | ---: |
| paired groups | `3010` | `3010` | same |
| max-rank `pre_dispatch_to_fp8_fp4_ms` sum | `5384.24 ms` | `5278.87 ms` | `+2.00%` |
| max-rank `pre_dispatch_to_fp8_fp4_ms` mean | `1.7888 ms` | `1.7538 ms` | `+2.00%` |
| global incoming count ratio mean | `1.1548` | `1.0089` | flatter |
| global incoming count diff mean | `7458.4` | `446.0` | much flatter |
| max local expert tokens mean | `7776.6` | `5779.2` | `-25.7%` |
| local expert `block_m=128` blocks mean | `330.13` | `288.90` | `-12.5%` |
| remote shared entries mean | `0.0` | `5519.4` | added |

This is the clearest explanation for the low no-red B200 MoE span speedup:
Waterfill is absolutely balancing the load and shrinking the largest local
expert bucket, but the B200 Mega-MoE fused span only improves by about `2%`.
The fused path is not as sensitive to rank-count ratio as the H20 normal
DeepEP full path. The main 8K source-token prefill also remains constrained by
the `8320` cap bucket, so Waterfill is not getting an additional padded-shape
reduction in this path.

Red32/static timing job `2701611`:

| Metric, prefill groups >=1024 tokens | red32 static fused | red32 static + Waterfill | Change |
| --- | ---: | ---: | ---: |
| paired groups | `3010` | `3010` | same |
| max-rank `pre_dispatch_to_fp8_fp4_ms` sum | `5216.97 ms` | `5292.66 ms` | `-1.43%` |
| max-rank `pre_dispatch_to_fp8_fp4_ms` mean | `1.7332 ms` | `1.7584 ms` | `-1.43%` |
| Waterfill before/after rank ratio mean | `1.0394` | `1.1966` | worse |
| global incoming count ratio mean | `1.1465` | `1.1402` | nearly flat |
| max local expert tokens mean | `7776.6` | `4194.9` | lower |
| local expert `block_m=128` blocks mean | `349.22` | `318.21` | lower |
| remote shared entries mean | `0.0` | `4193.3` | added |

For red32/static, the MMLU-derived static placement has already made the
rank-load objective nearly flat before Waterfill. Waterfill reduces the largest
expert bucket but introduces remote shared traffic and does not reduce the
true slow-rank span; in this random sanity run, it makes the measured
Mega-MoE core slightly slower.

The B200-vs-H20 difference is therefore not that Waterfill is disabled. It is:

1. H20 normal DeepEP benefits across dispatch, expert GEMM, and combine.
   Its full max-rank DeepEP MoE path improved by `+9.76%`.
2. B200 Mega-MoE mostly exposes the fused packed/core interval. On the
   comparable random no-red workload, even strong balance improvement converts
   to only `+2.00%` max-rank core speedup.
3. On red32/static, the remaining Waterfill opportunity is largely gone or
   offset by remote shared fanout. The measured max-rank core span is
   `-1.43%`.
4. Because the profiled B200 Mega-MoE span share is only about `18-19%` of the
   EXTEND GPU window and closer to `10-12%` of client-observed serving wall,
   a `0-2%` core delta naturally becomes `0-0.5%` e2e.

## B200 Cap/Overhead Diagnostics After Random Timing

Two follow-up random `8192 -> 1` jobs tested whether the remaining gap is a
cap/Waterfill-overhead issue.

First, job `2701784` forced the scheduler to chunk prefill at `4096` tokens
while keeping the same random prompt source:

| Case | Mean tok/s | Trimmed tok/s | Drop-first tok/s |
| --- | ---: | ---: | ---: |
| fused, 4096 chunk | `30904.0` | `31187.0` | `31212.6` |
| fused + Waterfill, 4096 chunk | `31256.6` | `31489.1` | `31517.8` |

Speedup was `+1.14%` mean, `+0.97%` trimmed, and `+0.98%` drop-first. This
confirms smaller source chunks make the relative Waterfill signal easier to
see, but the absolute throughput drops from the 8192-chunk baseline
(`~36.3K tok/s`) to only `~31.5K tok/s`. So forcing 4096 chunks is a useful
diagnostic, not a viable e2e optimization.

Second, job `2702038` kept 8192 chunks and enabled reusable exact-shape
Waterfill TopK output buffers:

```bash
SGLANG_WATERFILL_REUSE_TOPK_BUFFER=1
SGLANG_WATERFILL_REUSE_TOPK_BUFFER_CACHE_SIZE=8
```

| Case | Mean tok/s | Trimmed tok/s | Drop-first tok/s |
| --- | ---: | ---: | ---: |
| fused, 8192 chunk | `35638.5` | `35842.4` | `35844.4` |
| fused + Waterfill, 8192 chunk + reuse | `35981.7` | `36156.6` | `36162.9` |

The paired speedup was `+0.96%` mean, `+0.88%` trimmed, and `+0.89%`
drop-first. All five pairs were positive. However, the absolute Waterfill
throughput with reuse (`36162.9` drop-first tok/s) did not exceed the earlier
no-reuse Waterfill run (`36352.4` drop-first tok/s in job `2701169`). This is
job-to-job baseline variation, not enough evidence to make reusable buffers
the default.

The current actionable conclusion is unchanged:

- Waterfill is active and gives a stable small positive signal on B200 no-red
  random/MMLU workloads.
- The main 8K source-token prefill still uses the large `8320` cap because
  pre-dispatch requires `num_tokens <= padded_max`.
- Splitting source chunks to 4096 increases relative Waterfill speedup but
  loses too much absolute throughput.
- Reusing Waterfill TopK buffers is a reasonable diagnostic knob but is not
  proven as a default optimization.

## B200 Fused Waterfill Pre-Dispatch Check

The next hypothesis was that B200 no-red random throughput was losing the
Mega-MoE span gain to Waterfill materialization overhead:

```text
TopK [N, 6] -> materialized Waterfill TopK [N, 7]
             -> Mega-MoE pre-dispatch reads [N, 7]
```

An experimental gated path was added behind:

```bash
SGLANG_WATERFILL_FUSE_MEGA_MOE_PREDISPATCH=1
```

It skips the standalone Waterfill-expanded TopK tensor for the B200 EP2
Mega-MoE case and writes the shared column directly inside the Mega-MoE
pre-dispatch kernel. Correctness smoke on B200 job `2702363` compared the
fused pre-dispatch output against `materialize_waterfill_dispatch_fused(...)`
plus the existing `mega_moe_pre_dispatch(...)`; `x`, `x_sf`, `topk_idx`, and
`topk_weights` all matched exactly.

Throughput job `2702419` reran random `8192 -> 1`, 512 prompts, concurrency
128, five rounds, `MEGA_MOE_CAP_BUCKETS=4096:8320`:

| Case | Rounds tok/s | Mean tok/s | Trimmed tok/s | Drop-first tok/s |
| --- | --- | ---: | ---: | ---: |
| fused | `34224, 35882, 35994, 36027, 35996` | `35624.5` | `35957.3` | `35974.8` |
| fused + Waterfill | `34947, 36283, 36333, 36233, 36223` | `36003.7` | `36246.3` | `36267.9` |
| fused + Waterfill pre-dispatch | `35111, 36203, 36311, 36183, 36233` | `36008.3` | `36206.4` | `36232.7` |

Speedups:

| Comparison | Mean | Trimmed | Drop-first |
| --- | ---: | ---: | ---: |
| Waterfill vs fused | `+1.06%` | `+0.80%` | `+0.81%` |
| fused pre-dispatch vs Waterfill | `+0.01%` | `-0.11%` | `-0.10%` |
| fused pre-dispatch vs fused | `+1.08%` | `+0.69%` | `+0.72%` |

A short diagnostic job `2702771` confirmed the fused branch actually ran on
large prefill batches:

```text
MEGA_MOE_WATERFILL_FUSE_PREDISPATCH ... tokens=8058 ...
  can_fuse=True plan_ready=True fused=True
```

The early `tokens=2` and final `tokens=4` decode/cleanup calls correctly
reported `plan_ready=False fused=False`, because Waterfill only builds a
dispatch plan for sufficiently large batches.

So the missing B200 gain is not explained by the materialized Waterfill TopK
tensor or the extra pre-dispatch launch/read. Removing that overhead leaves the
same `~0.7-0.8%` random no-red e2e gain. The remaining dominant explanation is
still the Mega-MoE core/cap model:

- Waterfill strongly improves rank/expert balance, but the random 8K source
  batch still uses the `8320` source-token cap.
- The fused DeepGEMM Mega-MoE span only moves by about `2%` on this random
  no-red workload, far less than the full H20 DeepEP MoE path.
- Since the Mega-MoE interval is only a minority of the serving wall time,
  `~2%` span gain naturally lands near `~0.5-0.8%` end-to-end.

## B200 Min-Batch Waterfill Threshold Check

Another possible explanation was that Waterfill's fixed setup/materialization
cost was hurting medium MMLU batches, so the existing small-batch bypass was
made configurable:

```bash
SGLANG_WATERFILL_MIN_BATCH_FOR_BALANCE=64  # default, unchanged behavior
```

The implementation keeps the old default behavior and only changes the
threshold used by `DeepEPWaterfillBalancer._is_low_batch(...)`. The B200 MMLU
throughput runner was also updated to record the threshold in each case
summary and to support per-case aliases such as
`fused_waterfill_min512` and `fused_waterfill_min1024`.

Two launch issues were found during the diagnostic:

- `sbatch --export` treats commas as environment separators, so multi-case
  `CASE_ORDER` must use the script-supported colon form, e.g.
  `fused:fused_waterfill:fused_waterfill_min512`.
- The short sweep needs the same cap buckets as the successful B200 MMLU runs:
  `MEGA_MOE_CAP_BUCKETS=4096:8320`. Without this, the 512-token MMLU prefill
  case uses only the large `8320` buffer and can OOM on nodes with slightly
  less dynamic headroom.

Job `2703308` then ran a short same-job sweep on the MMLU 2k file with:

```text
CASE_ORDER=fused:fused_waterfill:fused_waterfill_min512:fused_waterfill_min1024
MEGA_MOE_CAP_BUCKETS=4096:8320
MEM_FRACTION_STATIC=0.90
TPUT_SAMPLE_SIZE=512
TPUT_WARMUP_ROUNDS=2
TPUT_MEASURE_ROUNDS=4
TPUT_CONCURRENCY=128
```

The run is intentionally a quick diagnostic, not a production-quality serving
number; four rounds are too noisy for final claims. It is still enough to test
whether a larger bypass threshold opens a clear win.

| Case | Threshold | Rounds tok/s | Mean tok/s | Trimmed tok/s | vs fused trimmed |
| --- | ---: | --- | ---: | ---: | ---: |
| fused | 64 | `10453, 6958, 6578, 34934` | `14730.75` | `8705.50` | baseline |
| default Waterfill | 64 | `10525, 6978, 6608, 35662` | `14943.25` | `8751.50` | `+0.53%` |
| Waterfill min512 | 512 | `10508, 7006, 6599, 35493` | `14901.50` | `8757.00` | `+0.59%` |
| Waterfill min1024 | 1024 | `10501, 6989, 6613, 35566` | `14917.25` | `8745.00` | `+0.45%` |

The threshold variants are within noise and do not improve on default
Waterfill. This rejects the "medium-batch Waterfill setup overhead" hypothesis
for the current B200 MMLU path. The bottleneck remains the Mega-MoE fused
core/cap model rather than the small-batch bypass policy.
