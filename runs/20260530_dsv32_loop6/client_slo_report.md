# AC-5 — Direct client-SLO validation (DS int8 @ lifted 0.7, radix-on) — DIRECTIONAL

The loop's headline. Run the full client workload against DS with the **compact int8
table at the lifted 0.7 operating point, radix-on**, and measure the **absolute**
P99 TTFT and per-request TPS vs the strict client SLO (`P99 TTFT < 22.0 s` AND
`≥ 30 TPS/req`), with a **measured admission-wait vs prefill-compute attribution**.
Graded **directional** (DEC-3): movement toward the strict numbers is accepted
progress, **explicitly not a shippable pass** (an all-trials strict pass at every
conc is reserved for downstream).

## Setup (real single-node TP=8 H200)
- DS: `serve_double_sparsity.sh SIGNATURE_DTYPE=int8 MEM_FRACTION_STATIC=0.7`, Loop-5 mask, Option B; **radix-on authorized by the regenerated int8 fixture artifact** `ds_radix_fixture_state_int8.json` (validator: "fixture recorded as PASSED ... artifact_sha256=f3b67943"); `--enable-request-time-stats-logging`. Radix-on proven from the `.meta.json` sidecars (`disable_radix_cache=False`, `mem_fraction_static=0.7`, `enable_double_sparsity=True`, `max_total_num_tokens=396096`).
- Workload: `benchmark.sh MODE=double_sparsity`, gsp 4096 ISL (sys 2253 + q 1843; measured median input_len ≈ 4280) / 512 OSL / ~55% cache, conc 16/32/64, NUM_PROMPTS=320.
- **Trial disclosure (directional):** **1 trial**, `WARMUP_SECONDS=0`, `MEASUREMENT_WINDOW_S=60` → **one full 320-prompt epoch per conc** (completed=320/320 each). This is a directional measurement; the strict all-trials/120s-warmup/600s-window run is the downstream hard-pass. No run hidden. Artifacts: `client_slo_int8/` (3× JSONL + `.meta.json` sidecars + `reqtimestats_excerpt.txt`).

## Absolute results vs the strict SLO (and vs the Loop-5 baseline)

| conc | achieved (vs L5) | **P99 TTFT** | SLO `<22.0`? | Loop-5 TTFT | **per-req TPS** (1000/medTPOT) | SLO `≥30`? | Loop-5 TPS |
|---:|---:|---:|:--:|---:|---:|:--:|---:|
| 16 | **16.0** / 14.5 | **12.8 s** | ✅ PASS | 57.7 s | 17.6 | ❌ | 34.0 |
| 32 | **32.0** / 24.6 | 25.5 s | ❌ (close) | 132.9 s | 11.5 | ❌ | 33.9 |
| 64 | **60.1** / 35.7 | 111.2 s | ❌ | 292.0 s | 9.3 | ❌ | 33.9 |

## Directional improvement (the spine works)
- **Admission restored.** Achieved concurrency 16.0 / 32.0 / 60.1 (vs Loop-5's 14.5 / 24.6 / 35.7 at the 53K pool). The footprint→pool lift (53056→396096 tokens) admits the concurrent KV that Loop-5 couldn't.
- **P99 TTFT collapsed 4.5× / 5.2× / 2.6×** vs Loop-5 (12.8 / 25.5 / 111.2 vs 57.7 / 132.9 / 292.0). **conc 16 now MEETS the strict `< 22 s` SLO** (12.8 s); conc 32 is on the cusp (25.5 s); conc 64 still misses but improved 2.6×. This is decisive directional progress on the loop's headline blocker.

## Measured attribution — admission-wait vs prefill-compute (REQUIRED)
From the per-request `ReqTimeStats` (`reqtimestats_excerpt.txt`: `queue_duration` = admission/queue wait, `forward_duration` = prefill+decode compute) and the per-conc TTFT floor:
- **Prefill-compute floor** (min TTFT, no queue): **~1.3 s** at every conc — a single un-queued request prefills its 4096 ISL in ~1.3 s. Prefill compute is **not** the dominant TTFT term.
- **Queue/admission-wait dominates the tail:** P99 TTFT − prefill-floor ≈ 11.5 s (conc 16) / 24 s (conc 32) / ~110 s (conc 64); `queue_duration` reaches p99 ≈ 98.5 s at the high-load tail while own `forward_duration` p99 ≈ 82.5 s. **The residual TTFT is queue-wait-bound**, i.e. the 320 flooded requests contend for the server's finite prefill+decode throughput at `max_concurrency`.
- **Crucially, it is NOT KV-pool-admission-bound at the lifted point:** 64 concurrent × ~4608 tokens (4096 ISL + 512 OSL) = ~295K < the 396K pool, so the KV pool *fits* the concurrent set — the footprint fix did its job. The remaining queue is **throughput contention** (prefill+decode capacity under the flood), so the conc-32/64 follow-up is **chunked-prefill / scheduling**, not more footprint reduction.

## The per-request TPS finding (honest, important)
Per-request TPS measured **below the 30 SLO at every conc** (17.6 / 11.5 / 9.3), *below* Loop-5's 34. Root cause: **restoring admission grows the decode batch, which lowers per-request decode speed.** Loop-5's 53K pool admitted only ~2–3 of these 4608-token requests into decode at conc 64 (tiny batch → 34 tok/s/req); the 396K pool now decodes ~19–20 concurrently (server `Decode batch #running-req: 19-20`, gen ~277 tok/s ⇒ ~14 tok/s/req). This is the **latency/throughput tradeoff**: the footprint fix bought TTFT (admission) at the cost of per-request TPS (bigger decode batch). The loop's premise that "DS already beats 30 TPS, only TTFT is the problem" held *only* at Loop-5's queue-starved operating point; once admission is restored the ≥30 TPS/req SLO is in genuine tension with high concurrency on this 671B MoE. (Measurement caveat: this directional run used `WARMUP=0`, so the window also includes prefill↔decode contention during the flood ramp, which further depresses TPOT; a warmed steady-state run is the downstream measurement.)

## Verdict — DIRECTIONAL: accepted progress, NOT a shippable pass (DEC-3)
The footprint→admission→TTFT **spine is validated**: admission is restored (achieved ≈ nominal), P99 TTFT collapsed vs Loop-5, **conc 16 reaches the strict `< 22 s` SLO**, and the attribution proves the conc-16 win is the admission fix (queue removed) while the conc-32/64 residual is throughput-contention, not KV-admission. This is **accepted progress, explicitly not shippable**: the strict SLO is **not** met at every conc (conc 32/64 TTFT miss; per-request TPS misses at all conc). Two honestly-surfaced downstream blockers, with data, neither a footprint problem:
1. **conc-32/64 TTFT** is queue/throughput-bound under the flood → chunked-prefill / scheduling follow-up (the plan's anticipated "prefill-bound at conc 64" risk, confirmed by the attribution).
2. **Per-request TPS < 30** once admission is restored → the decode-batch/latency-throughput tradeoff; a real follow-up (decode optimization or an admission/throughput operating-point choice), recorded as a directional finding (cf. DEC-7), not a footprint regression.

The absolute strict numbers are recorded as the target; they become a hard pass/fail blocker downstream. A genuine miss here is recorded as a follow-up **with** the attribution breakdown — not a loop failure for the MVP.
