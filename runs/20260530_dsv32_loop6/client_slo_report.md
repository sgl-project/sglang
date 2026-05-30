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
- **Trial disclosure (directional):** **1 trial**, `WARMUP_SECONDS=0`, `MEASUREMENT_WINDOW_S=60` → **one full 320-prompt epoch per conc** (completed=320/320 each). This is a directional measurement; the strict all-trials/120s-warmup/600s-window run is the downstream hard-pass. No run hidden.
- **Durable evidence (recomputable from tracked files, no gitignored `.jsonl` dependency):**
  - `client_slo_int8/ac5_evidence_addendum.txt` — per conc: completed count, errors (all-empty proof), ISL/OSL distributions, and TTFT/TPOT/ITL percentile sources + the radix-on sidecar path.
  - `client_slo_int8/attribution_per_conc.txt` — corrected per-conc admission-wait attribution (queue p50/p95/p99; parsed-vs-valid rows; invalid/health filtering policy; measured-vs-inferred note).
  - `client_slo_int8/decode_batch_excerpt.txt` — server decode-batch evidence backing the per-request-TPS root cause.
  - plus `client_slo_int8/` 3× `.meta.json` sidecars (radix-on proof) + `client_slo_metrics.txt`. Raw 4 MB `.jsonl` remain gitignored (`*.jsonl`).

## Absolute results vs the strict SLO (and vs the Loop-5 baseline)

| conc | achieved (vs L5) | **P99 TTFT** | SLO `<22.0`? | Loop-5 TTFT | **per-req TPS** (1000/medTPOT) | SLO `≥30`? | Loop-5 TPS |
|---:|---:|---:|:--:|---:|---:|:--:|---:|
| 16 | **16.0** / 14.5 | **12.8 s** | ✅ PASS | 57.7 s | 17.6 | ❌ | 34.0 |
| 32 | **32.0** / 24.6 | 25.5 s | ❌ (close) | 132.9 s | 11.5 | ❌ | 33.9 |
| 64 | **60.1** / 35.7 | 111.2 s | ❌ | 292.0 s | 9.3 | ❌ | 33.9 |

## Directional improvement (directional characterization, not yet validated)
- **Admission restored.** Achieved concurrency 16.0 / 32.0 / 60.1 (vs Loop-5's 14.5 / 24.6 / 35.7 at the 53K pool). The footprint→pool lift (53056→396096 tokens) admits the concurrent KV that Loop-5 couldn't.
- **P99 TTFT collapsed 4.5× / 5.2× / 2.6×** vs Loop-5 (12.8 / 25.5 / 111.2 vs 57.7 / 132.9 / 292.0). **conc 16 now MEETS the strict `< 22 s` SLO** (12.8 s); conc 32 is on the cusp (25.5 s); conc 64 still misses but improved 2.6×. This is decisive directional progress on the loop's headline blocker.

## Measured attribution — admission-wait vs prefill (corrected; `attribution_per_conc.txt`)
Reprocessed from the **full** server request-time-stat log (967 rows parsed; **5 invalid `queue_duration<0` rows + 3 HEALTH_CHECK probes filtered with disclosed policy** → 959 valid; the >960 nominal is per-conc warmup requests). Rows bucketed per conc by wall-clock run windows (`T0 + cumulative measured durations`). **Measured vs inferred is now kept honest:**
- **MEASURED directly:** `queue_duration` = scheduler/admission wait — per conc **p99 = 10.5 / 22.3 / 99.4 s** (conc 16/32/64). Client TTFT (authoritative) p99 = 12.8 / 25.5 / 111.2 s; **min TTFT ≈ 1.3 s = the uncontended prefill-compute floor**.
- **INFERRED (not a raw counter):** post-admission residual = `TTFT_p99 − queue_p99` = **2.2 / 3.2 / 11.8 s** — time-to-first-token *after* admission (prefill compute **plus** chunked-prefill/decode interleave, **not** pure prefill). Tail-to-tail (p99−p99) is used, not p50−p50, since per-percentile rows are not the same request. `forward_duration` (p99 ≈ 41.6 / 69.2 / 82.9 s) is **completion-time** (prefill + all 512 decode steps) and is reported **for context only — never used as the first-token prefill term** (the R6 misuse is corrected).
- **Conclusion:** P99 TTFT is **admission-wait-dominated** at every conc (queue p99 ≫ post-admission residual); the queue term grows 10.5→22.3→99.4 s while the prefill floor stays ~1.3 s.
- **It is NOT KV-pool-admission-bound at the lifted point:** 64 concurrent × ~4608 tokens (4096 ISL + 512 OSL) = ~295K < the 396K pool, so the KV pool *fits* the concurrent set — the footprint fix did its job. The remaining queue is **throughput contention** (prefill+decode capacity under the 320-prompt flood), so the conc-32/64 follow-up is **chunked-prefill / scheduling**, not more footprint reduction.

## The per-request TPS finding (honest, important; `decode_batch_excerpt.txt`)
Per-request TPS measured **below the 30 SLO at every conc** (17.6 / 11.5 / 9.3), *below* Loop-5's 34. Root cause: **restoring admission grows the decode batch, which lowers per-request decode speed** — and the server decode-batch log confirms it quantitatively: aggregate gen throughput stays ~270–370 tok/s while the steady-state decode batch climbs from **16 (conc 16) → ~32 (conc 32) → up to ~38 (conc 64)**, so per-request decode TPS = gen/`#running-req` = **17.7 / 11.5 / 9.7 tok/s** — matching the client p50 TPS (17.6 / 11.5 / 9.3) almost exactly. (This **corrects the R6 summary's `#running-req: 19-20` figure**; the real steady-state batch sizes are 16 / ~32 / ~38, tabulated in `decode_batch_excerpt.txt`; the conclusion — per-req TPS < 30 — is unchanged.) Loop-5's 53K pool admitted only a small decode batch at conc 64 (→ 34 tok/s/req); restoring admission grows it, so the ≥30 TPS/req SLO is in genuine tension with high concurrency on this 671B MoE. The loop's premise that "DS already beats 30 TPS, only TTFT is the problem" held *only* at Loop-5's queue-starved operating point. (Measurement caveat: this directional run used `WARMUP=0`, so the window also includes prefill↔decode contention during the flood ramp, which further depresses TPOT; a warmed steady-state run is the downstream measurement.)

## Verdict — DIRECTIONAL: accepted progress, NOT a shippable pass (DEC-3)
This is a **directional characterization** of the footprint→admission→TTFT spine on the real client workload (one directional trial), **not** a validated shippable result. The directional evidence: admission is restored (achieved ≈ nominal), P99 TTFT collapsed vs Loop-5, **conc 16 reaches the strict `< 22 s` SLO** (12.8 s), and the corrected per-conc attribution characterizes the conc-16 win as the admission fix (queue removed) while the conc-32/64 residual is throughput-contention, not KV-admission. This is **accepted progress, explicitly not shippable**: the **strict SLO is NOT met at every conc — conc 32/64 P99 TTFT miss `< 22 s` (25.5 / 111.2 s) and per-request TPS misses `≥ 30` at every conc (17.6 / 11.5 / 9.3)** — and this strict miss remains a live mainline blocker on the loop's done-criterion. Two honestly-surfaced downstream blockers, with data, neither a footprint problem:
1. **conc-32/64 TTFT** is queue/throughput-bound under the flood → chunked-prefill / scheduling follow-up (the plan's anticipated "prefill-bound at conc 64" risk, confirmed by the attribution).
2. **Per-request TPS < 30** once admission is restored → the decode-batch/latency-throughput tradeoff; a real follow-up (decode optimization or an admission/throughput operating-point choice), recorded as a directional finding (cf. DEC-7), not a footprint regression.

The absolute strict numbers are recorded as the target; they become a hard pass/fail blocker downstream. A genuine miss here is recorded as a follow-up **with** the attribution breakdown — not a loop failure for the MVP.
