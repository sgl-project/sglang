# AC-7 — AC-11 directional DS-vs-DSA re-sweep at the lifted operating point (characterized, DEC-9)

3-trial sweep, conc 16/32/64, `num_prompts=64`, warmup 120 s / window 600 s, request_rate=inf,
radix-on both sides, at the **lifted** operating point: **DS int8 @ mem 0.7** vs **DSA-default @
mem 0.85** (per-side mem consistency; the admission-gap lever the loop moved). Both sweeps ran on
node 0 (localhost) sequentially — see "Methodology note" below. Comparator: `benchmark_compare.py
--ac11` (`ac11_resweep.md`); recomputable per-trial metrics + source SHA256 in `ac7_resweep_metrics.json`.

## Headline — admission RESTORED at the lifted point (the spine's payoff)

Effective (achieved) concurrency, median of 3 trials — vs the Loop-5 queue-starved baseline:

| conc (nominal) | DS achieved @ lifted | DS/nominal | Loop-5 DS (mem 0.6) | DSA achieved |
|---:|---:|:--:|---:|---:|
| 16 | **16.0** | **100%** | 14.5 (≈91%) | 16.0 |
| 32 | **32.0** | **100%** | 24.6 (≈77%) | 32.0 |
| 64 | **47.0** | **73%** | 35.7 (≈56%) | 63.9 |

- **DS now admits full nominal concurrency at conc 16 and 32** (100%, vs Loop-5's queue-starved
  91%/77%) and **improves conc-64 admission to 73%** (vs Loop-5's 56%). The footprint→admission
  spine works across 3 trials; errors 0 on every run. conc-64 does not fully track nominal —
  consistent with the AC-5 finding that conc-64 is **throughput/prefill-bound**, not KV-pool-bound.

## DS-vs-DSA parity gates: FAIL (expected; a DEC-7 directional follow-up, not a footprint issue)

Comparator gates (DS TPS ≥ 95% of DSA; DS P99 TTFT ≤ 1.10× DSA) **FAIL at every conc**:

| conc | DS TPS / DSA TPS (ratio) | DS P99 TTFT / DSA P99 TTFT (ratio) |
|---:|---|---|
| 16 | 17.7 / 47.0 (0.38) | 12.8 s / 0.72 s (17.9×) |
| 32 | 11.5 / 37.6 (0.31) | 25.5 s / 1.28 s (19.9×) |
| 64 | 9.8 / 29.6 (0.33) | 100.8 s / 2.04 s (49.3×) |

This is the **known, expected DS-vs-DSA gap**, not a regression and not a footprint problem:
1. **DSA wins per-request** because V3.2's *trained* DSA indexer + the shared kernel are faster
   and lower-latency than DS's offline channel-mask selection at the same 2048 budget (the AC-1
   strategic-gate premise; DEC-7 directional).
2. **DS per-req TPS fell once admission was restored** — bigger decode batch → lower per-req decode
   TPS (`BL-admission-restore-tps-tradeoff`).
3. **DS TTFT is admission-queue-dominated under the request_rate=inf flood** — DS's lower throughput
   drains the flood-queue slower than DSA.

## Profiling obligation — discharged at AC-7 methodology (the failing rows)
Captured at the **AC-7 methodology** (DS int8/mem0.7/radix-on, `num_prompts=64`, warmup 120 s /
window 600 s, `--enable-request-time-stats-logging`; this profiling run reproduces the AC-7
sweep — TTFT 12.8/25.4/100.8 s, TPS 17.2/11.5/9.8, achieved 16/32/47 — so it is the same regime,
not the AC-5 WARMUP=0/320/60 directional run, which is now only background):
- **`decode_batch_ac7.txt`** (from the AC-7 3-trial sweep DS server log) — per-request decode TPS =
  gen/`#running-req`: **17.7 / 11.5 / 9.8 tok/s** at decode batch 16 / 32 / ~38, reconciling the
  comparator's per-req **TPS-gate FAIL** exactly.
- **`queue_attribution.txt`** (request-time stats, bucketed per conc by `.meta.json` run windows,
  1082 valid benchmark rows) — `queue_duration` p99 = **10.5 / 22.6 / 96.7 s** vs client P99 TTFT
  12.8/25.4/100.8 s (post-admission residual ~2–4 s). DS **TTFT is admission-queue-dominated** at
  every conc: the request_rate=inf flood holds requests in the scheduler queue, and DS drains it
  slower than DSA (lower throughput) → the **TTFT-gate FAIL**. At conc-64 the queue is largest,
  consistent with the achieved-concurrency deficit (47/64).

Per **DEC-7 / DEC-9**, the DS-vs-DSA TPS/TTFT miss is a **recorded directional follow-up** with
measured AC-7-methodology attribution (decode batch + request-time queue), **not** a build-break or
a footprint regression; AC-7 is **soft** (characterized).

## AC-7 verdict (characterized, DEC-9)
**Admission restored** (DS achieved 100% at conc 16/32, 73% at conc-64, up from Loop-5's
91/77/56%) — the footprint→admission spine validated across 3 trials. The **DS-vs-DSA parity
gates fail** (DS per-request TPS ~1/3 of DSA, TTFT 18–49× DSA), the expected DSA-trained-indexer
advantage + the admission-restore throughput tradeoff — a DEC-7 directional follow-up, attributed
at AC-7 methodology by `decode_batch_ac7.txt` (TPS) + `queue_attribution.txt` (TTFT/queue), **not**
a footprint regression. DSA-default reproduces its baseline (0.72/1.28/2.04 s,
46.9/37.5/29.5 TPS; conc-64 TPS ~29.5 is the queued pre-existing DSA limit).

## Methodology note (node1 unavailable → both sides on node 0, sequential)
The intended cross-node bring-up (DS node 0 + DSA node 1) was abandoned: node 1's remote server
boot proved intractable this round (setsid/nohup/tmux all failed — fast-ssh-close teardown +
accumulated zombie procs; no DSA process ever loaded weights). Both sweeps therefore ran on **node 0
sequentially** (DS, then DSA), each **localhost** — which is comparator-clean (same node/session/
commit; only per-side mem differs, as in Loop-5) and avoids any cross-node host-mismatch. Because
neither sweep was cross-node, the cross-node wrapper smoke is **N/A this round**; the `--host`
benchmark fix (R12) is verified working in-wrapper (the bench `Waiting up to 60s for
http://127.0.0.1:30000` banner + the matching DS `.meta.json` sidecar) and stands for future
cross-node use (R11 also proved `bench_serving --host node1` targets node 1).

## Artifacts
- `ac11_resweep.md` — `benchmark_compare.py --ac11` report (gates, ratios, effective concurrency).
- `ac7_resweep_metrics.json` + `ac7_metrics_tool.py` — exact per-trial DS+DSA arrays (effective `concurrency`, `ttfts_s`, `per_req_gen_tps`) + full SHA256; `--verify` recomputes the `ac11_resweep.md` rows from committed data, fail-closed.
- `decode_batch_ac7.txt` — AC-7-methodology decode-batch evidence (per-req TPS = gen/#running-req) reconciling the TPS-gate FAIL.
- `queue_attribution.txt` — AC-7-methodology DS request-time stats (per-conc `queue_duration`) reconciling the TTFT-gate FAIL (admission-queue-dominated).
- `*.meta.json` — per-run sidecars (radix-on, per-side mem, commit SHA, operating point).
