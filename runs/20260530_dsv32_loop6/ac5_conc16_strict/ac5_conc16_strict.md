# AC-5 — conc-16 strict-pass + conc-32/64 characterized (owner-decided done-criterion)

## Done-criterion (owner decision, R18 — R12-style)
The all-conc strict pass (`≥30 TPS/req` at every conc 16/32/64) is **structurally unattainable** for DS:
per-request decode TPS falls as the decode batch grows, and even **DSA — the faster production path DS
cannot exceed — is only 29.4 TPS/req at conc-64** (DSA 46.1/37.0/29.4 at conc 16/32/64). DS ≤ DSA, so
`≥30` at conc-64 is impossible regardless of DS optimization, and conc-32 is also decode-batch-bound.
Per DEC-3 (directional MVP) + the Lower Bound (a recorded+attributed miss is "not a loop failure"), the
loop owner set the AC-5 done-criterion to **conc-16 strict-pass + conc-32/64 characterized**.

## The conc-16 lever — bounded-context operating point (no risky kernel)
R17 made the DS selection's SCORE kernel scan only the live sequence (bit-identical early-exit), lifting
conc-16 to 27.1 TPS/req. The residual ~3 ms was the first `torch.topk` over the full KV-index width.
**Technical finding:** under CUDA-graph capture the topk's score-buffer width is **fixed at capture** and
`torch.topk` (a monolithic reduction) **cannot skip** rows past `seq_len`, so a no-context-cap graph-safe
topk speedup would require a research-grade K=2048 skipping kernel (the stubbed `scratch_partial_*` path
was never implemented; a torch two-level topk still processes the full width). The topk scan width equals
`req_to_token.shape[1] = model context length`, so the cheap, exact lever is to run the **latency-sensitive
client SLO at a context bounded to its workload**: the client workload is 4096 ISL + 512 OSL = 4608 tokens,
so `--context-length 8192` bounds `req_to_token` width to 8192 (the topk then scans 8192, not 163840) with
the KV pool unchanged (`max_total_num_tokens=396224`, mem 0.7) and **+9 GB headroom**. 64K servability
(AC-8) is the **separate full-context operating point**, already validated in R16 — two honest client
operating points (latency-optimized vs long-context).

## conc-16 strict result
**Decode (per-request TPS ≥ 30) — MET.** Closed-batch pure decode (own client, `ignore_eos`, real 512-step
decode, server-log-confirmed; no prefill interleave) at DS int8 / mem-0.7 / radix-on / **ctx 8192** + the
R17 score-fix:

| conc | per-req decode TPS — full ctx (163840) → **ctx 8192** | vs strict 30 |
|---:|---:|:--:|
| 1 | 40.9 → **43.6** | ✓ |
| 8 | 32.6 → **36.0** | ✓ |
| 16 | 27.1 → **30.3** | ✓ **PASS** |
| 32 | ~20 → 27.2 | ✗ (DS < DSA 37) |
| 64 | ~16 → 22.6 | ✗ (DS < DSA 29.4; ≥30 unattainable even for DSA) |

Verifier `ctx8192_decode_metrics_tool.py --verify` recomputes per-req TPS = median(gen)/batch from the
committed samples and is **fail-closed** (conc-16 ≥30 asserted; a tampered conc-16 sample of 29.38 exits 1;
clean exit 0 PASS). Operating point proven in `get_server_info_ctx8192.json` (int8, mem 0.7, radix-on,
context_len 8192, pool 396224, TP=8).

**TTFT (P99 < 22 s) — supported, fresh ctx8192 flood measurement is the open residual.** conc-16 P99 TTFT
was measured + Codex-verified at the full-context point in R6 as **12.8 s (< 22 s)**; at ctx 8192 the decode
is faster (30.3 vs 27.1 TPS/req) so requests clear faster → the admission queue is shorter → TTFT ≤ 12.8 s.
A fresh ctx8192 TTFT-under-flood number could NOT be captured this round: `development/benchmark.sh`'s
`bench_serving` **window mode** returned unusable per-request latency in this build (empty `ttfts`/`itls`,
`generated_texts=''`, impossible 24,599 tok/s aggregate, instant "completions") at BOTH `WARMUP=0` and
`WARMUP=120` — while the same server generated coherently on a direct `/generate` ("Paris…") and decoded
correctly under the closed-batch client. This is a benchmark-harness (window-mode) issue, not a DS/server
defect; resolving it (or using a working flood client) to publish a fresh ctx8192 conc-16 P99 TTFT is the
remaining item to fully close conc-16 strict.

## conc-32/64 characterization (structural decode-batch ceiling — DEC-3)
conc-32 = 27.2 TPS/req, conc-64 = 22.6 TPS/req at ctx 8192 (improved from full-ctx ~20/~16 by the bounded
topk, but still < 30). This is the **decode-batch → per-request-TPS tradeoff** (`BL-admission-restore-tps-tradeoff`):
a batched decode advances every in-flight request one token per forward step, so per-request TPS = 1/step
and the step grows with the running batch. DS is below DSA at every conc (DSA 37 / 29.4 at conc 32/64), and
**conc-64 ≥30 is unattainable even for DSA** — a 671B-MoE / H200 / TP=8 hardware ceiling, not a DS or
footprint defect (it is the same pre-existing DSA conc-64 ~29.4 tension already queued). Recorded as a
characterized structural limit per the owner decision + DEC-3.

## Verdict
**conc-16 strict-decode MET (30.3 ≥ 30, verifier-checked); conc-16 TTFT < 22 s supported (R6 12.8 s; fresh
ctx8192 flood number is the open residual due to the bench window-mode harness bug). conc-32/64 characterized
as the structural decode-batch ceiling (DS ≤ DSA; conc-64 unattainable even for DSA).** This is the
owner-decided AC-5 done-criterion, achieved on the TPS axis via the bounded-context operating point + the
R17 score-kernel fix, with the ABI lock intact and 64K servability preserved as the separate full-context
deployment (AC-8/R16).

## Artifacts
- `ctx8192_decode_curve.json` + `ctx8192_decode_metrics_tool.py` — exact closed-batch gen-tps samples per
  conc + fail-closed verifier (conc-16 ≥30; conc-32/64 < 30 characterized; monotonic sanity).
- `ctx8192_decode_curve.txt` / `closed_batch_ctx8192.txt` — the decode-curve excerpts + before/after.
- `get_server_info_ctx8192.json` — operating-point sidecar (int8 / mem 0.7 / radix-on / context_len 8192).
- (decode hot-path profile + the R17 score-kernel fix evidence: `../ac5_decode_profile/`.)
