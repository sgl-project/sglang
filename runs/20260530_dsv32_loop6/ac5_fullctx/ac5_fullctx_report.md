# AC-5 — full-context client-SLO, MEASURED (DS int8 / mem-0.7 / radix-on / full context / TP=8)

The measured AC-5 client workload at the **fixed full-context Option-B operating point** Codex requires
(no `--context-length` cap — the point AC-8 validated), with the R17 decode score-kernel fix in place.
Steady-state methodology (warmup 120 s / window 300 s, per the cold-flood lesson), GSP 4096 ISL / 512 OSL,
conc 16/32/64, radix-on proven, `--enable-request-time-stats-logging` for attribution.

## Streaming root cause (R18 empty-array) — resolved
The R18 empty-latency arrays (`ttfts`/`itls`/`generated_texts` empty, impossible throughput) were
**ctx8192-specific** (the bounded-context server), NOT a general bench bug: at full-context, both
fixed-count and window-mode `bench_serving` produce real per-request arrays (probe: completed=64,
all ttfts real, 32659 ITL tokens, p99_ttft 12.76 s). The R19 fail-closed guard is the safety net (it
would raise on any future empty-latency run); this full run passed the guard (no empty-latency rows).

## Operating point (proven — `get_server_info_fullctx.json`, `meta_c16.json`)
`enable_double_sparsity=True`, `signature_dtype=int8`, `mem_fraction_static=0.7`, `disable_radix_cache=False`,
fixture artifact set, `context_length=None` (= full model context 163840), `max_total_num_tokens=396096`,
TP=8, `enable_request_time_stats_logging=True`. Identical to the AC-4/AC-8 full-context lifted point.

## Measured results vs the strict SLO (verifier `ac5_fullctx_metrics_tool.py --verify` PASS)
| conc | achieved | **P99 TTFT** | `<22 s`? | **per-req TPS p50** | `≥30`? |
|---:|---:|---:|:--:|---:|:--:|
| 16 | 16.00 | **13.13 s** | ✅ | 24.9 | ❌ |
| 32 | 31.99 | 25.33 s | ❌ (close) | 19.5 | ❌ |
| 64 | 47.03 | 77.90 s | ❌ | 17.3 | ❌ |

**Exact per-request source committed** (`ac5_fullctx_arrays.json`, R21 rebuild to the R9 standard):
per-request `ttfts_s`, `itl_sum_s`, `output_lens`, `input_lens`, `errors_empty`, gen-nonempty count, full
64-hex source SHA256, and the stored headline. The verifier **recomputes** P99 TTFT = p99(ttfts) and per-req
TPS p50 = p50(output_len/itl_sum) **from the raw committed arrays** (never a stored derived metric — the R20
defect where a derived `per_req_gen_tps` was tamperable), plus aggregate **mean** integrity (sensitive to
every element, so a single-element tamper a robust median would miss is caught), the R18 empty-latency class
(every ttft>0, every itl_sum>0, output_len==512, errors empty, len==completed, gen-nonempty==completed), and
the operating point from ALL THREE `.meta.json` sidecars (int8 / mem0.7 / radix-on / fixture / full context /
TP=8 / stats-on). **Fail-closed: 6 tamper tests each exit 1** (single itl_sum, single output_len, single
ttft=0, stored TPS p50→100, stored P99 TTFT→5000, sidecar disable_radix_cache→True); clean exits 0 PASS.
No inferred TTFT — all measured.

## Measured admission-wait attribution (`queue_duration` from ReqTimeStats; `ac5_fullctx_attribution.txt`)
Benchmark rows (`output_len=512`) bucketed per conc by print-time gaps (n=256/320/315):
- conc 16: **queue p99 = 10.5 s** (p50 4.9) → client P99 TTFT 13.13 s, residual ~2.6 s = prefill compute.
- conc 32: **queue p99 = 22.6 s** (p50 10.7) → TTFT 25.33 s, residual ~2.7 s.
- conc 64: **queue p99 = 74.0 s** (p50 21.4) → TTFT 77.90 s, residual ~3.9 s.

**P99 TTFT is admission-queue-dominated at every conc** (queue p99 ≈ TTFT; the prefill floor ≈ min TTFT
~1–3 s). `forward_duration` (completion-time = prefill + all 512 decode steps) is recorded for context only,
never used as a first-token term. The KV pool fits the concurrent set (64×4608 ≈ 295K < 396K pool), so the
queue is **throughput/decode contention** under the request_rate=inf flood, not KV-pool admission — DS's
slower-than-DSA decode drains the flood-queue slower.

## Component breakdown (where the decode step goes)
Per-req TPS = 1/decode_step_time. At conc-16 the client-measured 24.9 TPS/req (step ~40 ms incl. prefill
interleave) decomposes as: the **DSA FlashMLA+MoE floor** (AC-7 verified DSA conc-16 = 46.1 TPS/req → step
~21.7 ms, the per-request cost DS shares and cannot beat) + the **DS-specific selection** (R17 microbench:
~12.5 ms/step at full context after the score-kernel early-exit, of which ~6 ms is the residual `torch.topk`
over-scan over the full 163840-wide score row) + token-label-write (~2.7 ms) + prefill-interleave under the
flood. The residual topk over-scan is the conc-16 ≥30 lever (see verdict).

## Verdict — DIRECTIONAL (DEC-3); conc-16 passes TTFT, misses TPS at full context
- **conc-16 P99 TTFT 13.13 s < 22 s — MEETS the strict tail-latency SLO** at full context. Admission is
  restored (achieved 16.00) and the queue (10.5 s) is well within the 22 s budget.
- **Per-req TPS misses 30 at every conc** (24.9 / 19.5 / 17.3). At conc-16 the full-context gap (24.9 < 30)
  is the residual DS-selection `torch.topk` over-scan: the R18 bounded-context op-point (which shrinks the
  topk scan width) reached conc-16 **closed-batch 30.3 TPS/req**, but at full context the exact fix needs the
  blocked-topk kernel (research-grade — within-block K=2048 under CUDA-graph). conc-32/64 TPS (19.5/17.3) is
  the **structural decode-batch ceiling**: DS ≤ DSA, and even DSA is 37.0/29.4 at conc 32/64 (conc-64 ≥30 is
  unattainable for either on this 671B-MoE/H200/TP=8 hardware).
- vs Loop-5 (57.7/132.9/292.0 s; 34/33.9/33.9 cold-starved TPS) and vs the R6 full-context cold-flood
  (12.8/25.5/111.2 s; 17.6/11.5/9.3 TPS): TTFT collapsed and steady-state per-req TPS improved markedly
  (24.9/19.5/17.3 vs R6 17.6/11.5/9.3) thanks to the R17 decode score-fix + steady-state methodology.

This is **accepted progress, not a shippable pass** (DEC-3): conc-16 meets the TTFT axis at full context;
the TPS axis and conc-32/64 are recorded with measured attribution. The owner-set criterion (conc-16 strict
+ characterize 32/64) is met on TTFT; the conc-16 TPS axis at full context remains gated on the blocked-topk
kernel vs an explicit bounded-context rescope (the bounded-context op-point already demonstrates conc-16
30.3 closed-batch).

## Artifacts
- `ac5_fullctx_arrays.json` + `ac5_fullctx_metrics_tool.py` — exact per-request arrays + fail-closed verifier.
- `ac5_fullctx_attribution.txt` — per-conc measured queue_duration attribution + decode-component lines.
- `get_server_info_fullctx.json` + `meta_c16.json` — operating-point + radix-on `.meta.json` proof.
