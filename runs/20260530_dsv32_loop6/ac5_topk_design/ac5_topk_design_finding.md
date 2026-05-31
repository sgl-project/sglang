# AC-5 conc-16 lever — blocked-top-k design microbench (DECISIVE: full-context kernel does NOT win)

Owner (R22) chose the full-context blocked-topk kernel over the bounded-context op-point, expecting it to
lift full-context conc-16 from 24.9 to ≥30 TPS/req. This GPU microbench (61 layers, bs=16, seq=4096,
max_seq_len=163840 — the per-decode-step selection over-scan) measures the candidate designs and shows the
expectation is empirically false.

## Measured (torch.topk, the fast primitive; median of 30 iters)
| design | topk ms/step (61L) | implied conc-16 step | implied conc-16 TPS |
|---|---:|---:|---:|
| A — monolithic over 163840 (current production merge) | 6.56 | 36.90 ms | **27.1** |
| B — skip-ideal: merge over the LIVE region 4096 only (CAPS context) | 2.38 | 32.72 ms | **30.6** |
| C — blocked bw=8192/partial_k=2048, SKIP, no context cap (merge 40960) | 8.50 | 38.84 ms | 25.7 |
| C′ — blocked torch-full, no skip kernel | 12.33 | 42.68 ms | 23.4 |

(implied step uses R17's measured 36.9 ms decode step / 6.56 ms merge; B=30.6 cross-validates R18's measured
bounded-context closed-batch 30.3.)

## Findings
1. **The only design that reaches conc-16 ≥30 is B — and B caps the merge to the live region, i.e. it IS the
   bounded-context operating point** (the merge width = context length; capping it = capping context). This is
   exactly the bounded-context op-point the owner declined in R22 (which independently measured conc-16 30.3).
2. **The full-context blocked top-k (C) is WORSE than monolithic** (25.7 < 27.1 TPS): under CUDA-graph fixed
   shapes the Stage-2 merge must process num_blocks×partial_k candidates (40960 here, or the full 163840 for
   Codex's bw=512/partial_k=512), and two topk passes (Stage1 + Stage2) cost MORE kernel-launch + memory
   overhead than the single monolithic topk — even at smaller widths. Skipping dead blocks in Stage 1 does not
   help enough because the fixed-shape merge still dominates. Codex's prescribed bw=512/partial_k=512 is design
   A by another name (merge over 163840) — no win.
3. The R23 deterministic tie-break uses a full `argsort` (correct as the eager oracle) which is even SLOWER
   than topk — so the hot-path kernel would additionally need a fast position-asc-tie top-k, not a sort.

## Conclusion
There is **no graph-safe FULL-CONTEXT blocked-top-k design that reaches conc-16 ≥30 TPS/req** — the blocked
variants are worse than the current monolithic, and the only win requires bounding the merge/scan width to the
workload (= the bounded-context op-point). The owner's R22 kernel choice is empirically infeasible for the
conc-16 perf goal; conc-16 ≥30 at full context is not achievable by a top-k kernel. conc-32/64 ≥30 remain
structurally unattainable regardless (DS ≤ DSA; DSA itself 29.4 at conc-64). Surfacing to the owner with these
numbers (the R22 choice was made without this perf evidence).

## Artifacts
- `topk_design_microbench.py` + `topk_design_microbench.json` — the GPU timing (reproducible).
