# AC-11 directional comparator — result + #F accounting (Round 10)

3-trial radix-on DSA+DS sweep at conc 16/32/64, 120s warmup, 600s measurement window,
NUM_PROMPTS=64, GSP ISL≈4096 / OSL 512. Both sides radix-ON (DSA default; DS via the
fixtures-passed artifact, no env override). 18 JSONLs (9 DSA + 9 DS), each `duration ≥ 602s`;
sidecars carry the locked Option-B fields with matching `disable_radix_cache=false`,
`tp_size=8`, `page_size=64`, `kv_cache_dtype=fp8_e4m3`, `chunked_prefill_size=8192`.
Comparator: `benchmark_compare.py --ac11`; report `mvp_compare_ac11.{md,json}`.

## Verdict: directional MISS (recorded per DEC-7, not a build-break)

| Conc | DS/DSA TPS | TPS gate | DSA P99 TTFT | DS P99 TTFT | TTFT gate |
|------|-----------|----------|--------------|-------------|-----------|
| 16 | 0.726 | FAIL | 0.73s | 57.7s | FAIL |
| 32 | 0.900 | FAIL | 1.37s | 132.9s | FAIL |
| 64 | 1.146 | **pass** | 2.04s | 292.0s | FAIL |

Gates: DS TPS ≥ 95% of DSA TPS; DS P99 TTFT ≤ 1.10× DSA. The comparator exited 3 (directional
miss published), which per DEC-7 is a recorded AC-11 failure + follow-up, NOT a build-break.

## #F: effective vs nominal concurrency (the dominant cause of the TTFT miss)

| Conc (nominal) | DSA achieved | DS achieved | DS/nominal |
|----------------|--------------|-------------|------------|
| 16 | 15.99 | 14.50 | 91% |
| 32 | 31.98 | 24.59 | 77% |
| 64 | 63.93 | 35.70 | 56% |

DS reserves a per-rank TokenLabelTable on top of the ~84 GB/rank V3.2 FP8 weights and serves at
`mem_fraction_static=0.6`; the baseline serves at 0.85. DS's smaller KV pool admits fewer
concurrent requests, so under sustained offered load (conc 16/32/64) requests queue and DS P99
TTFT is dominated by admission wait, not per-request prefill latency. DSA, with a larger KV
pool and the GSP ~55% shared prefix cached once (radix-on), admits full concurrency and answers
with a sub-second prefix-cached TTFT. radix-on lifted DS effective concurrency far above the
radix-off smoke (~2 in Round 4) to 14–36, but it is still admission-bound at conc 64.

Notably DS per-request TPS is competitive (within 10% at conc 32, and ABOVE DSA at conc 64,
ratio 1.146) — the gap is TTFT/admission, not generation rate.

## Follow-up (filed; does not block the recorded AC-11 artifact)

1. Reduce DS's per-rank memory overhead (TokenLabelTable footprint) or otherwise raise the DS
   KV budget at the radix-on operating point so effective concurrency approaches nominal, then
   re-run the sweep. (Raising `mem_fraction_static` past 0.6 currently OOMs DS during
   generation — the TokenLabelTable sizing is the lever.)
2. Capture a DS decode/admission profile at conc 64 to separate queue wait from compute.
3. Re-evaluate the TTFT directional target once effective concurrency matches DSA.

The mem-fraction asymmetry is recorded (not hidden): the comparator treats `mem_fraction_static`
and per-boot `random_seed` as recorded-not-matched fields and surfaces achieved concurrency, so
the TTFT comparison is interpreted with the admission caveat. All locked Option-B fields
(radix, TP, page, dtype, backends, graph flags) are still strictly matched.
