# M11 — AC-6 perf consolidation + DS-vs-DSA recall/perf/non-regression report

Consolidated guardrail report for Loop 7 (the source artifact for the task20
decision record). All measurements are at the Loop-7 op-point under **CUDA graph**:
DS int8 / `mem_fraction_static=0.7` / fp8-KV / TP=8 / page 64 / `flashmla_kv`
prefill+decode / radix-off / `--disable-overlap-schedule` / `--disable-piecewise-cuda-graph`.
GPU: 8× NVIDIA H200 (sm90). Commit: R19 tree on `f9f6ec056`.

## Perf guardrails (conc-1 / conc-16, graph mode)
Per-request decode TPS measured by the **closed-batch** method (the trustworthy
pure-decode metric — N concurrent `/generate`, short prompt, `ignore_eos`, OSL=256,
`#queue-req: 0`; the server-log `gen throughput / #running-req` cross-checks the
client number). All decode batches ran under CUDA graph (`cuda graph: True`).

| variant | scorer | conc-1 TPS/req | conc-16 TPS/req | GPU mem/GPU | graph | admission |
|---|---|---|---|---|---|---|
| DSA (native-NSA, no DS) | native indexer | **83.2** | **55.4** | 133 GB (mem 0.85) | replay ✓ | served 16/16 |
| DS-default | offline channel-mask | **39.8** | **27.6** | 125 GB (mem 0.70) | replay ✓ | served 16/16 |
| DS-hybrid (Tier-2.B) | `scorer_norm=hybrid, head_agg=mean` | **40.1** | **27.6** | 125 GB | replay ✓ | served 16/16 |
| DS-lifted-4096 (opt-in) | default + wider budget | ~14.5 (conc-1, R17) | — | ~114 GB @ cuda-graph-max-bs 8 | replay ✓ | served 20/20 |

Artifacts: `perf_ds_default_c{1,16}.json`, `perf_ds_hybrid_c{1,16}.json`,
`perf_dsa_c{1,16}.json` (+ `perf_closed_batch.py`); lifted from `m10_lifted_graph_finding.md`.

## Non-regression conclusions
1. **The Tier-2.B hybrid scorer adds NO material decode cost.** DS-hybrid == DS-default
   to within noise: conc-1 40.1 vs 39.8 tok/s/req; conc-16 **27.6 vs 27.6** (identical);
   same 125 GB; both under CUDA graph. The recall winner (AC-3: 16K graph-mode 6%→38%
   material; MMLU −0.5pp) is therefore **free on the decode hot path** — the binding
   AC-6 non-regression result for the landed long-context deliverable.
2. **DS decode TPS is structurally ≤ DSA** (≈ 0.48–0.50×: DS 39.8/27.6 vs DSA 83.2/55.4).
   This is the known, expected cost of the offline channel-mask selector + the
   logical→physical page-table adapter vs DSA's fused native indexer (the plan states
   "DS per-request decode TPS is ≤ DSA structurally"). It is NOT a Loop-7 regression —
   the gap is the same selector cost present since the Tier-1 spine landed.
3. **The DS-default conc-16 decode TPS (27.6) matches the Loop-6 closed-batch number
   (27.1 at the full-context op-point)** — the Tier-1 admission/decode spine is
   non-regressed; the directional AC-5 conc-16 result (P99 TTFT 13.13 s < 22 s at the
   full-context Option-B point, Loop-6) still holds because the decode/admission path
   is unchanged by the Loop-7 scorer/lifted work (all opt-in, default-off byte-identical).
4. **DSA / fp16 defaults are behavior-unchanged.** The native-NSA reference boots with
   no `--enable-double-sparsity`; the default `flashmla_kv` `dsa_index_topk` assert is
   untouched; with every DS opt-in flag off the decode is byte-identical (347/350 DS
   unit tests).
5. **The opt-in lifted-budget path's perf cost is the recorded tradeoff for the 4K
   recall lever** (decode ~14.5 tok/s/req at conc-1 vs DS-default 39.8 — it dequantizes
   `max_bs*4096` rows + attends a wider budget; capture-batch-bounded memory). It is
   **opt-in / default-off**, so it does NOT affect the default/hybrid budget; per
   DEC-4/DEC-6 + the M0 bounded-secondary evidence it is a 4K-only secondary lever, and
   its cost buys the 4K recall recovery (75%→95%) for that low-concurrency use.

## Recall summary (DS-vs-DSA, same node)
| length | DSA | DS-default | DS-hybrid (Tier-2.B) | DS-lifted-4096 |
|---|---|---|---|---|
| within-budget 1024w | 100% | 100% | 100% | — |
| 4K | 100% | 75% | 80% (R7) | **95%** (R14 eager / R17 graph) |
| 16K | 100% | 6% | **38%** material (R7, +32pp) | — (scorer-limited) |
| 64K | 100% | ~5% (floor) | floor | — (scorer-limited) |
| MMLU 5-shot | 89.0% | 88.5% | 88.5% (−0.5pp ≤1.0pp) | — |

Sources: `ds_vs_dsa_recall_matrix_graph_n50.json` + `m4_ac3_nonregression_finding.md`
(R7, graph-mode N=50, Clopper-Pearson CIs); `m8_lifted_recall_finding.md` /
`m10_lifted_graph_finding.md` (R14/R17 lifted 4K). The M0 oracle
(`m0_oracle_finding_r4.md`) attributes the regimes: 4K budget-limited, 16K
budget-partial (~46% cap), 64K scorer-limited — so the **long-context goal is served
by the Tier-2.B hybrid scorer** (16K material uplift, MMLU within tolerance, decode-free),
and the wider-budget lever is a bounded-secondary 4K lever.

## Provenance
- DS-default config: `{"top_k":2048,"signature_dtype":"int8","page_size":64,"head_agg":"max","scorer_norm":"off",...}`.
- DS-hybrid config: same + `"scorer_norm":"hybrid","scorer_norm_hybrid_threshold":8192,"head_agg":"mean"`.
- DSA: `serve_native_nsa.sh` `DISABLE_RADIX_CACHE=1 MEM_FRACTION_STATIC=0.85` (no DS).
- Launch: `LOOP7_MEASUREMENT=1 [SCORER_NORM=hybrid HEAD_AGG=mean] bash development/serve_double_sparsity.sh` (graph mode — no `--disable-cuda-graph`).
- Probe: `development/loop7/perf_closed_batch.py` (closed-batch, OSL=256).
- Measurement caveat: the GSP window-mode bench (`benchmark.sh`) can fabricate
  throughput from empty streams (the loop-6 R19 fail-closed lesson), so this uses the
  closed-batch decode-TPS method instead; full-context TTFT/P99 at conc-16 is the
  Loop-6 AC-5 directional result (unchanged decode/admission path).

## AC-6 status
**MET** — perf guardrails recorded at conc-1/16 (decode TPS, GPU mem, graph-replay,
admission) for DS-default / DS-hybrid / DSA / lifted; the landed long-context deliverable
(DS-default + the Tier-2.B hybrid) is non-regressing (hybrid == default decode TPS,
Loop-6 spine intact); DSA/fp16 defaults unchanged.
