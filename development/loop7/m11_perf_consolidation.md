# M11 — AC-6 perf consolidation + DS-vs-DSA recall/perf/non-regression report

Consolidated guardrail report for Loop 7 (the source artifact for the task20
decision record). All measurements are at the Loop-7 op-point under **CUDA graph**:
DS int8 / `mem_fraction_static=0.7` / fp8-KV / TP=8 / page 64 / `flashmla_kv`
prefill+decode / radix-off / `--disable-overlap-schedule` / `--disable-piecewise-cuda-graph`.
GPU: 8× NVIDIA H200 (sm90), TP=8. **Commit provenance** (verified via `git log`;
`f9f6ec056`=R18, `68969deb0`=R19, `30173f08b`=R20 — R19 and R20 commits touched only
`development/loop7/`, so the DS/DSA production serving code is unchanged across R18→R19→R20):
the **R19 decode-TPS** servers were launched from the R18 tree `f9f6ec056`, with the R19
probe+artifacts uncommitted, committed as `68969deb0`; the **R20 TTFT** servers were launched
from the R19 tree `68969deb0`, with the `--stream` probe uncommitted, committed as
`30173f08b`. Each `ttft_*.json` carries a per-run `run_provenance` block (launch command,
effective config, mem, graph evidence, served count, GPU, commits) reconstructed in R21.

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

## TTFT guardrails (conc-1 / conc-16, graph mode) — R20
Fresh time-to-first-token measured by the **streaming** mode of the same closed-batch
probe (`perf_closed_batch.py --stream`): per request TTFT = first-streamed-token arrival −
submit; it mirrors the canonical SGLang SSE parser (`data: {"text": <cumulative>,
"meta_info": {"completion_tokens": N}}`) and **fails closed on an HTTP-200 empty stream**
(never records a no-token response as a completion — the loop-6 R19 fail-closed lesson).
Two closed-batch prompt regimes at the same Loop-7 op-point, under CUDA graph: a SHORT
prompt (the R19 decode cross-check) and a ~770-token prompt (a prefill-bound TTFT
guardrail; stays in the dense-prefill regime, < the 2048 DSA prefill threshold). TTFT in
**milliseconds**; conc-16 reports p50 / p99 across the 16 concurrent requests.

| variant | c1 short | c16 short (p50 / p99) | c1 ~770-tok | c16 ~770-tok (p50 / p99) | graph | served |
|---|---|---|---|---|---|---|
| DSA (native-NSA) | 150.8 | 307.1 / 309.2 | 150.9 | 1161.5 / 1322.1 | replay ✓ | 16/16 |
| DS-default | 183.3 | 371.7 / 374.0 | 180.4 | 1210.9 / 1400.2 | replay ✓ | 16/16 |
| DS-hybrid (Tier-2.B) | 178.4 | 363.3 / 365.1 | 177.7 | 1218.1 / 1405.2 | replay ✓ | 16/16 |

Artifacts: `ttft_{ds_default,ds_hybrid,dsa}_c{1,16}.json` (short) +
`ttft_{ds_default,ds_hybrid,dsa}_c{1,16}_p770.json` (~770-tok prefill); each carries the
per-request `ttft_ms_all` array + mean/p50/p99/min/max.

**Streaming decode-TPS cross-check** (the same `--stream` run also yields a *clean*
post-first-token decode TPS = `(completion_tokens − 1) / (t_last − t_first)`, which —
unlike the R19 e2e number — excludes prefill+first-token, so it is the theoretically
correct pure-decode rate and runs slightly higher): DSA **87.3 / 58.7**, DS-default
**40.8 / 28.5**, DS-hybrid **41.1 / 28.5** (conc-1 / conc-16). This reproduces the R19
closed-batch ordering and the DS ≈ 0.48–0.49× DSA structural ratio, validating both
methods.

### TTFT findings (non-regression)
1. **DS-hybrid TTFT ≈ DS-default TTFT** at every point (178 vs 183 ms c1; 363 vs 372 ms
   c16-short p50; 1218 vs 1211 ms c16-p770 p50 — all within run-to-run noise). The
   Tier-2.B hybrid scorer adds **no material TTFT cost** — the same decode-free result the
   R19 decode-TPS table showed, now confirmed on the first-token latency too.
2. **DS TTFT is modestly above DSA** (~+30 ms c1, ~+60 ms c16-short, ~+50–80 ms c16-p770) —
   the small per-step cost of the DS selection + logical→physical adapter, the same
   structural overhead as the decode-TPS gap; NOT a Loop-7 regression. In the prefill-bound
   c16-p770 case TTFT is dominated by prefill and DS ≈ DSA + ~5%.
3. **Every measured TTFT is far below the Loop-6 directional ceiling** (P99 22 s): the
   heaviest measured point (DS conc-16, ~770-tok prefill) is P99 ≈ **1.4 s** — over an
   order of magnitude under budget. (The Loop-6 directional P99 13.13 s < 22 s was at the
   much longer full-context Option-B prompt; that point is unchanged because the
   admission/decode path is untouched by the opt-in/default-off Loop-7 work, so the
   directional result still holds and the fresh op-point TTFT shows no regression.)

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
   non-regressed. The **fresh R20 conc-1/16 TTFT** (see the TTFT table: DS-default P99
   374 ms short / 1.40 s at ~770-tok prefill; DS-hybrid within noise of it) confirms no
   first-token-latency regression at the measured op-point, and is far under the P99 22 s
   ceiling. The Loop-6 directional AC-5 result (P99 TTFT 13.13 s < 22 s at the much longer
   full-context Option-B point) still holds as the historical full-context reference
   because the decode/admission path is unchanged by the Loop-7 scorer/lifted work (all
   opt-in, default-off byte-identical).
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
| 4K | 100% | 80% (graph N=50, R7) | 80% (R7) | **95%** (R14 eager / R17 graph) |
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
- Probe: `development/loop7/perf_closed_batch.py` — closed-batch, OSL=256; default mode
  for the R19 e2e decode-TPS, `--stream` mode for the R20 TTFT + clean post-first-token
  decode-TPS.
- Measurement caveat: the GSP window-mode bench (`benchmark.sh`) can fabricate
  throughput from empty streams (the loop-6 R19 fail-closed lesson), so this uses the
  closed-batch probe; the `--stream` TTFT path adopts the same fail-closed guard (an
  HTTP-200 empty stream raises, never recorded as a completion). The R20 TTFT here is the
  fresh conc-1/16 guardrail at the Loop-7 op-point; the Loop-6 full-context P99 13.13 s
  is retained only as the historical full-context directional reference.

## AC-6 status
**MET** — the full conc-1/16 guardrail set required by the plan
(`refined_plan_v1.md:80`: **TTFT, decode TPS/req, GPU memory, graph-replay success,
admission**) is recorded for DS-default / DS-hybrid / DSA (+ lifted decode TPS): the R19
decode-TPS/mem/graph/admission table **and** the R20 fresh conc-1/16 TTFT table above. The
landed long-context deliverable (DS-default + the Tier-2.B hybrid) is non-regressing on
**both** decode TPS (hybrid == default, 27.6 == 27.6 conc-16) **and** TTFT (hybrid ≈
default within noise); every measured TTFT is far under the P99 22 s ceiling; the Loop-6
Tier-1 spine is intact; DSA/fp16 defaults are behavior-unchanged.
