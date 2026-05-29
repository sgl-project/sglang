# DeepSeek-V3.2 Double Sparsity MVP — Evidence Bundle

Loop 5 final evidence index. All paths are relative to
`runs/20260528_dsv32_mvp/` unless noted. Assembled Round 11.

## Headline claim (honest)

- **TIER 1 — Smoke MVP: COMPLETE.** DS-on V3.2 FP8 serves real requests on 8×H200 at the locked
  Option B operating point, produces genuinely sparse decode selection, has a DS benchmark + a
  matching DSA benchmark at a labeled smoke shape, and passes the paired quality smoke (AC-Q).
  DS short-context quality is identical to DSA (AC-12 MMLU 5-shot: 89.00% == 89.00%).
- **TIER 2 — Loop4-compatible MVP: INCOMPLETE.** AC-10 (radix-on final serving), AC-11
  (directional comparator), AC-6 (CUDA-graph status), AC-1b (chunked-prefill probe) are done, and
  AC-12 MMLU passes — but **AC-12 long-context NIAH hard-fails** (DS recall is `top_k=2048`-bounded;
  DS cannot serve 64K at the mem-0.6 operating point) and **AC-11's directional TTFT target is
  missed** (admission-bound at mem 0.6). Per the plan's Ultimate Goal, with AC-12 full quality not
  met this is a **useful smoke milestone, not the minimal viable loop4 version** — recorded, not
  reclassified.

## Operating point (locked "Option B", plan §13 / DEC-1)

`tp_size=8`, `kv_cache_dtype=fp8_e4m3`, `page_size=64`, `chunked_prefill_size=8192`,
`dsa_prefill_backend=flashmla_kv`, `dsa_decode_backend=flashmla_kv`,
`disable_overlap_schedule=true`, `disable_piecewise_cuda_graph=true`. Regular CUDA graphs ON
(52 batch sizes captured). DS `mem_fraction_static=0.6` (per-rank TokenLabelTable overhead;
0.7 OOMs during generation); DSA `mem_fraction_static=0.85`. The DS↔DSA mem-fraction gap is the
sanctioned asymmetry (BL-20260529-ds-vs-dsa-memfraction-admission-asymmetry).

## Mask provenance (root artifact for every DS-on AC)

- Channel mask: `/models/dsv32-fp8-channel-mask.safetensors`
  `sha256 = a63c5b9555e2a7f3c8675343604098850cd2098f30576af5617e2484c99da244`
  (L=61, H=128, label_dim=16; dtype fp8_e4m3; top_k=2048).
- Calibration load fix (deepseek_v3 remap + Triton FP8 fallback + fail-closed dry-run):
  `calibration_provenance.md`, `ROUND0_dryrun_finding.md`; calibrate log
  `/tmp/dsv32-fp8-channel-mask.calibrate.log` (node 0). Mask validation: `mask_validation.txt`.
- The mask SHA above matches `ds_radix_fixture_state.json.config.channel_mask_sha256` — the AC-10
  radix authorization is bound to this exact mask.

## Acceptance Criteria — status + evidence

| AC | Status | Evidence |
|----|--------|----------|
| AC-0 producer fix (radix-capture meta) | MET | `ac0_capture_probe.json`, `ac0_capture_positive.json`; producer regression. |
| AC-4 real calibrated mask | MET | mask + SHA above; `calibration_provenance.md`, `mask_validation.txt`. |
| AC-1 DS boot + `/generate` + invalid-mask reject | MET | `ac1_server_info.json`, `ac1_generate.json`, `ds_boot_knobs_AC1.json`, `ac1_invalid_mask_rejection.md`. |
| AC-1.1 genuine sparse decode (>top_k) | MET | `ac1_1_genuine_sparsity.json` (sparsity_rate 0.105, selected 2048, dense_fallback 0). |
| AC-1b chunked-prefill probe | MET | `ac1b_probe.json`, `ac1b_server_info.json` (10565-tok multi-chunk prefill, needle recalled at radix-on point). |
| AC-6 CUDA-graph status | MET | recorded ON (52 batches) in DS/DSA `*_server_info.json` + boot logs; piecewise graph OFF by design. |
| AC-8 / AC-9 smoke benchmarks + comparator | MET | `smoke_results/*.meta.json` (6 sidecars), `mvp_compare.md`, `mvp_compare_c{16,32,64}.{md,json}`. |
| AC-10 radix-on final serving (no env override) | MET | `ds_radix_fixture_state.json`, `ac10_label_capture.json`, `ac10_fp8_scale_stability.json`, `ac10_radixon_server_info.json` (disable_radix_cache=false, authorized by artifact). |
| AC-11 directional comparator | EXECUTED — **directional MISS** | `ac11_analysis.md`, `mvp_compare_ac11.{md,json}`, `ac11_results/*.meta.json` (18 sidecars), `ac11_{ds,dsa}_server_info.json`. See below. |
| AC-Q paired quality smoke | MET | `ac_q_analysis.md`, `dsv32_quality_smoke_concise.json` (all four gates pass, 19/20 exact). |
| **AC-12 full quality gate** | **HARD FAIL** | `ac12_analysis.md`, `ac12_results/` (MMLU pass; NIAH 4K/16K/64K fail). See below. |

## Raw JSONL locations (gitignored) + committed sidecars

Raw `bench_serving` JSONLs are gitignored (large); the per-run `.meta.json` sidecars and the
comparator JSON/Markdown are committed and carry every locked field + the measured metrics.

- AC-8/9 smoke: raw `runs/20260528_dsv32_mvp/smoke_results/*.jsonl` (node 0, local) →
  sidecars `smoke_results/*.meta.json` (6).
- AC-11 sweep: raw `runs/20260528_dsv32_mvp/ac11_results/*.jsonl` (node 0, local, 18 files, each
  `duration ≥ 602s`) → sidecars `ac11_results/*.meta.json` (18) + `mvp_compare_ac11.{md,json}`.
- AC-12: per-gate JSON artifacts `ac12_results/ac12_{mmlu_5shot,niah_4096,niah_16384,niah_65536}_*.json`
  (all four committed) + `ac12_results/ac12_pytest_summary.txt` + `ac12_results/ds_boot_log_excerpt.txt`.
  The 64K artifact (`ac12_niah_65536_*.json`, Round-12 #L artifact-safe path) records DSA served
  20/20 at 100% recall and DS `ds_served=0` / `verdict=FAIL` with the HTTP 400 rejection body
  (`Input length (69970 tokens) exceeds the maximum allowed length (53050 tokens)`).

## Server args / server_info

`ac1_server_info.json`, `ac1b_server_info.json`, `ac10_radixon_server_info.json`,
`ac11_{ds,dsa}_server_info.json`, `ac12_{ds,dsa}_server_info.json`, `ds_smoke_server_info.json`,
`dsa_smoke_server_info.json` — captured via `/get_server_info` (the endpoint that crashed on DS
private CUDA-tensor attrs pre-loop5-R3; fixed by filtering `_`-prefixed attrs).

## AC-10 label-capture provenance note

The `ac10_label_capture.json` fixture authorizes radix-on by recording cold==warm DS label SHAs
with `cached_tokens>0`. Caveat preserved from Round 8: some `server_args` snapshots in the early
capture fixtures show `server_args=null` / a stale boot SHA because the capture predates a reboot;
the AUTHORITATIVE binding is the config fingerprint in `ds_radix_fixture_state.json` (model_path,
tp_size, page_size, kv_cache_dtype, channel_mask_sha256), which the validator re-verifies
fail-closed at every boot before permitting radix-on. The AC-10/AC-11/AC-12 DS servers all booted
radix-on authorized solely by that artifact (no `SGLANG_DS_RADIX_OVERRIDE`).

## AC-11 — directional comparator (executed; directional target missed)

3-trial radix-on sweep, conc 16/32/64, 120s warmup / 600s window. **DS per-request TPS is
competitive-to-better** (DS/DSA ratio 0.726/0.900/**1.146**), but **DS P99 TTFT misses** the
≤1.10× target (57.7/132.9/292.0 s vs DSA 0.73/1.37/2.04 s). The miss is **admission/queue-bound**:
DS achieved 91%/77%/56% of nominal concurrency (mem-0.6 KV pool) vs DSA ~100% (#F surfaced in the
comparator's effective-vs-nominal table). Per DEC-7 this is a recorded AC-11 failure + follow-up,
**not** a build-break. Detail + follow-up: `ac11_analysis.md`.

## AC-12 — full quality gate (HARD FAIL)

| Gate | DSA | DS | Δ | Thr | Verdict |
|------|-----|-----|---|-----|---------|
| MMLU 5-shot (200) | 89.00% | 89.00% | 0.00 pp | ≤1.0 | **PASS** |
| NIAH 4K | 100% | 75% | 25.0 pp | ≤5 | FAIL |
| NIAH 16K | 100% | 5% | 95.0 pp | ≤5 | FAIL |
| NIAH 64K | served 20/20 (100%) | 0/20 served — HTTP 400 unservable | 100.0 pp | ≤5 | FAIL |

Two mechanisms (both real, neither a bug): (1) DS sparse decode is `top_k=2048`-bounded → needle
recall degrades monotonically with context (BL-20260529-ds-longcontext-needle-recall-vs-topk,
quantified); (2) DS at mem 0.6 has `max_total_num_tokens=53,056` < the 69,970-token 64K prompt →
cannot admit 64K (DSA pool = 910,784). Full detail: `ac12_analysis.md`. AC-12 is hard pass/fail
(DEC-7 directional handling is AC-11-only) → **recorded as a hard failure**.

## Bottom line

DS-on DeepSeek-V3.2 FP8 is demonstrably correct and quality-preserving for normal-length requests
and serves at the locked Option B point with radix-on production knobs (TIER 1 complete). The
loop4-compatible MVP is **not** complete: DS's defining `top_k`-sparse decode bounds long-context
needle recall (AC-12 NIAH) and its mem-0.6 KV budget bounds both max concurrency (AC-11 TTFT) and
max context (AC-12 64K). These are recorded as the AC-11 directional miss and the AC-12 hard
failure, with follow-ups filed in `ac11_analysis.md` / `ac12_analysis.md`.
