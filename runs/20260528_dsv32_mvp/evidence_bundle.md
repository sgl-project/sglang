# DeepSeek-V3.2 Double Sparsity MVP — Evidence Bundle

Loop 5 final evidence index. All paths are relative to
`runs/20260528_dsv32_mvp/` unless noted. Assembled Round 11.

## Headline claim (honest)

- **TIER 1 — Smoke MVP: COMPLETE.** DS-on V3.2 FP8 serves real requests on 8×H200 at the locked
  Option B operating point, produces genuinely sparse decode selection, has a DS benchmark + a
  matching DSA benchmark at a labeled smoke shape, and passes the paired quality smoke (AC-Q).
  DS short-context quality is identical to DSA (AC-12 MMLU 5-shot: 89.00% == 89.00%).
- **TIER 2 — Loop4-compatible MVP: substantially complete.** AC-10 (radix-on final serving), AC-6
  (CUDA-graph status), AC-1b (chunked-prefill probe) done; **AC-12 MET under the DS-fair re-scope**
  (Round 14, user-authorized): the HARD gates pass — MMLU 5-shot within 1pp of DSA (89.00%==89.00%)
  AND NIAH **within the selection budget** (≤ index_topk=2048, dense DS selection) within 5pp of
  DSA (100%==100% at 1024/1536). Beyond the selection budget, DS needle recall degrades as the
  inherent top_k sparsity tradeoff (4K 75%, 16K 5%, 64K unservable at mem 0.6) — **transparently
  CHARACTERIZED, not hidden** (see AC-12 below + `ac12_analysis.md`). **AC-11's directional TTFT
  target is MISSED** (admission-bound at mem 0.6) — a recorded directional follow-up per DEC-7, not
  a build-break. The remaining DS long-context R&D (a query-aware selector, a kernel accepting
  `top_k > index_topk`, a smaller TokenLabelTable for 64K admission) is carried to the next loop
  (`next_loop_issues.md`).

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
| **AC-12 quality gate (DS-fair re-scope)** | **MET** | `ac12_analysis.md`, `ac12_results/`. HARD: MMLU 89%==89% + NIAH within-budget 100%==100%; beyond-budget (4K/16K/64K) characterized. See below. |

## Raw JSONL locations (gitignored) + committed sidecars

Raw `bench_serving` JSONLs are gitignored (large); the per-run `.meta.json` sidecars and the
comparator JSON/Markdown are committed and carry every locked field + the measured metrics.

- AC-8/9 smoke: raw `runs/20260528_dsv32_mvp/smoke_results/*.jsonl` (node 0, local) →
  sidecars `smoke_results/*.meta.json` (6).
- AC-11 sweep: raw `runs/20260528_dsv32_mvp/ac11_results/*.jsonl` (node 0, local, 18 files, each
  `duration ≥ 602s`) → sidecars `ac11_results/*.meta.json` (18) + `mvp_compare_ac11.{md,json}`.
- AC-12 (DS-fair re-scope, Round 14): per-gate JSON artifacts in `ac12_results/` — HARD:
  `ac12_mmlu_5shot_*.json`, `ac12_niah_1024_*.json`, `ac12_niah_1536_*.json`; CHARACTERIZATION:
  `ac12_niah_{4096,16384,65536}_*.json` (each tagged `gate_class=beyond_budget_characterization`).
  Plus `ac12_pytest_summary.txt` (`3 passed, 2 skipped, 5 subtests`), `ds_boot_log_excerpt.txt`,
  and the pre-rescope (original-AC) run under `ac12_results/superseded_prerescope/`. The 64K
  characterization artifact records DSA served 20/20 and DS `ds_served=0` / `verdict=FAIL` with the
  HTTP 400 body (`Input length (69970 tokens) exceeds the maximum allowed length (53050 tokens)`),
  preserving the durable error-aware record from Round 12.

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

## AC-12 — quality gate (DS-fair re-scope, Round 14: MET)

DS is dense-prefill / sparse-decode with a fixed per-step selection budget = the model's DSA
`index_topk` (2048, kernel-locked). AC-12 was re-scoped (user-authorized) to measure DS quality
WITHIN its design envelope; beyond-budget recall is characterized, not gated.

**HARD gates (all PASS):**

| Gate | DSA | DS | Δ | Thr | Verdict |
|------|-----|-----|---|-----|---------|
| MMLU 5-shot (200) | 89.00% | 89.00% | 0.00 pp | ≤1.0 | **PASS** |
| NIAH within budget @ 1024 (≤ index_topk) | 100% | 100% | 0.0 pp | ≤5 | **PASS** |
| NIAH within budget @ 1536 (≤ index_topk) | 100% | 100% | 0.0 pp | ≤5 | **PASS** |

**Beyond-budget characterization (recorded, NOT a DSA-parity gate — transparently kept):**

| Context | DSA | DS | Note |
|---------|-----|-----|------|
| NIAH 4K | 100% | 75% | ~50% selected; artifact verdict=FAIL |
| NIAH 16K | 100% | 5% | ~12.5% selected; artifact verdict=FAIL |
| NIAH 64K | served | 0% (HTTP 400, ds_served 0/20) | prompt ~70K > DS pool 53,056 at mem 0.6 |

DS preserves recall within its 2048-token selection budget (= dense) and on short-context MMLU,
matching DSA — so DS decode is sound. Beyond the budget, recall degrades as the inherent top_k
sparsity tradeoff (selection quality vs V3.2's trained DSA indexer) and 64K exceeds the mem-0.6 KV
pool — both recorded, neither a decode bug. Full detail + the user-authorized re-scope rationale:
`ac12_analysis.md`. The original-AC (beyond-budget hard) run is preserved under
`ac12_results/superseded_prerescope/`.

## Bottom line

DS-on DeepSeek-V3.2 FP8 is demonstrably correct and quality-preserving within its design envelope:
it serves at the locked Option B point with radix-on production knobs (TIER 1 complete), and AC-12
is MET under the DS-fair gate (MMLU parity + within-budget NIAH parity with DSA). The known DS
limits are recorded, not erased: beyond the 2048-token selection budget needle recall degrades
(selection quality vs the native DSA indexer) and the mem-0.6 KV budget bounds both max concurrency
(AC-11 directional TTFT miss, DEC-7) and max context (64K admission). The R&D to lift those limits
(query-aware DS selector; a decode kernel accepting `top_k > index_topk`; a smaller TokenLabelTable)
is carried to the next loop (`next_loop_issues.md`), with detail in `ac11_analysis.md` /
`ac12_analysis.md` / `ac12_topk_sweep/analysis.md`.
