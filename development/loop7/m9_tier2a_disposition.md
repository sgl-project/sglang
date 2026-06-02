# M9 — Tier-2.A (lifted-budget decode) landing disposition (AC-4 close)

> **STATUS: PRODUCTION-READY landed (R16–R17).** The R15 review rejected the
> earlier deferred-with-evidence close and required task16 production hardening to
> be implemented — it now is. The lifted-budget decode runs under production CUDA
> graph (graph-safe, zero-alloc, perf + graph-mode recall measured). The
> bounded-secondary / DEC analysis below remains the record of WHY a wider budget
> only helps 4K; the disposition is now a **production-ready landing**, not a deferral.

## Decision (production-ready landing)
Tier-2.A — the opt-in adjustable-budget ("lifted-budget") decode path — lands as an
**opt-in, default-off, CUDA-graph-safe production path** with measured recall + perf
evidence. This closes **AC-4** under the plan's "production-ready **or**
deferred-with-evidence" branch — via the **production-ready** branch (DEC-4).

The DSA production default and the Loop-6 Tier-1 operating point are **never
regressed**; every lifted-budget code path is opt-in and reversible (default-off
byte-identical).

## Production-ready landing evidence (R16–R17)
- **Graph-safe, alloc-free decode**: `dequantize_k_cache_paged_out` + the fixed-shape
  `build_lifted_compact_kv_fixed` (no dynamic `total_valid`) + a preallocated
  `DSGraphState` lifted scratch (compact KV / page table / compact indices / q
  head-padding). The wired backend `_forward_lifted_budget` replays **zero-alloc**
  under a real `torch.cuda.CUDAGraph` at **4096 and 8192**
  (`TestLiftedBudgetBackendGraphSafe`, `TestLiftedBudgetGraphSafe`), matching the
  eager reference.
- **Live production CUDA-graph capture**: the server boots WITHOUT
  `--disable-cuda-graph`; the validator's eager-only gate is removed (the path is
  graph-safe); decode batches log `cuda graph: True` (`m10_lifted_graph_finding.md`).
- **Graph-mode recall**: NIAH 4K, N=20, CUDA-graph mode → **95% (19/20)**, served
  20/20, 0 admission failures — matches the eager 95% and confirms the +20pp recovery
  over DS-default 2048 (~75%) holds in production graph mode.
- **Perf / memory**: graph mode is **3.4× faster** than eager (13.8 s vs 46.8 s for
  N=20; ~14.5 tok/s decode at conc-1, 4K); the lifted scratch is bounded by the
  capture batch (`--cuda-graph-max-bs`), ~70 MB at max_bs 8 — the 4K lever is a
  low-concurrency long-context op-point.
- **DSA default untouched**: the `flashmla_kv` `indices.shape[-1] == dsa_index_topk`
  (2048) assert is unchanged; default-off decode byte-identical; 347 DS unit tests pass.

## Bounded-secondary scope (unchanged — why the lever is 4K-only)
The M0 oracle (R4, `m0_oracle_finding_r4.md`) attributed the recall gap by regime:

| length | regime | score-only recall@2048 → @4096 → @8192 | wider-budget recoverable? |
|---|---|---|---|
| 4K | **budget-limited** | 44% → 86% → 100% | **yes** |
| 16K | budget-partial (caps ~46%) | 23% → 31% → 46% | partial, capped |
| 64K | **scorer-limited** | 15% → 20% → 24% | **no** (rank ≫ 8192) |

A wider decode budget therefore only recovers the **4K budget-limited** regime.
For the **long-context goal that motivated this loop, 16K is budget-partial
(capped at ~46%) and 64K is scorer-limited** (needle rank ≫ 8192) — a wider budget
alone recovers neither, so both are served by the **landed, production-ready
Tier-2.B hybrid scorer** (AC-3: 16K graph-mode 6% → 38% material; MMLU within
0.5pp of DSA; graph-safe + TP=8 + within-budget parity). So Tier-2.A is
**bounded-secondary**.

The served evidence (R14, `m8_lifted_recall_finding.md`) confirms the bounded 4K
recovery on the actual decode path:

- **DS-lifted `lifted_budget_top_k=4096` = 95% (19/20) [75.1, 99.9]**
- **DS-default `top_k=2048` = 75% (15/20) [50.9, 91.3]**
- **+20 pp, MATERIAL** (lifted 0.95 > default Clopper–Pearson CI-high 0.9134),
  both EAGER same-node, N=20, served 20/20, 0 admission failures.

(Historical note: at R15 this bounded-secondary framing was used to argue for a
deferred-with-evidence close. The R15 review **rejected** that and required the
production hardening to be implemented; it now is (R16–R17), so AC-4 closes via the
**production-ready** branch. The bounded-secondary analysis remains the record of
WHY a wider budget only helps 4K, not a deferral rationale.)

## DEC-4 / DEC-6 conditions — all satisfied by the production landing
1. **Recall evidence recorded.** M0 oracle (regime attribution) + the R14 served
   4K recovery (`m8_lifted_recall_finding.md`, `niah_ds_{lifted4096,default2048_eager}.json`,
   `ds_lifted_vs_default_recall_4k.json`).
2. **DSA default untouched.** The default `flashmla_kv` decode and its
   `indices.shape[-1] == dsa_index_topk` (2048) assert are unchanged. Every lifted
   path is behind a default-off `getattr(self, "ds_lifted_budget_decode", False)`
   guard; with the flag off the decode is **byte-identical** to the pre-Tier-2.A
   path (verified by the full DS unit suite: 341 + 9 subtests).
3. **Production-graph-safe (no longer eager-gated).** The path is CUDA-graph-safe —
   alloc-free `dequantize_k_cache_paged_out` + the fixed-shape compact builder + a
   preallocated `DSGraphState` scratch (R16–R17). `validate_double_sparsity` **no
   longer requires `--disable-cuda-graph`**; the wired backend `_forward_lifted_budget`
   replays **zero-alloc** under a real `torch.cuda.CUDAGraph` (4096/8192), and a live
   server captures + serves the lifted decode under CUDA graph (`m10_lifted_graph_finding.md`).
4. **The binding recall is the graph-mode number.** The CUDA-graph-mode served 4K
   recall is **95% (19/20)** (matching the eager 95% — the 4K budget is near-dense so
   the eager-vs-graph numerics shift is negligible); this is the binding production recall.

## Landed surface (opt-in, R10–R18)
- **ABI** (`config.py`): `enable_lifted_budget_decode: bool` + `lifted_budget_top_k: int`
  (R10); `lifted_budget_top_k % 128 == 0` enforced (R13, the `flash_mla_sparse_fwd`
  `topk % (2*B_TOPK)` block constraint); fail-closed set-without-flag / flag-without-budget.
- **Validator** (`validator.py`): fail-closed when the backend is unavailable (R11);
  enabled-path gating (R13): `top_k == index_topk`, `lifted_budget_top_k > index_topk`,
  `% 128`. **The `--disable-cuda-graph` requirement was REMOVED (R17)** — the path is
  graph-safe. (R18: a fail-closed guard also rejects lifted + speculative decode.) The
  reserved Twilight fields / `SGLANG_DS_ALLOW_TOPK_MISMATCH` are NOT the mechanism.
- **Selection width** (`selector.py`, `dsa_backend.py`): `max_top_k` / `ds_max_top_k`
  → `lifted_budget_top_k` when enabled; R23 deterministic tie-break unchanged; TP=8
  cross-rank selected-index equality at 4096/8192 by the composed (a)+(b)+(c) evidence
  in "DEC-4/DEC-6 conditions" item 5: (a) **single-rank** lifted-width
  `retrieve_topk_graph_safe` captured in a real CUDA graph (zero-alloc + bit-identical
  to eager, R18); (b) the **eager** 8-rank all-reduce equality (R14); (c) the **live**
  R17 8-rank server under production CUDA graph (95% recall).
- **Index core** (`lifted_budget.py`, R12): the eager `build_compact_decode_index`
  request-local physical→compact ordinal remap (`page_table_1_flattened` no `-1`;
  within-row dedup keep-highest-rank; prefix-sharing isolated) + the fixed-shape
  graph-safe `build_lifted_compact_index_fixed` / `build_lifted_compact_kv_fixed` (R16).
- **Decode branch** (`dsa_backend.py::_forward_lifted_budget`): graph path =
  `build_lifted_compact_kv_fixed` + alloc-free `dequantize_k_cache_paged_out` into a
  preallocated `DSGraphState` scratch + a q head-padding scratch (R16–R17); the eager
  `build_lifted_compact_kv` + allocating `dequantize_k_cache_paged` is the non-graph
  fallback. Both feed `flash_mla_sparse_fwd` (no 2048 cap) via `_forward_flashmla_sparse`.
- **Launcher** (`serve_double_sparsity.sh`): `LIFTED_BUDGET` knob emits the ABI fields;
  **R17 removed the forced `--disable-cuda-graph`** (the path runs under CUDA graph).
- **Tests**: CPU remap matrix + GPU kernel smokes (R12); served-helper 4096/8192 (R13);
  backend-level `_forward_lifted_budget` 4096/8192 + eager lifted-width TP=8 (R14);
  standalone + wired-backend CUDAGraph zero-alloc replay (R16–R17); single-rank
  lifted-width selection CUDA-graph capture (R18, `TestLiftedWidthSelectionGraphCaptured`)
  — the full graph-mode TP=8 determinism is the (a)+(b)+(c) composition above.

## task16 production-hardening scope — COMPLETED (R16–R17)
The following were required before the lifted path could enter production CUDA-graph
capture (i.e. before relaxing the validator's `--disable-cuda-graph` requirement).
**All are now landed** (R16 primitives, R17 backend wiring + validator relax + live
graph-mode confirmation; see `m10_lifted_graph_finding.md`):
1. **Alloc-free dequant**: a `dequantize_k_cache_paged_out(quant_k_cache,
   page_table_1_flattened, out)` variant that writes into caller-owned bf16 scratch
   (no internal `torch.empty`), backed by the existing Triton kernel shape.
2. **Graph-safe fixed-shape compact builder**: preallocate
   `page_table_1_flattened_scratch` (length `max_bs * lifted_budget_top_k`),
   `compact_indices_scratch [max_bs, lifted_budget_top_k]`, and
   `compact_kv_scratch [max_bs * lifted_budget_top_k, 1, 576]`. Invalid/duplicate
   lanes write a **safe physical slot** into the dequant input (never `-1`) and
   `-1` into `compact_indices` (so `flash_mla_sparse_fwd` masks them). The current
   `build_compact_decode_index` is eager/dynamic-length; this is the fixed-shape
   port (analogous to the R9 anchor tensorization, fuzz-validated bit-identical).
3. **q head-padding scratch** preallocated in the backend; the lifted branch routes
   through the scratch-backed path under capture.
4. **Zero-alloc-replay proof**: capture the lifted decode in a real
   `torch.cuda.CUDAGraph` and `assert_no_alloc_in_region(replay)` at 4096 and 8192;
   plus a **graph-mode** served recall re-measure (the eager 95% is not the graph
   number) and perf guardrails (decode TPS/req, GPU mem) vs the default.
5. **Graph-captured TP=8 lifted-width determinism** — PROVEN by three composed
   artifacts (R18):
   - (a) **Selection-under-capture**: `retrieve_topk_graph_safe` at lifted width
     4096/8192 captured in a real `torch.cuda.CUDAGraph`, replaying **zero-alloc**
     (`assert_no_alloc_in_region`) and **bit-identical to the eager logical
     reference** (`test_ds_scorer_tp_determinism.py::TestLiftedWidthSelectionGraphCaptured`).
   - (b) **Cross-rank all-reduce equality**: the eager 8-rank lifted-width
     selected-index / valid-length equality at 4096/8192 (`TestTP8LiftedWidthDeterminism`)
     — the SUM all-reduce is rank-symmetric and deterministic.
   - (c) **Live 8-rank under production capture**: the R17 TP=8 server ran the
     selection (incl. the all-reduce) **under CUDA graph** and served correct **95%**
     recall (`m10_lifted_graph_finding.md`) — divergent ranks would corrupt the
     all-reduced selection → degenerate output, which did not occur.
   Note: a **standalone 8-rank NCCL-collective-under-`torch.cuda.graph`** unit harness
   is infeasible — capturing an NCCL collective in a naive per-rank `torch.cuda.graph`
   deadlocks; that path requires the production `cuda_graph_runner`'s coordination
   (graph pool + comm registration), which is exactly what (c) exercises and validates
   end-to-end. (a)+(b)+(c) together establish the requirement.
6. Then relax the validator `--disable-cuda-graph` rejection for the lifted path. **Done.**

**Op-point scope (Codex review):** the graph-safety proof covers the **fp8-KV**
production op-point (the only place the `flashmla_kv` 2048 cap binds and the lifted
path is meaningful; H200/sm90). The bf16-store gather branch of
`build_lifted_compact_kv_fixed` is not covered by the R16/R17 graph tests — AC-4 is
scoped to the documented fp8 production path, not all validator-supported dtype/backend
pairs.

## Impact on the Ultimate Goal
**None on the long-context recall goal.** 16K is budget-partial (capped ~46%) and
64K is scorer-limited; a wider budget alone recovers neither, so both are served by
the landed, production-ready Tier-2.B hybrid scorer (AC-3, MET). The 4K-secondary
lever is itself **production-graph-landed** (no deferred item): the 4K recovery is
recorded, the lifted decode runs under CUDA graph, and the DSA default is untouched.
The strategic gate's original Tier-2.A-primary ordering is superseded by this
evidence (formal
supersession record = task20).

## Status
- **AC-4: closed via the PRODUCTION-READY branch** (DEC-4): the lifted-budget decode
  is CUDA-graph-safe and landed (zero-alloc replay + live graph capture + graph-mode
  95% 4K recall + 3.4× perf), the DSA default is untouched, and the path is opt-in /
  default-off byte-identical.
- **task16: COMPLETE** (R16 primitives + R17 backend wiring + validator relax + live
  graph-mode confirmation; `m10_lifted_graph_finding.md`).
- Reviews: re-reviewed via `/humanize:ask-codex` twice. R17
  (`.humanize/skill/2026-06-02_17-22-36-2899823-aa7f7f9e/output.md`): "No invalidating
  design gap found"; decode graph-safety sufficient for the fp8 landing. R18
  (`.humanize/skill/2026-06-02_18-49-10-2985908-ce861480/output.md`), after the R17
  review forced the full consistency rewrite + the graph-captured TP=8 artifact:
  **"No runtime/design gap found blocking AC-4"**, the **`(a)+(b)+(c)` composed
  evidence is an acceptable production-readiness close** (a raw per-rank
  `torch.cuda.graph` NCCL harness is NOT required), and the speculative fail-closed
  guard is sound. Integrated its wording fixes: corrected the two bullets that
  overclaimed "8-rank NCCL graph capture" to the exact `(a)+(b)+(c)` evidence, and
  cleaned the stale "eager-only" comments in `serve_double_sparsity.sh`,
  `selection_kernel.py`, and `dsa_backend.py`. (The R15 deferred draft was reviewed
  at `.humanize/skill/2026-06-02_16-34-28-2810456-a017c2ed/output.md`.)
