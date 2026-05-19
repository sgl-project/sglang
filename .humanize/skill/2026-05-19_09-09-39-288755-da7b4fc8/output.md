AGREE:
- The restart recommendation is sound: PR #22992 is not a viable base, and PR #25304 should be treated as a kernel/calibration reference, not an upstream branch.
- HiSparse is the right integration surface. Adding `--enable-double-sparsity` or a second sparse coordinator would fight the repo direction.
- `NSABackendAdaptor.adapt_for_attn_metadata` is correctly identified as a central blocker.
- Dense + native NSA baselines are required before any SLO claim is meaningful.
- The plan is right to keep top-p/Twilight ABI pressure visible from day one.

DISAGREE:
- AC-3 page-size negative test: Claude says `page_size=64` with `bfloat16` + `flashmla_kv` should fail; I say this is the wrong assertion. Current validation pairs `bfloat16` with `flashmla_sparse` and `fp8_e4m3` with `flashmla_kv`. The negative test should cover explicit backend/dtype mismatch, not imply BF16+page64 is invalid.
- AC-10 metrics: Claude says a normal run should show non-zero `dense_fallback_total`; I say healthy DS should have `dense_fallback_total == 0`. Non-zero fallback belongs in an injected-failure test.
- AC-10 vs AC-4: Claude says missing labels fail startup in AC-4, but AC-10 says missing labels produce runtime fallback metrics. Pick one. For production DS, fail-fast is better; fallback should only be tested via an explicit internal fault injection.
- AC-11 / Lower Bound: Claude says top-p is deferred, but AC-11 requires top-p end-to-end. I say AC-11 should be ABI-only unless Twilight top-p is explicitly moved into initial scope.
- Milestone 1 Phase B: Claude says wire `quest` onto MLA as a smoke test. I say use `deepseek_nsa` or a synthetic adaptor test. Quest-on-MLA is extra scope and not supported by the stated repo facts.
- DEC-5: Claude says V3.2 DS replaces “DSA” selection while keeping DSA compressor/dequant plumbing. I say this conflates repo paths. The facts distinguish DeepSeek-V3.2 NSA under `layers/attention/nsa/` from DeepSeek-V4 DSA under `layers/attention/dsv4/`.

REQUIRED_CHANGES:
- 1. Add an explicit coordinator-integration task before AC-2/task3. The plan assumes `SparseCoordinator` and `HiSparseCoordinator` are already one path, but the repo fact says they coexist and are not wired together.
- 2. Rewrite DEC-5 using repo terms: for DeepSeek-V3.2, define exactly whether DS replaces the native NSA indexer/selection path, and which `nsa/` quant/dequant/cache components remain authoritative.
- 3. Fix AC-3’s BF16 negative test to assert backend/dtype validation, not that BF16 page64 is invalid.
- 4. Resolve the Python packaging conflict: the plan cannot create both `algorithms/double_sparsity.py` and `algorithms/double_sparsity/calibrate.py`. Use a package directory or a separate calibration module name.
- 5. Split “calibration artifact” from “runtime label cache.” Offline artifacts should describe calibration/schema/channel selection; runtime `K_label`/page labels must be generated or maintained per served KV page, especially with radix cache hits.
- 6. Make DEC-2 a hard blocker for AC-7/AC-8. With today’s `disable_radix_cache` assertion, the stated ~55% prefix-cache workload cannot be validated.
- 7. Fix AC-8/AC-9 uses of `algorithm=none`. The registry does not have `none`; dense baseline should mean HiSparse disabled unless the plan explicitly adds a reporting-only label.
- 8. Clarify AC-8 metric semantics before convergence: hardware, TP/EP, mean vs P50 per-request output tok/s, and whether TTFT includes queueing.
- 9. Fix AC-9’s negative test: “disabling sparsity does not change MMLU” is not a useful DS regression guard. Keep corrupted-label NIAH sensitivity; treat MMLU as a secondary quality check.
- 10. Move ABI shape work from Milestone 6 into Milestone 3/task6. Buffer shape, `valid_lengths`, and max-bounded layout must be fixed before CUDA graph capture and kernel benchmarking.
- 11. Move a small version of task15 before task5. GLM-5/128K/FP4 can remain deferred, but artifact schema choices must not block them later.
- 12. Fix AC-12’s allowed paths. Required work may touch `managers/hisparse_coordinator.py`, scheduler/meta_info plumbing, metrics code, and benchmark utilities outside the current whitelist.
- 13. Fix DEC-4 vs Milestone 4: if artifacts are not shipped in repo, “produce + version artifact” must mean external deployment artifact plus documented recipe, not a committed file.
- 14. Adjust AC-5’s positive test. Running DeepSeek-V3.2 FP8 with `--tp 1` is not a credible small-fixture test for the real model; use a tiny NSA fixture for CI and a production recipe for agreed hardware.

OPTIONAL_IMPROVEMENTS:
- Replace mutation-style tests like “mutating adaptor to `pass` causes failure” with direct assertions that adapted metadata reaches the backend.
- Add a “feature unavailable until real selector” guard so the placeholder top-k skeleton cannot accidentally pass benchmarks.
- Build `benchmark_baseline.sh` earlier; it is independent of DS kernels and will sharpen SLO discussions.
- Rename AC-10 per-request fields directly, e.g. `sparsity_rate`, `selected_pages`, `dense_fallback`, without referencing `accept_rate`.
- Correct DEC-2 text where it says “AC-5 cache+sparsity coexistence”; that is really AC-7/AC-8.

UNRESOLVED:
- DEC-1: exact SLO hardware, TP/EP shape, and metric definition.
- DEC-2: radix cache compatibility versus renegotiating the workload.
- DEC-3: quality thresholds are proposed, not agreed.
- DEC-4: who owns production calibration artifacts and where they are versioned.
- DEC-5: exact semantic replacement point inside DeepSeek-V3.2 NSA.
- DEC-6: whether top-p is ABI-only now or behavior-required now.
- DEC-7: “Extensions” interpretation is acceptable as `sparse_extra_config`, but still needs user confirmation.
