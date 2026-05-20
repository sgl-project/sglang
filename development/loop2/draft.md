Complete M1-C of the standalone Double Sparsity path on DeepSeek-V3.2 (FP8): write the page-table adapter that takes the DS selector's (selected_indices, valid_lengths) page-level tuple, maps logical page IDs to physical via `req_to_token` / `req_pool_indices`, and emits the FlashMLA `block_table` in sequence order. The adapter REPLACES the current NotImplementedError raise in DeepseekV2AttentionMLA._select_topk_indices's DS branch; it must also bypass the existing NSA topk_indices consumer on the DS branch (per AC-2, DS does not stack on NSA — it is an alternative selection path that drives FlashMLA directly).

Remove both the per-step NotImplementedError and the SGLANG_DS_ALLOW_NO_ADAPTER startup gate in validate_double_sparsity once the adapter is in place. The dev override env vars exported from serve_double_sparsity.sh become unused and are deleted from the launcher.

Wire the scheduler-side `customized_info_for_request` glue at the existing `customized_info` hook in tokenizer_manager.py (search for the symbol; line numbers drift) so the sglang_double_sparsity_* metrics actually surface in per-request meta_info.

Land the M3-B page-stability fixture in CI against a synthetic V3.2-shape fixture (the function already exists; the CI hook is the new piece). Provide a numbered operator runbook at development/loop2/RUNBOOK.md covering: (1) calibrate against /cluster-storage/models/deepseek-ai/DeepSeek-V3.2/ to produce the channel mask safetensors; (2) boot serve_double_sparsity.sh + serve_native_nsa.sh; (3) run benchmark.sh twice (DS + baseline) at the agreed concurrencies; (4) run benchmark_compare.py to produce the side-by-side SLO + quality report; (5) run M3-B on real hardware and decide DEC-2's default.

Out of scope (operator phase 2 / phase 3):
- The actual calibration run, benchmark run, and M3-B hardware run.
- Flipping the DEC-2 default based on the M3-B result.
- Capturing AC-8 SLO numbers and AC-9 NIAH/MMLU results.

Branch: dev/double-sparsity-standalone (continues from the same branch). Existing 87-test suite must stay green; add adapter unit tests (synthetic FlashMLA shape fixture verifies block_table emission), integration test that runs end-to-end through the adapter without raising, and a CI test that drives the M3-B fixture against a small synthetic prompt.

Acceptance criteria for the loop close:
- `--enable-double-sparsity` boots successfully without any SGLANG_DS_ALLOW_* override.
- A request through the DS branch reaches FlashMLA and returns a result without raising.
- meta_info on a DS request contains the sparsity_rate / selected_pages / dense_fallback fields.
- M3-B fixture has a CI hook (synthetic shape).
- development/loop2/RUNBOOK.md exists and is reviewable.
- 87+ unit tests pass; the test count grows by at least 5.
