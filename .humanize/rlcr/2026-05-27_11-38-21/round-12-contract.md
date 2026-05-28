# Round 12 Contract

## Mainline Objective

Fix three remaining AC-4 calibration gaps identified by Codex Round 11 review so
`task-ac4-calibrate` is coding-complete and `task-ac4-hwrun` can proceed safely.

## Target ACs

- **AC-4** — Fix 3-D hook output shape, implement exact Pile-val token-block recipe,
  update calibration doc and module header. CI suite ≥ 187 passed (185 + 2 new).

## Blocking Issues

1. `_extract_mla_nope_prefix` assumes 2-D inputs; real HF projection hook outputs are
   3-D `[batch, seq, width]`. Fix: `flat = tensor.reshape(-1, tensor.shape[-1])`.

2. Pile-val path returns raw text truncated to `block_size`; must produce exactly
   `num_samples` concatenated fixed-size blocks via tokenizer. Fix: add
   `_build_pile_val_token_blocks(tokenizer, num_blocks, block_size, seed)` called
   after tokenizer loads in `_collect_channel_importance`.

3. `docs/advanced_features/double_sparsity_calibration.md` and the `calibrate.py`
   module header still advertise K-only L2 + NIAH recipe — must be updated before
   the H200 run.

## Queued (Out of Scope This Round)

- `task-ac4-hwrun`: H200 hardware run (analyze/Codex)
- `task-ac5-tp`, `task-ac6-cuda-graph`: next coding tasks after AC-4 is closed
- Hardware / analyze gates: AC-1, AC-1b, AC-8, AC-9–12

## Success Criteria

1. `_extract_mla_nope_prefix` flattens all leading dimensions via
   `flat = tensor.reshape(-1, tensor.shape[-1])` before per-head reshape; handles
   `[T, W]`, `[B, T, W]`, and any shape where `shape[-1] == H*(nope+suffix)`.

2. Add `_build_pile_val_token_blocks(tokenizer, num_blocks, block_size, seed)` that:
   - Loads Pile-val, shuffles with `seed`
   - Tokenizes each doc with `add_special_tokens=False`
   - Concatenates token IDs across document boundaries
   - Splits into exactly `num_blocks` blocks of `block_size` tokens each
   - Raises `RuntimeError` if total tokens < `num_blocks * block_size`

3. In `_collect_channel_importance`: when `use_pile_val=True`, call
   `_build_pile_val_token_blocks` after the tokenizer is loaded; feed resulting
   `[1, block_size]` tensors directly to `model(input_ids=...)`.

4. In `calibrate()`: production path (no `--dataset`, no `--allow-synthetic`) passes
   `use_pile_val=True` to `_collect_channel_importance` instead of constructing
   text prompts.

5. Two new tests in `TestCalibrateMethod1`:
   - `test_3d_hook_output_handled`: kv_b_proj and q_b_proj return `[1, T, width]` (3-D);
     verify importance[0] is finite and correct (same values as 2-D case).
   - `test_pile_val_blocks_concatenate_across_docs`: fake tokenizer + 3 short docs
     (200 tokens each); verify `_build_pile_val_token_blocks(..., num_blocks=1,
     block_size=512, ...)` returns a `[1, 512]` block spanning document boundaries;
     test fails if docs are merely truncated.

6. `docs/advanced_features/double_sparsity_calibration.md` and `calibrate.py` module
   header updated to: Method 1 Q/K noPE, Pile-val seed=42 256×512, no NIAH default.

7. Full suite: `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` ≥ 187 passed, 0 failed.
