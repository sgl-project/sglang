# Round 11 Contract

## Mainline Objective

Fix two AC-4 calibration bugs identified by Codex Round 10 review:
1. MLA projection extraction uses flat-slice-then-reshape instead of reshape-then-slice, so Method 1 computes `Q*K` over V/RoPE columns for later heads.
2. The production calibration dataset path uses NIAH-synthetic prompts instead of the required Pile-val seed=42, 256×512 recipe.

## Target ACs

- **AC-4** — Fix MLA extraction helper + implement Pile-val dataset path + add sentinel regression tests; CI green at ≥183 passed.

## Blocking Issues

None outside the two mainline AC-4 gaps.

## Queued (Out of Scope This Round)

- `task-ac4-hwrun`: H200 hardware run (analyze/Codex; hardware not available here)
- `task-ac5-tp`: TP=2 multiprocess all-reduce test
- `task-ac6-cuda-graph`: graph capture coding
- Hardware / analyze gates: AC-1, AC-1b, AC-8, AC-9–12
- Doc update for `docs/advanced_features/double_sparsity_calibration.md` (queued; no hardware yet)

## Success Criteria

1. **MLA extraction fix**: introduce `_extract_mla_nope_prefix(tensor, num_heads, nope_dim, suffix_dim)` helper that reshapes to `[-1, num_heads, nope_dim + suffix_dim]` then slices `[..., :nope_dim]`. Used for K (`suffix_dim=v_head_dim`) and Q (`suffix_dim=qk_rope_head_dim`). `qk_rope_head_dim` derived from config (`head_dim - qk_nope_head_dim` for MLA). Falls back to direct reshape when `suffix_dim==0` (standard attention).

2. **Sentinel regressions** (2 new tests in `TestCalibrateMethod1`):
   - `test_mla_k_extraction_ignores_v_columns`: 2-head fixture, `kv_b_proj` output layout `[K0|V0|K1|V1]`; verify K importance uses only K_nope columns, not V columns. Test must fail under the old flat-slice implementation.
   - `test_mla_q_extraction_ignores_rope_columns`: 2-head fixture, `q_b_proj` output `[Q0_nope|Q0_rope|Q1_nope|Q1_rope]`; verify Q importance uses only Q_nope columns. Test must fail under the old flat-slice implementation.

3. **Pile-val dataset path**: when `allow_synthetic=False` and `--dataset` is not supplied, the production path loads `datasets.load_dataset("mit-han-lab/pile-val-backup", split="validation")`, shuffles with `seed=42`, tokenizes, concatenates, and yields exactly 256 blocks of 512 tokens. Custom `--dataset` override stays. `--allow-synthetic` keeps existing NIAH fallback. Parser adds `--block-size` (default 512) and `--seed` (default 42). Output metadata records `dataset_source`, `seed`, `num_samples`, `block_size`.

4. **CI path stays green**: `_collect_channel_importance(allow_synthetic=True)` is unchanged. Full suite: `PYTHONPATH=python pytest test/registered/unit/layers/attention/test_double_sparsity_unit.py -q` ≥ 185 passed (183 + 2 new tests), 0 failed.
