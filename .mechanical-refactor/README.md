# Kernel-namespace migration — reproducible transform scripts

These are the reproducible transform scripts for the mechanical parts of the
`sglang.kernels` migration (RFC #29630), per the `mechanical-refactor-verify`
convention. Each script moves a set of kernels into `sglang.kernels.ops.<group>`
and rewrites all import sites; the move is defined by the script, not the diff.

Run from the repo root. Each script has two modes:

```bash
# Reproduce/verify a migration commit exactly (spins a worktree at BASE,
# re-runs the transform, diffs against the committed result):
python3 .mechanical-refactor/transform_kernels_migrate_clean.py
python3 .mechanical-refactor/transform_kernels_migrate_lora.py
python3 .mechanical-refactor/transform_kernels_migrate_trtllm.py

# Apply to the current tree (used to produce the commit in the first place):
python3 .mechanical-refactor/transform_kernels_migrate_clean.py apply
```

Each script's `BASE_COMMIT` / `TARGET_COMMIT` pin the commit it reproduces:

| Script | Scope | Modules |
|---|---|---|
| `..._clean.py`  | attention/mem_cache/model_executor/layers/constrained/speculative triton_ops → kvcache/attention/memory/activation/grammar/speculative; deletes dead `models/triton_ops/deepseek_v4.py` | 31 |
| `..._lora.py`   | `lora/triton_ops` → gemm/moe (+ moves `csgmv_configs`) | 13 |
| `..._trtllm.py` | experimental `lora/trtllm_lora_temp/triton_ops` → `{gemm,moe}.trtllm_lora_temp` | 7 |

These scripts are refactor tooling, not shipped code; they can be dropped (or
moved to a gist) before final merge.
