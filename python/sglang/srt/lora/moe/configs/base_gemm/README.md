# MoE LoRA base-GEMM launch-config store

M-bucketed JSON tables consumed by `gemm_config_store.load_config_table`.
One file per (provider, geometry, device); the provider key names the
weight dtype, so the file name carries no separate dtype field:

```
provider={cutedsl_bf16_masked|cutedsl_bf16_contiguous|cutedsl_fp8_masked|cutedsl_fp8_contiguous},E={E_local},N1={gate_up_slices*I},N2={H},K={H},device_name={NVIDIA_...}.json
```

Payload keys are `expected_m` buckets (nearest-M lookup); each bucket payload
carries `token_width`. The optional `tiles` list declares the tile set to
compile at attach — one `persistent_clusters` per `token_width`. The
optional `version` map (e.g. `{"cutedsl": "..."}`) records the package
release the table was tuned with. It is provenance, not a guarantee: with a
different installed release the table is still used (its tiles are compiled
and checked before use) and a warning names the table and both releases,
because the tile choices have not been re-validated on that release.

When the pinned compiler release changes:

1. List the tables whose recorded release differs (the startup warnings).
2. For each affected provider, geometry and device you can run, check the
   numerics against the FP32 reference and compare the table with the
   built-in heuristics on a representative serving shape.
3. Re-tune only where that evidence calls for it, and record the new release
   in `version`. Keep a note of the scopes you could not test.

Missing or malformed files use the provider's built-in heuristics.

`SGLANG_LORA_MOE_CONFIG_DIR` names an override config root at load time;
tables are read from its `base_gemm/` subdirectory.

The [maintained LoRA tuner](../../../../../../../benchmark/kernels/lora_moe/README.md)
currently searches LoRA launch tiles while retaining these base configurations.
It does not emit base-GEMM tables. The old partial-GEMM sweep is historical
screening evidence, not a complete-runner correctness or performance gate.
New tables need numerical checks and complete-runner validation on their exact
device/geometry; do not populate this directory from isolated stage timings.
