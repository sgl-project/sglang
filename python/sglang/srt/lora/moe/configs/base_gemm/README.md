# MoE LoRA base-GEMM launch-config store

M-bucketed JSON tables consumed by `gemm_config_store.load_config_table`.
One file per (provider, geometry, device); the provider key names the
weight dtype, so the file name carries no separate dtype field:

```
provider={cutedsl_bf16_masked|cutedsl_bf16_contiguous},E={E_local},N1={gate_up_slices*I},N2={H},K={H},device_name={NVIDIA_...}.json
```

Payload keys are `expected_m` buckets (nearest-M lookup); each bucket payload
carries `token_width`. The optional `tiles` list declares the tile set to
compile at attach — one `persistent_clusters` per `token_width`. The
optional `version` map (e.g. `{"cutedsl": "..."}`) is checked against the
installed package; a mismatch warns and falls back to the heuristics.

No file, or an invalid file, means the providers use their built-in
heuristics — byte-identical to a build without this directory.

Generate entries on the target device (do not hand-write):

```
python benchmark/kernels/lora_moe/sweep_masked_gemm_configs.py \
    --num-local-experts <E_local> --hidden-size <H> \
    --intermediate-size <I> --gate-up-slices 2 --top-k <K> \
    --output-dir python/sglang/srt/lora/moe/configs/base_gemm
```

`SGLANG_LORA_MOE_CONFIG_DIR` names an override config root at load time;
tables are read from its `base_gemm/` subdirectory (the layout
`tune_lora_config.py --sweep` emits).
