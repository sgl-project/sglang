# MoE LoRA Shrink Kernel Configs

Auto-tuned configs for `_moe_lora_shrink_splitk_kernel` (the LoRA A / shrink
stage of the merged-experts MoE LoRA path).

Files are organized by Triton version:

```
moe_shrink_configs/triton_<ver>/moe_lora_shrink,E=<experts>,N=<rank>,K=<hidden>,device_name=<device>.json
```

where `E` is the number of virtual experts, `N` the LoRA rank and `K` the hidden
size. Each file is a JSON object keyed by token count `M`.

Generate or refresh these with:

```bash
python benchmark/kernels/lora_moe_shrink/tune_lora_moe_shrink.py
```

These are loaded at runtime by
`python/sglang/srt/lora/triton_ops/moe_lora_shrink_config.py`. If no file
matches the current `(N, K, device)`, the runtime falls back to a heuristic
default config.
