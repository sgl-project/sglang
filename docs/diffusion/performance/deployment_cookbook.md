# Deployment Cookbook

This page gives practical defaults for choosing CPU offload, FSDP, CFG parallelism, SP, and TP.

## Quick Rule

Use the simplest setting that fits your memory target:

| Goal | Recommended setting |
|---|---|
| Fastest single-GPU run when the model fits | Disable CPU offload and do not use FSDP. |
| Lower single-GPU memory usage | Use component CPU offload, or layerwise DiT offload for supported Wan/MOVA models. |
| Faster multi-GPU Qwen/Wan CFG generation | Use FSDP with CFG parallelism and disable CPU offload. |
| Sequence length or video-shape scaling | Use SP/Ulysses/Ring when the model benefits from sequence parallelism. |
| TP compatibility or encoder-heavy paths | Set TP explicitly; do not treat TP as the default latency optimization. |

Base the decision on available memory on the selected GPU(s), not only the device's total capacity. For multi-GPU runs, the least-free selected GPU is the bottleneck. A busy 80GiB GPU can behave like a much smaller GPU.

FSDP shards DiT weights across multiple GPUs. It is not useful for keeping a single-GPU deployment on one GPU; for that case use CPU offload.

## Performance Modes

`--performance-mode` applies safe presets without overriding explicit offload, FSDP, or parallelism flags. `--mode` is a short alias.

| Mode | Meaning |
|---|---|
| `auto` | Default. Applies only high-confidence defaults, currently multi-GPU Qwen/Wan CFG models. |
| `throughput` | Favors GPU-resident execution. Disables CPU offload when unset; may OOM. Alias: `aggressive`. |
| `memory` | Favors lower GPU memory. Uses component offload, or Wan/MOVA layerwise DiT offload when supported. Alias: `conservative`. |
| `balanced` | Keeps existing single-GPU defaults; on validated multi-GPU Qwen/Wan CFG models prefers FSDP+CFG. Alias: `balance`. |

`auto` and `balanced` check selected GPU memory before applying GPU-resident FSDP+CFG defaults. In multi-GPU runs they use the least available memory across selected GPUs. `throughput` intentionally does not check memory; it is the mode for users who prefer speed and accept OOM risk.

The preset is intentionally coarse. A future continuous value such as `0.0` to `1.0` could express the speed-memory tradeoff more precisely, but it would need model-specific memory models and clearer user expectations. Until then, use the preset plus explicit flags for overrides.

Examples:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --num-gpus 2 \
  --performance-mode balanced
```

```bash
sglang generate \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --performance-mode memory
```

Explicit flags win over the mode:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --num-gpus 2 \
  --performance-mode balanced \
  --use-fsdp-inference false
```

In this example, `balanced` will not re-enable FSDP. The same applies to parallelism; for example, `--enable-cfg-parallel false` keeps CFG parallelism disabled.

## Interpreting The Levers

**No offload** keeps model components resident on GPU. It is usually fastest when memory is sufficient.

**Component CPU offload** lowers GPU memory by moving large components to CPU. It is simple and robust, but it usually trades latency for memory.

**Layerwise DiT offload** lowers DiT memory further for supported Wan/MOVA models by moving DiT layers between CPU and GPU. It can be the best single-GPU memory mode, but may reduce throughput.

**FSDP** shards DiT weights across multiple GPUs and all-gathers weights during forward. It can reduce CPU offload cost on multi-GPU deployments, especially when combined with CFG parallelism for Qwen/Wan.

FSDP sharding granularity matters. SGLang Diffusion prefers sharding direct repeated transformer block entries such as `transformer_blocks.0` or `blocks.0`. Coarser sharding lowers wrapper count but can increase all-gather peak memory; finer sharding can reduce transient memory but adds communication and scheduling overhead. If a model does not define an explicit sharding rule, the loader falls back to repeated block class names and common direct numbered block paths.

**CFG parallelism** splits positive and negative CFG branches across GPUs. For Qwen/Wan workloads with normal step counts, this is the most reliable multi-GPU speedup observed so far.

**SP/Ulysses/Ring** splits sequence work. It can help video workloads, but validated Qwen/Wan runs showed CFG parallelism outperforming SP for latency.

**TP** is supported for compatibility and some model structures, but current measurements do not make it the default latency path for Qwen/Wan.

## Current Benchmark Takeaways

Observed regular-scale trends:

- Z-Image: single-GPU no-offload was faster than FSDP/SP in the tested setting; keep FSDP off unless memory or parallelism requires it.
- Qwen-Image: FSDP+CFG on 2 GPUs reduced latency from about 12.7s to 6.7s in the tested 1024x1024, 50-step run.
- Wan 1.3B: FSDP+CFG on 2 GPUs reduced latency from about 47.8s to 26.7s in the tested 832x480, 81-frame, 50-step run.
- Component offload mainly reduced memory; it did not improve latency in the tested no-offload-vs-offload runs.

Always benchmark with your actual resolution, frame count, step count, and GPU type before locking production defaults.
