# Deployment Cookbook

This page gives practical defaults for choosing CPU offload, FSDP, CFG parallelism, SP, and TP.

## Quick Rule

Use the simplest setting that fits your memory target:

| Goal                                       | Recommended setting                                                                |
|--------------------------------------------|------------------------------------------------------------------------------------|
| Fastest single-GPU run when the model fits | Disable CPU offload and do not use FSDP.                                           |
| Lower single-GPU memory usage              | Use component CPU offload, or layerwise DiT offload for supported Wan/MOVA models. |
| Faster multi-GPU Qwen/Wan CFG generation   | Use FSDP with CFG parallelism and disable CPU offload.                             |
| Sequence length or video-shape scaling     | Use SP/Ulysses/Ring when the model benefits from sequence parallelism.             |
| TP compatibility or encoder-heavy paths    | Set TP explicitly; do not treat TP as the default latency optimization.            |

Base the decision on available memory on the selected GPU(s).

- For multi-GPU deployment: the least-free selected GPU is the bottleneck. A busy 80GiB GPU can behave like a much smaller GPU.
- For single-GPU deployment: FSDP shards DiT weights across multiple GPUs. It is not useful for keeping a single-GPU deployment on one GPU; for that case use CPU offload.

## Performance Modes

`--performance-mode` applies safe presets without overriding explicit offload, FSDP, or parallelism flags. `auto` is the default. Use `manual` when you need to keep performance-related server args under explicit user control. `--mode` is a short alias.

| Mode       | Meaning                                                                                                                   |
|------------|---------------------------------------------------------------------------------------------------------------------------|
| `manual`   | Keeps performance-related server args under explicit user control.                                                       |
| `auto`     | Default. Keeps legacy safe offload defaults and uses FSDP/CFG only on validated multi-GPU deployments where FSDP can replace DiT offload. |
| `speed`    | Favors GPU-resident execution for lower latency and higher throughput. Disables CPU offload when unset; may OOM.           |
| `memory`   | Favors lower GPU memory. Uses component offload, or Wan/MOVA layerwise DiT offload when supported.                         |

`auto` checks selected GPU memory before applying FSDP. In multi-GPU runs it uses the least available memory across selected GPUs, and only turns on FSDP automatically when doing so can replace DiT offload. Text encoder, image encoder, and other component residency still follow the offload policy unless the model marks a high-memory resident path as safe. When the model default uses CFG and the user did not set a parallelism policy, `auto` may also enable CFG parallelism. `speed` intentionally does not check memory; it is the mode for users who prefer latency/throughput and accept OOM risk.

The modes tune residency for native pipeline components declared to the component residency manager. Today this covers the major DiT, text/image encoder, VAE, vocoder, and upsampler components; DiT can use layerwise offload when supported, while text encoders use either resident execution or component CPU offload. Do not assume text-encoder layerwise offload unless a model implements and validates it.

NOTE:
The preset is intentionally coarse. A future continuous value such as `0.0` to `1.0` could express the speed-memory tradeoff more precisely, but it would need model-specific memory models and clearer user expectations. Until then, use the preset plus explicit flags for overrides.

Examples:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --num-gpus 2 \
  --performance-mode auto
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
  --performance-mode auto \
  --use-fsdp-inference false
```

In this example, `auto` will not re-enable FSDP. The same applies to parallelism; for example, `--enable-cfg-parallel false` keeps CFG parallelism disabled.

## Interpreting The Levers

**No offload** keeps model components resident on GPU. It is usually fastest when memory is sufficient.

**Component CPU offload** lowers GPU memory by moving large components to CPU. It is simple and robust, but it usually trades latency for memory.

**Layerwise DiT offload** lowers DiT memory further for supported Wan/MOVA models by moving DiT layers between CPU and GPU. It can be the best single-GPU memory mode, but may increase latency and lower throughput.

**FSDP** shards DiT weights across multiple GPUs and all-gathers weights during forward. It can reduce DiT CPU offload cost on multi-GPU deployments, especially for validated Wan I2V workloads.

FSDP sharding granularity matters. SGLang Diffusion prefers sharding direct repeated transformer block entries such as `transformer_blocks.0` or `blocks.0`. Coarser sharding lowers wrapper count but can increase all-gather peak memory; finer sharding can reduce transient memory but adds communication and scheduling overhead. If a model does not define an explicit sharding rule, the loader falls back to repeated block class names and common direct numbered block paths.

**CFG parallelism** splits positive and negative CFG branches across GPUs. For Qwen/Wan workloads with normal step counts, this is the most reliable multi-GPU speedup observed so far.

**SP/Ulysses/Ring** splits sequence work. It can help video workloads, but validated Qwen/Wan runs showed CFG parallelism outperforming SP for latency.

**TP** is supported for compatibility and some model structures, but current measurements do not make it the default latency path for Qwen/Wan.

## Current Benchmark Takeaways

Observed regular-scale trends:

- Z-Image: single-GPU no-offload was faster than FSDP/SP in the tested setting; keep FSDP off unless memory or parallelism requires it.
- Qwen-Image: keep the default non-FSDP path unless a specific FSDP/SP/Ring setting has been benchmarked on the target hardware.
- Wan: FSDP can replace DiT offload on validated multi-GPU workloads, while text/image encoders may still need component offload. Keep model-specific precision checks before making FSDP automatic for a path.
- Component offload mainly reduced memory; it did not improve latency in the tested no-offload-vs-offload runs.

Always benchmark with your actual resolution, frame count, step count, and GPU type before locking production defaults.
