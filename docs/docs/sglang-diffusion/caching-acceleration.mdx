---
title: "Caching Acceleration"
description: "Compare caching acceleration strategies for diffusion models."
---
SGLang can reuse exact conditioning across requests and cache intermediate denoising results. Conditioning reuse preserves the encoded values; denoising caches such as Cache-DiT, TeaCache, and Spectrum use approximations.

## Conditioning cache

Native pipelines enable a conditioning cache by default. Repeated inputs can reuse text embeddings (including negative prompts), image or vision-language encoder outputs, and VAE posteriors. The shared LRU holds at most **512 MiB per executor/rank** across CPU and device entries, with at most 128 entries. The common text stage keeps consumed negative conditioning on its original device to avoid repeated CPU-to-GPU transfers and postprocessing. Other entries live on the CPU. Oversized results are not admitted.

Within a grouped or batched stage, the same conditioning boundary can reuse a device result before consulting the cross-request cache. The common text stage and the Qwen-Image 2.1, Ming-Image, and LingBot-Video text stages share read-only consumed tensors and give each request its own containers. Other cached boundaries return private copies, with device snapshots limited to 512 MiB and 128 entries per stage. These temporary references are released when the stage finishes or fails. Group reuse remains available when cross-request caching is disabled or its budget is too small.

The standard text encoding stage stores the final embeddings, pooled outputs, masks, and sequence lengths. A hit skips encoder weight preparation, token uploads, and mask construction; CPU tokenization still runs to match the inputs. Qwen-Image 2.1 also caches its consumed prompt embeddings and image-token slots before processor execution, while retaining separate vision-feature reuse for changed edit instructions. Other custom stages and batch-DP encoding retain caching at the native encoder boundary.

CPU hits restore outputs to their original devices; device hits return private device copies. Negative conditioning can therefore retain GPU memory between requests, within the shared cache budget. Measure memory as well as latency when using a tight VRAM budget.

This is most useful when generating several variations from the same prompt or reference image, especially with a layerwise-offloaded encoder. A changed edit instruction recomputes the joint vision-language embedding, while the unchanged image can still reuse its vision features and VAE posterior. Separate CLI invocations start separate processes and cannot share this cache. Denoising and output decoding still run for every request, so end-to-end gains depend on how much time the original request spends encoding conditioning.

To disable cross-request conditioning reuse:

```bash
sglang serve --model-path Qwen/Qwen-Image-2.1 --disable-conditioning-cache
```

To increase the shared cache budget:

```bash
sglang serve --model-path Qwen/Qwen-Image-2.1 --conditioning-cache-max-size-mb 1024
```

The same settings are accepted by `DiffGenerator.from_pretrained()` as `disable_conditioning_cache` and `conditioning_cache_max_size_mb`. A capacity of `0` also disables both cross-request tiers. Neither setting disables conditioning reuse within a grouped stage or changes per-output seeds.

### Matching and correctness

- Keys include the loaded encoder instance, actual input contents, shapes, dtypes, masks, and autocast precision. Image tensors and PIL images are matched by content, so replacing a file at the same path cannot reuse stale pixels. Alpha is part of the VAE input.
- Positive and negative inputs use the same bounded cache. Different seeds can reuse conditioning; initial noise is still generated for each request.
- VAE caching stores the posterior before sampling and normalization. Every request still calls `sample()` or `mode()` as its model requires, preserving the random-number sequence.
- Cross-request hits return independent copies. Grouped consumed outputs may share read-only tensors; other boundaries isolate mutable tensors and posteriors. In-place normalization and downstream mutations cannot change a persistent entry.
- Encoder TP and folding groups agree on hits before skipping collective work. Weight updates and LoRA changes invalidate the cache, including failed updates that may have modified weights.
- Warmup refreshes cached results and executes each distinct encoding within a group once. This primes kernels while preserving multi-output reuse; subsequent requests can reuse conditioning such as the model-default negative prompt. Compare disabled, cold, and repeated-input latency separately when benchmarking.

### Coverage and limits

The common encoder boundary covers native CLIP, T5/UMT5, Llama, Mistral 3, Gemma 2/3, Qwen 2.5-VL, Qwen 3/3-VL, Ideogram, and MiniMax-H3 encoders, including custom pipeline stages. Qwen-VL and Gemma 3 also cache vision features independently of text. Hunyuan3D image encoders and the LTX-2.3 video condition encoder participate. Native image, video, and audio VAE encode entry points cache deterministic tensors or supported posterior containers.

Cosmos3 has no separate text encoder: its deterministic understanding pathway caches immutable per-layer K/V results, which are copied into fresh request state. SenseNova-U1 caches reference-image vision features; its noisy-image vision path still runs at each denoising step. VLA keeps its model-owned observation/prefix cache.

FSDP disables cross-request caching and caching inside sharded modules. Consumed text-conditioning outputs still support group-local reuse, with all ranks agreeing before skipping the encoder. The common text stage also reuses consumed outputs from library fallback encoders within a group and keeps their negative conditioning device-hot across requests. Batch-DP can skip encoding and its output gather only when all participating ranks have the consumed result; this includes device-hot negatives. Otherwise each encoder copy retains its normal cache and gather behavior.

Stateful autoregressive calls (`use_cache=True` or existing `past_key_values`), training/autograd, CUDA Graph capture, unsupported tensor or output types, and distributed VAE encoding bypass the shared caches. External SRT encoders and the Diffusers pipeline backend keep their own cache behavior. The global disable flag also disables existing realtime text and VLA prefix reuse; those model-specific caches otherwise retain their own memory settings.

Joint vision-language embeddings depend on both the prompt and image. The cache does not reuse a previous turn's combined embedding when either changes, and does not carry mutable denoising KV state between requests.

Dynamic batching still decides which requests can execute together. Conditioning reuse does not change encoder batch shapes, padding, or output order: a batched encoder call is matched as a whole, rather than split into individually cached rows. Model-specific grouped execution that also prepares materials, broadcasts results, or manages state retains its own execution rules.

The common text stage also keeps group-local stage deduplication: equivalent requests execute the stage once and receive their own output containers. This avoids repeating tokenization and stage bookkeeping. When stage deduplication leaves only one computation, it also skips the temporary group cache and its hit synchronization; cross-request caching remains available. The conditioning cache shares encodings between otherwise different requests, including a common negative prompt. Custom stages that prepare request-specific metadata or realtime session state use encoder-result reuse while still preparing each request.

## Overview

The following approaches reduce denoising computation:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "18%"}} />
    <col style={{width: "18%"}} />
    <col style={{width: "42%"}} />
    <col style={{width: "22%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Strategy</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Scope</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Mechanism</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Best For</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Cache-DiT</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Block-level</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Skip individual transformer blocks dynamically</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Advanced, higher speedup</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>TeaCache</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Timestep-level</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Skip entire denoising steps based on L1 similarity</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Simple, built-in</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Spectrum</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Timestep-level</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Forecast DiT features to skip selected denoising steps</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Experimental, model-validated tuning</td>
    </tr>
  </tbody>
</table>

## Cache-DiT

[Cache-DiT](https://github.com/vipshop/cache-dit) provides block-level caching with
advanced strategies like DBCache and TaylorSeer. It can achieve up to **1.69x speedup**.

See [Cache-DiT](./cache_dit) for detailed configuration.

<Note>
Cache-DiT currently cannot be combined with `--use-fsdp-inference`. Keep FSDP disabled when enabling Cache-DiT. DiT layerwise offload is compatible: skipped blocks are not streamed, and the first layer after a skip may sync-load.
</Note>

### Quick Start

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

### Key Features

- **DBCache**: Dynamic block-level caching based on residual differences
- **TaylorSeer**: Taylor expansion-based calibration for optimized caching
- **SCM**: Step-level computation masking for additional speedup

## TeaCache

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

See [TeaCache](./teacache) for detailed documentation.

### Quick Overview

- Tracks L1 distance between modulated inputs across timesteps
- When accumulated distance is below threshold, reuses cached residual
- Uses separate positive/negative caches for supported CFG model families

### Supported Models

- Wan2.1
- Z-Image
- Wan2.2: coefficients are not calibrated yet; enabling TeaCache is accepted but currently no-ops
- HunyuanVideo: not supported yet

For Flux and Qwen models, TeaCache is automatically disabled when CFG is enabled.

## Spectrum

Spectrum forecasts DiT features and skips selected denoising steps. It is
approximate and currently applies only to selected native implementation paths.

See [Spectrum Acceleration](./spectrum) for supported model families,
constraints, and request controls.


## References

- [Cache-DiT Repository](https://github.com/vipshop/cache-dit)
- [TeaCache Paper](https://arxiv.org/abs/2411.14324)
