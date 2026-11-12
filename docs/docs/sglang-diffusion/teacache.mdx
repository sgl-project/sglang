---
title: "TeaCache Acceleration"
description: "Configure TeaCache for temporal similarity-based diffusion acceleration."
tag: "approx"
---

> **Note**: This is one of several caching strategies in SGLang (Cache-DiT, TeaCache, Spectrum, plus exact conditioning reuse).
> For an overview of all caching options, see [caching](./caching-acceleration).

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

## Overview

TeaCache works by:
1. Tracking the L1 distance between modulated inputs across consecutive timesteps
2. Accumulating the rescaled L1 distance over steps
3. When accumulated distance is below a threshold, reusing the cached residual
4. Using separate positive/negative caches for supported CFG model families

## How It Works

### L1 Distance Tracking

At each denoising step, TeaCache computes the relative L1 distance between the current and previous modulated inputs:

```text
rel_l1 = |current - previous|.mean() / |previous|.mean()
```

This distance is then rescaled using polynomial coefficients and accumulated:

```text
accumulated += poly(coefficients)(rel_l1)
```

### Cache Decision

- If `accumulated >= threshold`: Force computation, reset accumulator
- If `accumulated < threshold`: Skip computation, use cached residual

### CFG Support

For models that support CFG cache separation, TeaCache maintains separate caches for positive and negative branches:
- `previous_modulated_input` / `previous_residual` for positive branch
- `previous_modulated_input_negative` / `previous_residual_negative` for negative branch

Flux and Qwen do not separate CFG caches and do not implement TeaCache yet (see [Supported Models](#supported-models)).

## Configuration

TeaCache is configured via `TeaCacheParams` in the sampling parameters. Start from the matching model's calibrated parameters; this example changes the threshold for `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` while preserving its coefficients and skip window:

```python Configuration
from dataclasses import replace
from sglang.multimodal_gen.configs.sample.wan import WanT2V_1_3B_SamplingParams

params = replace(
    WanT2V_1_3B_SamplingParams().teacache_params,
    teacache_thresh=0.1,
)
sampling_params_kwargs = {"enable_teacache": True, "teacache_params": params}
```

`--enable-teacache` / `enable_teacache` use the model's built-in `TeaCacheParams`. To customize them from Python, pass `sampling_params_kwargs` to `DiffGenerator.generate()`.

### Parameters

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "28%"}} />
    <col style={{width: "14%"}} />
    <col style={{width: "58%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Parameter</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>Type</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`teacache_thresh`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>float</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Threshold for accumulated L1 distance. Higher = more caching, faster but potentially lower quality</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`coefficients`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>list[float]</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Polynomial coefficients for L1 rescaling, highest power first (`c[0]*x**4 + c[1]*x**3 + c[2]*x**2 + c[3]*x + c[4]`). Model-specific tuning</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`start_skipping`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>int or float</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Steps before this always compute. A float is a fraction of the denoising steps; a negative int counts back from the end. Default `5`</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>`end_skipping`</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>int or float</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Skipping stops at this step (exclusive); same encoding as `start_skipping`. Default `-1`</td>
    </tr>
  </tbody>
</table>

### Model-Specific Configurations

Different models may have different optimal configurations. The coefficients are typically tuned per-model to balance speed and quality.

## Supported Models

TeaCache support status by model family:

<table style={{width: "100%", borderCollapse: "collapse", tableLayout: "fixed"}}>
  <colgroup>
    <col style={{width: "34%"}} />
    <col style={{width: "28%"}} />
    <col style={{width: "38%"}} />
  </colgroup>
  <thead>
    <tr style={{borderBottom: "2px solid #d55816"}}>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Model Family</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.05)"}}>CFG Cache Separation</th>
      <th style={{textAlign: "left", padding: "10px 12px", fontWeight: 700, whiteSpace: "nowrap", backgroundColor: "rgba(255,255,255,0.02)"}}>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Wan2.1</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Full support</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Wan2.2</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>Yes</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Coefficients are not calibrated yet; enabling TeaCache is accepted but currently no-ops</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Z-Image</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>—</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Not wired yet; <code>--enable-teacache</code> is accepted but currently no-ops</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>HunyuanVideo</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>No</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>Not supported yet</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Flux</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>No</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>To be supported</td>
    </tr>
    <tr>
      <td style={{padding: "9px 12px", fontWeight: 500, backgroundColor: "rgba(255,255,255,0.02)"}}>Qwen</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.05)"}}>No</td>
      <td style={{padding: "9px 12px", backgroundColor: "rgba(255,255,255,0.02)"}}>To be supported</td>
    </tr>
  </tbody>
</table>


## References

- [TeaCache: Accelerating Diffusion Models with Temporal Similarity](https://arxiv.org/abs/2411.14324)
