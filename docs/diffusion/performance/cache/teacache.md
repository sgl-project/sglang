# TeaCache

> **Note**: This is one of two caching strategies available in SGLang.
> For an overview of all caching options, see [caching](../index.md).

TeaCache (Temporal similarity-based caching) accelerates diffusion inference by detecting when consecutive denoising steps are similar enough to skip computation entirely.

## Overview

TeaCache works by:
1. Tracking the L1 distance between modulated inputs across consecutive timesteps
2. Accumulating the rescaled L1 distance over steps
3. When accumulated distance is below a threshold, reusing the cached residual
4. Supporting CFG (Classifier-Free Guidance) with separate positive/negative caches

## Implementation

TeaCache is split into three classes:

- **`TeaCacheParams`** — pure data class holding user-set parameters (`rel_l1_thresh`, `coefficients`, `start_skipping`, `end_skipping`). Set once per request, never mutated during inference.
- **`TeaCacheState`** — dataclass holding runtime state for one CFG branch: `step`, `previous_modulated_input`, `previous_residual`, `accumulated_rel_l1_distance`.
- **`TeaCacheStrategy`** — all the logic. Owns two `TeaCacheState` objects (positive + optional negative CFG branch). Constructed once per generation by `CachableDiT.maybe_init_cache()` with all parameters resolved upfront.

At each denoising step, the model calls:
1. `cache.step(modulated_input)` — advances the step counter, accumulates the rescaled L1 distance, returns `True` if the forward pass can be skipped
2. `cache.read()` — if skipping, reads the cached residual and applies it to hidden states
3. `cache.write()` — if computing, stores the new residual in the cache
4. `cache.reset_states()` — resets `state` and optionally `state_neg`, discarding any stale tensors

### L1 Distance Tracking

At each denoising step, TeaCache computes the relative L1 distance between the current and previous modulated inputs:

```
rel_l1 = |current - previous|.mean() / |previous|.mean()
```

This distance is then rescaled using polynomial coefficients and accumulated:

```
accumulated += poly(coefficients)(rel_l1)
```

### Cache Decision

- If `accumulated >= threshold`: Force computation, reset accumulator
- If `accumulated < threshold`: Skip computation, use cached residual

### CFG Support

For models that support CFG separation (Wan, Hunyuan, Z-Image), `TeaCacheStrategy` maintains separate `TeaCacheState` objects for the positive and negative branches.

For models that don't support CFG separation (Flux, Qwen), TeaCache is automatically disabled when CFG is enabled.

## Configuration

TeaCache is configured via `TeaCacheParams` in the sampling parameters:

```python
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams

params = TeaCacheParams(
    rel_l1_thresh=0.1,           # Threshold for accumulated L1 distance
    coefficients=[1.0, 0.0, 0.0],  # Polynomial coefficients for L1 rescaling
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `rel_l1_thresh` | float | Threshold for accumulated L1 distance. Lower = more caching, faster but potentially lower quality |
| `coefficients` | list[float] | Polynomial coefficients for L1 rescaling. Model-specific tuning |

### Model-Specific Configurations

Different models may have different optimal configurations. The coefficients are typically tuned per-model to balance speed and quality.

## Supported Models

TeaCache is built into the following model families:

| Model Family | CFG Cache Separation | Notes |
|--------------|---------------------|-------|
| Wan (wan2.1, wan2.2) | Yes | Full support |
| Hunyuan (HunyuanVideo) | Yes | To be supported |
| Z-Image | Yes | To be supported |
| Flux | No | To be supported |
| Qwen | No | To be supported |


## References

- [TeaCache: Accelerating Diffusion Models with Temporal Similarity](https://arxiv.org/abs/2411.14324)
