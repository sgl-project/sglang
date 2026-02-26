# Cache-DiT Acceleration

SGLang integrates [Cache-DiT](https://github.com/vipshop/cache-dit), a caching acceleration engine for Diffusion Transformers (DiT), to achieve up to **1.69x inference speedup** with minimal quality loss.

## Overview

**Cache-DiT** uses intelligent caching strategies to skip redundant computation in the denoising loop:

- **DBCache (Dual Block Cache)**: Dynamically decides when to cache transformer blocks based on residual differences
- **TaylorSeer**: Uses Taylor expansion for calibration to optimize caching decisions
- **SCM (Step Computation Masking)**: Step-level caching control for additional speedup

## Basic Usage

Enable Cache-DiT by exporting the environment variable and using `sglang generate` or `sglang serve` :

```bash
SGLANG_CACHE_DIT_ENABLED=true \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A beautiful sunset over the mountains"
```

## Diffusers Backend

Cache-DiT supports loading acceleration configs from a custom YAML file. For
diffusers pipelines (`diffusers` backend), pass the YAML/JSON path via `--cache-dit-config`. This
flow requires cache-dit >= 1.2.0 (`cache_dit.load_configs`).

### Single GPU inference

Define a `cache.yaml` file that contains:

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
```

Then apply the config with:

```bash
sglang generate \
  --backend diffusers \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config cache.yaml \
  --prompt "A beautiful sunset over the mountains"
```

### Distributed inference

- 1D Parallelism

Define a parallelism only config yaml `parallel.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: auto
  parallel_kwargs:
    attention_backend: native
    extra_parallel_modules: ["text_encoder", "vae"]
```

Then, apply the distributed inference acceleration config from yaml. `ulysses_size: auto` means that cache-dit will auto detect the `world_size` as the ulysses_size. Otherwise, you should manually set it as specific int number, e.g, 4.

Then apply the distributed config with: (Note: please add `--num-gpus N` to specify the number of gpus for distributed inference)

```bash
sglang generate \
  --backend diffusers \
  --num-gpus 4 \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config parallel.yaml \
  --prompt "A futuristic cityscape at sunset"
```

- 2D Parallelism

You can also define a 2D parallelism config yaml `parallel_2d.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: auto
  tp_size: 2
  parallel_kwargs:
    attention_backend: native
    extra_parallel_modules: ["text_encoder", "vae"]
```
Then, apply the 2D parallelism config from yaml. Here `tp_size: 2` means using tensor parallelism with size 2. The `ulysses_size: auto` means that cache-dit will auto detect the `world_size // tp_size` as the ulysses_size.

- 3D Parallelism

You can also define a 3D parallelism config yaml `parallel_3d.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: 2
  ring_size: 2
  tp_size: 2
  parallel_kwargs:
    attention_backend: native
    extra_parallel_modules: ["text_encoder", "vae"]
```
Then, apply the 3D parallelism config from yaml. Here `ulysses_size: 2`, `ring_size: 2`, `tp_size: 2` means using ulysses parallelism with size 2, ring parallelism with size 2 and tensor parallelism with size 2.

### Hybrid Cache and Parallelism

Define a hybrid cache and parallel acceleration config yaml `hybrid.yaml` file that contains:

```yaml
cache_config:
  max_warmup_steps: 8
  warmup_interval: 2
  max_cached_steps: -1
  max_continuous_cached_steps: 2
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
parallelism_config:
  ulysses_size: auto
  parallel_kwargs:
    attention_backend: native
    extra_parallel_modules: ["text_encoder", "vae"]
```

Then, apply the hybrid cache and parallel acceleration config from yaml.

```bash
sglang generate \
  --backend diffusers \
  --num-gpus 4 \
  --model-path Qwen/Qwen-Image \
  --cache-dit-config hybrid.yaml \
  --prompt "A beautiful sunset over the mountains"
```

## Advanced Configuration

### DBCache Parameters

DBCache controls block-level caching behavior:

| Parameter | Env Variable              | Default | Description                              |
|-----------|---------------------------|---------|------------------------------------------|
| Fn        | `SGLANG_CACHE_DIT_FN`     | 1       | Number of first blocks to always compute |
| Bn        | `SGLANG_CACHE_DIT_BN`     | 0       | Number of last blocks to always compute  |
| W         | `SGLANG_CACHE_DIT_WARMUP` | 4       | Warmup steps before caching starts       |
| R         | `SGLANG_CACHE_DIT_RDT`    | 0.24    | Residual difference threshold            |
| MC        | `SGLANG_CACHE_DIT_MC`     | 3       | Maximum continuous cached steps          |

### TaylorSeer Configuration

TaylorSeer improves caching accuracy using Taylor expansion:

| Parameter | Env Variable                  | Default | Description                     |
|-----------|-------------------------------|---------|---------------------------------|
| Enable    | `SGLANG_CACHE_DIT_TAYLORSEER` | false   | Enable TaylorSeer calibrator    |
| Order     | `SGLANG_CACHE_DIT_TS_ORDER`   | 1       | Taylor expansion order (1 or 2) |

### Combined Configuration Example

DBCache and TaylorSeer are complementary strategies that work together, you can configure both sets of parameters
simultaneously:

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_FN=2 \
SGLANG_CACHE_DIT_BN=1 \
SGLANG_CACHE_DIT_WARMUP=4 \
SGLANG_CACHE_DIT_RDT=0.4 \
SGLANG_CACHE_DIT_MC=4 \
SGLANG_CACHE_DIT_TAYLORSEER=true \
SGLANG_CACHE_DIT_TS_ORDER=2 \
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A curious raccoon in a forest"
```

### SCM (Step Computation Masking)

SCM provides step-level caching control for additional speedup. It decides which denoising steps to compute fully and
which to use cached results.

**SCM Presets**

SCM is configured with presets:

| Preset   | Compute Ratio | Speed    | Quality    |
|----------|---------------|----------|------------|
| `none`   | 100%          | Baseline | Best       |
| `slow`   | ~75%          | ~1.3x    | High       |
| `medium` | ~50%          | ~2x      | Good       |
| `fast`   | ~35%          | ~3x      | Acceptable |
| `ultra`  | ~25%          | ~4x      | Lower      |

**Usage**

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_PRESET=medium \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A futuristic cityscape at sunset"
```

**Custom SCM Bins**

For fine-grained control over which steps to compute vs cache:

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_COMPUTE_BINS="8,3,3,2,2" \
SGLANG_CACHE_DIT_SCM_CACHE_BINS="1,2,2,2,3" \
sglang generate --model-path Qwen/Qwen-Image \
    --prompt "A futuristic cityscape at sunset"
```

**SCM Policy**

| Policy    | Env Variable                          | Description                                 |
|-----------|---------------------------------------|---------------------------------------------|
| `dynamic` | `SGLANG_CACHE_DIT_SCM_POLICY=dynamic` | Adaptive caching based on content (default) |
| `static`  | `SGLANG_CACHE_DIT_SCM_POLICY=static`  | Fixed caching pattern                       |

## Environment Variables

All Cache-DiT parameters can be configured via environment variables.
See [Environment Variables](../../environment_variables.md) for the complete list.

## Supported Models

SGLang Diffusion x Cache-DiT supports almost all models originally supported in SGLang Diffusion:

| Model Family | Example Models              |
|--------------|-----------------------------|
| Wan          | Wan2.1, Wan2.2              |
| Flux         | FLUX.1-dev, FLUX.2-dev      |
| Z-Image      | Z-Image-Turbo               |
| Qwen         | Qwen-Image, Qwen-Image-Edit |
| Hunyuan      | HunyuanVideo                |

## Performance Tips

1. **Start with defaults**: The default parameters work well for most models
2. **Use TaylorSeer**: It typically improves both speed and quality
3. **Tune R threshold**: Lower values = better quality, higher values = faster
4. **SCM for extra speed**: Use `medium` preset for good speed/quality balance
5. **Warmup matters**: Higher warmup = more stable caching decisions

## Limitations

- **SGLang-native pipelines**: Distributed support (TP/SP) is not yet validated; Cache-DiT will be automatically
  disabled when `world_size > 1`.
- **SCM minimum steps**: SCM requires >= 8 inference steps to be effective
- **Model support**: Only models registered in Cache-DiT's BlockAdapterRegister are supported

## Troubleshooting

### SCM disabled for low step count

For models with < 8 inference steps (e.g., DMD distilled models), SCM will be automatically disabled. DBCache
acceleration still works.

## References

- [Cache-DiT](https://github.com/vipshop/cache-dit)
- [SGLang Diffusion](../index.md)
