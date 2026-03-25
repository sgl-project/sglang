# Quantization

SGLang-Diffusion supports quantized transformer checkpoints. Prefer keeping the
base model and quantized transformer override separate.

## CLI Usage

Use these paths:

- `--model-path`: the base or original model
- `--transformer-path`: a quantized transformers-style transformer component directory that already contains its own
  `config.json`
- `--transformer-weights-path`: quantized transformer weights provided as a single safetensors file, a sharded
  safetensors directory, a local path, or a Hugging Face repo ID

Recommended:

```bash
sglang generate \
  --model-path black-forest-labs/FLUX.2-dev \
  --transformer-weights-path black-forest-labs/FLUX.2-dev-NVFP4 \
  --prompt "a curious pikachu"
```

For quantized transformers-style transformer component folders:

```bash
sglang generate \
  --model-path /path/to/base-model \
  --transformer-path /path/to/quantized-transformer \
  --prompt "A Logo With Bold Large Text: SGL Diffusion"
```

Some model-specific integrations also accept a quantized repo or local directory directly as `--model-path`, but this is
a compatibility path.
If a repo contains multiple candidate checkpoints, pass `--transformer-weights-path` explicitly.

## Quant Families

Here, `quant_family` means a checkpoint and loading family with shared CLI usage and loader behavior.
It is not just the numeric precision or a kernel backend.

| quant_family     | checkpoint form                                                                            | canonical CLI                                        | supported models                                             | extra dependency                      | platform / notes                                                                                                      |
|------------------|--------------------------------------------------------------------------------------------|------------------------------------------------------|--------------------------------------------------------------|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `fp8`            | Quantized transformer component folder, or safetensors with `quantization_config` metadata | `--transformer-path` or `--transformer-weights-path` | ALL                                                          | None                                  | Component-folder and single-file flows are both supported                                                             |
| `nvfp4-modelopt` | NVFP4 safetensors file, sharded directory, or repo providing transformer weights           | `--transformer-weights-path`                         | FLUX.2                                                       | `comfy-kitchen` optional on Blackwell | Blackwell can use a best-performance kit when available; otherwise SGLang falls back to the generic ModelOpt FP4 path |
| `nunchaku-svdq`  | Pre-quantized Nunchaku transformer weights, usually named `svdq-{int4\| fp4}_r{rank}-...`  | `--transformer-weights-path`                         | Model-specific support such as Qwen-Image, FLUX, and Z-Image | `nunchaku`                            | Rank and precision can be inferred from the filename; supports both `int4` and `nvfp4` variants                       |
