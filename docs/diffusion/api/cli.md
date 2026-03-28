# SGLang Diffusion CLI

Use the CLI for one-off generation with `sglang generate` or to start a persistent HTTP server with `sglang serve`.

### Overlay repos for non-diffusers models

If `--model-path` points to a supported non-diffusers source repo, SGLang can resolve it
through a self-hosted overlay repo.

SGLang first checks a built-in overlay registry. Concrete built-in mappings can be added over time without changing the CLI surface.

Override example:

```bash
export SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY='{
  "Wan-AI/Wan2.2-S2V-14B": {
    "overlay_repo_id": "your-org/Wan2.2-S2V-14B-overlay",
    "overlay_revision": "main"
  }
}'

sglang generate \
  --model-path Wan-AI/Wan2.2-S2V-14B \
  --config configs/wan_s2v.yaml
```

The overlay repo should be a complete diffusers-style/componentized repo

You can also pass the overlay repo itself as `--model-path` if it contains `_overlay/overlay_manifest.json`.

Notes:
1. `SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY` is only an optional override for
development and debugging. It accepts either a JSON object or a path to a JSON
file, and can extend or replace built-in entries for the current process.


## Quick Start

### Generate

```bash
sglang generate \
  --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output
```

### Serve

```bash
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --port 30010
```

For request and response examples, see [OpenAI-Compatible API](openai_api.md).

```{tip}
Use `sglang generate --help` and `sglang serve --help` for the full argument list. The CLI help output is the source of truth for exhaustive flags.
```

## Common Options

### Model and runtime

- `--model-path {MODEL}`: model path or Hugging Face model ID
- `--lora-path {PATH}` and `--lora-nickname {NAME}`: load a LoRA adapter
- `--num-gpus {N}`: number of GPUs to use
- `--tp-size {N}`: tensor parallelism size, mainly for encoders
- `--sp-degree {N}`: sequence parallelism size
- `--ulysses-degree {N}` and `--ring-degree {N}`: USP parallelism controls
- `--attention-backend {BACKEND}`: attention backend for native SGLang pipelines
- `--attention-backend-config {CONFIG}`: attention backend configuration

### Sampling and output

- `--prompt {PROMPT}` and `--negative-prompt {PROMPT}`
- `--num-inference-steps {STEPS}` and `--seed {SEED}`
- `--height {HEIGHT}`, `--width {WIDTH}`, `--num-frames {N}`, `--fps {FPS}`
- `--output-path {PATH}`, `--output-file-name {NAME}`, `--save-output`, `--return-frames`

For frame interpolation and upscaling, see [Post-Processing](post_processing.md).

### Quantized transformers

For quantized transformer checkpoints, prefer:

- `--model-path` for the base pipeline
- `--transformer-path` for a quantized `transformers` transformer component folder
- `--transformer-weights-path` for a quantized safetensors file, directory, or repo

See [Quantization](../quantization.md) for supported quantization families and examples.

## Configuration Files

Use `--config` to load JSON or YAML configuration. Command-line flags override values from the config file.

```bash
sglang generate --config config.yaml
```

Example:

```yaml
model_path: FastVideo/FastHunyuan-diffusers
prompt: A beautiful woman in a red dress walking down a street
output_path: outputs/
num_gpus: 2
sp_size: 2
tp_size: 1
num_frames: 45
height: 720
width: 1280
num_inference_steps: 6
seed: 1024
fps: 24
precision: bf16
vae_precision: fp16
vae_tiling: true
vae_sp: true
enable_torch_compile: false
```

## Generate

`sglang generate` runs a single generation job and exits when the job finishes.

```bash
sglang generate \
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --prompt "A curious raccoon" \
  --save-output \
  --output-path outputs \
  --output-file-name "a-curious-raccoon.mp4"
```

```{note}
HTTP server-only arguments are ignored by `sglang generate`.
```

For diffusers pipelines, Cache-DiT can be enabled with `SGLANG_CACHE_DIT_ENABLED=true` or `--cache-dit-config`. See [Cache-DiT](../performance/cache/cache_dit.md).

## Serve

`sglang serve` starts the HTTP server and keeps the model loaded for repeated requests.

```bash
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 4 \
  --ulysses-degree 2 \
  --ring-degree 2 \
  --port 30010
```

### Cloud Storage

SGLang Diffusion can upload generated images and videos to S3-compatible object storage after generation.

```bash
export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=my-bucket
export SGLANG_S3_ACCESS_KEY_ID=your-access-key
export SGLANG_S3_SECRET_ACCESS_KEY=your-secret-key
export SGLANG_S3_ENDPOINT_URL=https://minio.example.com
```

See [Environment Variables](../environment_variables.md) for the full set of storage options.

## Component Path Overrides

Override individual pipeline components such as `vae`, `transformer`, or `text_encoder` with `--<component>-path`.

```bash
sglang serve \
  --model-path black-forest-labs/FLUX.2-dev \
  --vae-path fal/FLUX.2-Tiny-AutoEncoder
```

The component key must match the key in the model's `model_index.json`, and the path must be either a Hugging Face repo ID or a complete component directory.

## Diffusers Backend

Use `--backend diffusers` to force vanilla diffusers pipelines when no native SGLang implementation exists or when a model requires a custom pipeline class.

### Key Options

| Argument | Values | Description |
|----------|--------|-------------|
| `--backend` | `auto`, `sglang`, `diffusers` | Choose native SGLang, force native, or force diffusers |
| `--diffusers-attention-backend` | `flash`, `_flash_3_hub`, `sage`, `xformers`, `native` | Attention backend for diffusers pipelines |
| `--trust-remote-code` | flag | Required for models with custom pipeline classes |
| `--vae-tiling` and `--vae-slicing` | flag | Lower memory usage for VAE decode |
| `--dit-precision` and `--vae-precision` | `fp16`, `bf16`, `fp32` | Precision controls |
| `--enable-torch-compile` | flag | Enable `torch.compile` |
| `--cache-dit-config` | `{PATH}` | Cache-DiT config for diffusers pipelines |

### Example

```bash
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --trust-remote-code \
  --diffusers-attention-backend flash \
  --prompt "A serene Japanese garden with cherry blossoms" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --save-output \
  --output-path outputs \
  --output-file-name ovis_garden.png
```

For pipeline-specific arguments not exposed in the CLI, pass `diffusers_kwargs` in a config file.
