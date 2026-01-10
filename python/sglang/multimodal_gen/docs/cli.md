# SGLang diffusion CLI Inference

The SGLang-diffusion CLI provides a quick way to access the inference pipeline for image and video generation.

## Prerequisites

- A working SGLang diffusion installation and the `sglang` CLI available in `$PATH`.
- Python 3.11+ if you plan to use the OpenAI Python SDK.


## Supported Arguments

### Server Arguments

- `--model-path {MODEL_PATH}`: Path to the model or model ID
- `--vae-path {VAE_PATH}`: Path to a custom VAE model or HuggingFace model ID (e.g., `fal/FLUX.2-Tiny-AutoEncoder`). If not specified, the VAE will be loaded from the main model path.
- `--lora-path {LORA_PATH}`: Path to a LoRA adapter (local path or HuggingFace model ID). If not specified, LoRA will not be applied.
- `--lora-nickname {NAME}`: Nickname for the LoRA adapter. (default: `default`).
- `--num-gpus {NUM_GPUS}`: Number of GPUs to use
- `--tp-size {TP_SIZE}`: Tensor parallelism size (only for the encoder; should not be larger than 1 if text encoder offload is enabled, as layer-wise offload plus prefetch is faster)
- `--sp-degree {SP_SIZE}`: Sequence parallelism size (typically should match the number of GPUs)
- `--ulysses-degree {ULYSSES_DEGREE}`: The degree of DeepSpeed-Ulysses-style SP in USP
- `--ring-degree {RING_DEGREE}`: The degree of ring attention-style SP in USP


### Sampling Parameters

- `--prompt {PROMPT}`: Text description for the video you want to generate
- `--num-inference-steps {STEPS}`: Number of denoising steps
- `--negative-prompt {PROMPT}`: Negative prompt to guide generation away from certain concepts
- `--seed {SEED}`: Random seed for reproducible generation


#### Image/Video Configuration

- `--height {HEIGHT}`: Height of the generated output
- `--width {WIDTH}`: Width of the generated output
- `--num-frames {NUM_FRAMES}`: Number of frames to generate
- `--fps {FPS}`: Frames per second for the saved output, if this is a video-generation task


#### Output Options

- `--output-path {PATH}`: Directory to save the generated video
- `--save-output`: Whether to save the image/video to disk
- `--return-frames`: Whether to return the raw frames

### Using Configuration Files

Instead of specifying all parameters on the command line, you can use a configuration file:

```bash
sglang generate --config {CONFIG_FILE_PATH}
```

The configuration file should be in JSON or YAML format with the same parameter names as the CLI options. Command-line arguments take precedence over settings in the configuration file, allowing you to override specific values while keeping the rest from the configuration file.

Example configuration file (config.json):

```json
{
    "model_path": "FastVideo/FastHunyuan-diffusers",
    "prompt": "A beautiful woman in a red dress walking down a street",
    "output_path": "outputs/",
    "num_gpus": 2,
    "sp_size": 2,
    "tp_size": 1,
    "num_frames": 45,
    "height": 720,
    "width": 1280,
    "num_inference_steps": 6,
    "seed": 1024,
    "fps": 24,
    "precision": "bf16",
    "vae_precision": "fp16",
    "vae_tiling": true,
    "vae_sp": true,
    "vae_config": {
        "load_encoder": false,
        "load_decoder": true,
        "tile_sample_min_height": 256,
        "tile_sample_min_width": 256
    },
    "text_encoder_precisions": [
        "fp16",
        "fp16"
    ],
    "mask_strategy_file_path": null,
    "enable_torch_compile": false
}
```

Or using YAML format (config.yaml):

```yaml
model_path: "FastVideo/FastHunyuan-diffusers"
prompt: "A beautiful woman in a red dress walking down a street"
output_path: "outputs/"
num_gpus: 2
sp_size: 2
tp_size: 1
num_frames: 45
height: 720
width: 1280
num_inference_steps: 6
seed: 1024
fps: 24
precision: "bf16"
vae_precision: "fp16"
vae_tiling: true
vae_sp: true
vae_config:
  load_encoder: false
  load_decoder: true
  tile_sample_min_height: 256
  tile_sample_min_width: 256
text_encoder_precisions:
  - "fp16"
  - "fp16"
mask_strategy_file_path: null
enable_torch_compile: false
```


To see all the options, you can use the `--help` flag:

```bash
sglang generate --help
```

## Serve

Launch the SGLang diffusion HTTP server and interact with it using the OpenAI SDK and curl.

### Start the server

Use the following command to launch the server:

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
)

sglang serve "${SERVER_ARGS[@]}"
```

- **--model-path**: Which model to load. The example uses `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`.
- **--port**: HTTP port to listen on (the default here is `30010`).

For detailed API usage, including Image, Video Generation and LoRA management, please refer to the [OpenAI API Documentation](openai_api.md).


## Generate

Run a one-off generation task without launching a persistent server.

To use it, pass both server arguments and sampling parameters in one command, after the `generate` subcommand, for example:

```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --text-encoder-cpu-offload
  --pin-cpu-memory
  --num-gpus 4
  --ulysses-degree=2
  --ring-degree=2
)

SAMPLING_ARGS=(
  --prompt "A curious raccoon"
  --save-output
  --output-path outputs
  --output-file-name "A curious raccoon.mp4"
)

sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"

# Or, users can set `SGLANG_CACHE_DIT_ENABLED` env as `true` to enable cache acceleration
SGLANG_CACHE_DIT_ENABLED=true sglang generate "${SERVER_ARGS[@]}" "${SAMPLING_ARGS[@]}"
```

Once the generation task has finished, the server will shut down automatically.

> [!NOTE]
> The HTTP server-related arguments are ignored in this subcommand.

## Diffusers Backend

SGLang diffusion supports a **diffusers backend** that allows you to run any diffusers-compatible model through SGLang's infrastructure using vanilla diffusers pipelines. This is useful for:

- Running models that don't have native SGLang implementations yet
- Using models with custom pipeline classes defined in their repositories
- Quick prototyping with SGLang server support

### Backend Selection

Use the `--backend` flag to control which backend is used:

| Value | Behavior |
|-------|----------|
| `auto` (default) | Prefers native SGLang implementation; falls back to diffusers if not available |
| `sglang` | Forces native SGLang implementation (fails if not available) |
| `diffusers` | Forces vanilla diffusers pipeline |

### Basic Usage

```bash
# Explicitly use diffusers backend
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --prompt "A beautiful sunset over mountains" \
  --save-output \
  --output-path outputs/sunset.png

# Auto mode - uses native SGLang if available, otherwise diffusers
sglang generate \
  --model-path some-new-model/not-yet-supported \
  --prompt "A futuristic city" \
  --save-output
```

### Diffusers-Specific Parameters

#### `--backend`

Selects the model backend:
- `auto`: Automatically select (prefer native SGLang, fallback to diffusers)
- `sglang`: Use native optimized SGLang implementation
- `diffusers`: Use vanilla diffusers pipeline

#### `--diffusers-attention-backend`

Sets the attention backend specifically for diffusers pipelines. This is separate from `--attention-backend` which applies to native SGLang implementations.

Available options (see [HuggingFace diffusers attention backends](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends)):
- `flash`: FlashAttention
- `_flash_3_hub`: FlashAttention 3 from HuggingFace Hub
- `sage`: SageAttention
- `xformers`: xFormers memory-efficient attention
- `native`: PyTorch native attention

```bash
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --diffusers-attention-backend flash \
  --prompt "A cat wearing a hat" \
  --save-output
```

#### `--trust-remote-code`

Required for models with custom pipeline classes (e.g., Ovis, custom community models):

```bash
sglang generate \
  --model-path THUDM/Ovis2-4B \
  --backend diffusers \
  --trust-remote-code \
  --prompt "A photorealistic portrait" \
  --save-output
```

### Pipeline Configuration Options

The diffusers backend supports several pipeline-level configurations:

#### VAE Optimizations

- `--vae-tiling`: Enable VAE tiling for large image support (decodes tile-by-tile)
- `--vae-slicing`: Enable VAE slicing for lower memory usage (decodes slice-by-slice)

```bash
sglang generate \
  --model-path stabilityai/stable-diffusion-xl-base-1.0 \
  --backend diffusers \
  --vae-tiling \
  --height 2048 \
  --width 2048 \
  --prompt "A detailed landscape"
```

#### Precision Control

- `--dit-precision`: Precision for the diffusion transformer (`fp16`, `bf16`, `fp32`)
- `--vae-precision`: Precision for the VAE (`fp16`, `bf16`, `fp32`)

### Example: Running Ovis with Diffusers Backend

[Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B) Ovis-Image is a 7B text-to-image model specifically optimized for high-quality text rendering, designed to operate efficiently under stringent computational constraints.

```bash
# Basic Ovis generation
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --trust-remote-code \
  --prompt "A serene Japanese garden with cherry blossoms" \
  --height 1024 \
  --width 1024 \
  --num-inference-steps 30 \
  --guidance-scale 7.5 \
  --save-output \
  --output-path outputs/ovis_garden.png \
  --output-file-name ovis_garden.png
```

With attention backend optimization:

```bash
sglang generate \
  --model-path AIDC-AI/Ovis-Image-7B \
  --backend diffusers \
  --trust-remote-code \
  --diffusers-attention-backend flash \
  --prompt "A cyberpunk city at night with neon lights" \
  --height 1024 \
  --width 1024 \
  --save-output
```

### Passing Extra Diffusers Arguments

For pipeline-specific parameters not exposed via CLI, you can use a configuration file with `diffusers_kwargs`:

```json
{
    "model_path": "stabilityai/stable-diffusion-xl-base-1.0",
    "backend": "diffusers",
    "prompt": "A beautiful landscape",
    "height": 1024,
    "width": 1024,
    "diffusers_kwargs": {
        "output_type": "latent",
        "return_dict": false,
        "cross_attention_kwargs": {"scale": 0.5}
    }
}
```

```bash
sglang generate --config config.json
```

### How It Works

When using the diffusers backend:

1. **Model Loading**: The model is loaded using `DiffusionPipeline.from_pretrained()` with automatic dtype selection (bf16 if supported, otherwise fp16)
2. **Device Mapping**: Models are loaded directly to GPU using `device_map="cuda"` for faster initialization
3. **Custom Pipelines**: If the model has a custom pipeline class not in diffusers, it automatically tries loading with `custom_pipeline` from the repository
4. **Output Handling**: Pipeline outputs are automatically normalized and converted to the expected format for saving