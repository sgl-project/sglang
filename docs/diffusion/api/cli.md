# SGLang diffusion CLI Inference

The SGLang-diffusion CLI provides a quick way to access the inference pipeline for image and video generation.

## Prerequisites

- A working SGLang diffusion installation and the `sglang` CLI available in `$PATH`.


## Supported Arguments

### Server Arguments

- `--model-path {MODEL_PATH}`: Path to the model or model ID
- `--lora-path {LORA_PATH}`: Path to a LoRA adapter (local path or HuggingFace model ID). If not specified, LoRA will not be applied.
- `--lora-nickname {NAME}`: Nickname for the LoRA adapter. (default: `default`).
- `--num-gpus {NUM_GPUS}`: Number of GPUs to use
- `--tp-size {TP_SIZE}`: Tensor parallelism size (only for the encoder; should not be larger than 1 if text encoder offload is enabled, as layer-wise offload plus prefetch is faster)
- `--sp-degree {SP_SIZE}`: Sequence parallelism size (typically should match the number of GPUs)
- `--ulysses-degree {ULYSSES_DEGREE}`: The degree of DeepSpeed-Ulysses-style SP in USP
- `--ring-degree {RING_DEGREE}`: The degree of ring attention-style SP in USP
- `--attention-backend {BACKEND}`: Attention backend to use. For SGLang-native pipelines use `fa`, `torch_sdpa`, `sage_attn`, etc. For diffusers pipelines use diffusers backend names like `flash`, `_flash_3_hub`, `sage`, `xformers`.
- `--attention-backend-config {CONFIG}`: Configuration for the attention backend. Can be a JSON string (e.g., '{"k": "v"}'), a path to a JSON/YAML file, or key=value pairs (e.g., "k=v,k2=v2").
- `--cache-dit-config {PATH}`: Path to a Cache-DiT YAML/JSON config (diffusers backend only)
- `--dit-precision {DTYPE}`: Precision for the DiT model (currently supports fp32, fp16, and bf16).


### Sampling Parameters

- `--prompt {PROMPT}`: Text description for the video you want to generate
- `--num-inference-steps {STEPS}`: Number of denoising steps
- `--negative-prompt {PROMPT}`: Negative prompt to guide generation away from certain concepts
- `--seed {SEED}`: Random seed for reproducible generation


**Image/Video Configuration**

- `--height {HEIGHT}`: Height of the generated output
- `--width {WIDTH}`: Width of the generated output
- `--num-frames {NUM_FRAMES}`: Number of frames to generate
- `--fps {FPS}`: Frames per second for the saved output, if this is a video-generation task


**Output Options**

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

### Cloud Storage Support

SGLang diffusion supports automatically uploading generated images and videos to S3-compatible cloud storage (e.g., AWS S3, MinIO, Alibaba Cloud OSS, Tencent Cloud COS).

When enabled, the server follows a **Generate -> Upload -> Delete** workflow:
1. The artifact is generated to a temporary local file.
2. The file is immediately uploaded to the configured S3 bucket in a background thread.
3. Upon successful upload, the local file is deleted.
4. The API response returns the public URL of the uploaded object.

**Configuration**

Cloud storage is enabled via environment variables. Note that `boto3` must be installed separately (`pip install boto3`) to use this feature.

```bash
# Enable S3 storage
export SGLANG_CLOUD_STORAGE_TYPE=s3
export SGLANG_S3_BUCKET_NAME=my-bucket
export SGLANG_S3_ACCESS_KEY_ID=your-access-key
export SGLANG_S3_SECRET_ACCESS_KEY=your-secret-key

# Optional: Custom endpoint for MinIO/OSS/COS
export SGLANG_S3_ENDPOINT_URL=https://minio.example.com
```

See [Environment Variables Documentation](../environment_variables.md) for more details.

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

## Component Path Overrides

SGLang diffusion allows you to override any pipeline component (e.g., `vae`, `transformer`, `text_encoder`) by specifying a custom checkpoint path. This is useful for:

### Example: FLUX.2-dev with Tiny AutoEncoder

You can override **any** component by using `--<component>-path`, where `<component>` matches the key in the model's `model_index.json`:

For example, replace the default VAE with a distilled tiny autoencoder for ~3x faster decoding:

```bash
sglang serve \
  --model-path=black-forest-labs/FLUX.2-dev \
  # with a Huggingface Repo ID
  --vae-path=fal/FLUX.2-Tiny-AutoEncoder
  # or use a local path
  --vae-path=~/.cache/huggingface/hub/models--fal--FLUX.2-Tiny-AutoEncoder/snapshots/.../vae
```

**Important:**
- The component key must match the one in your model's `model_index.json` (e.g., `vae`).
- The path must:
    - either be a Huggingface Repo ID (e.g., fal/FLUX.2-Tiny-AutoEncoder)
    - or point to a **complete component folder**, containing `config.json` and safetensors files


## Diffusers Backend

SGLang diffusion supports a **diffusers backend** that allows you to run any diffusers-compatible model through SGLang's infrastructure using vanilla diffusers pipelines. This is useful for running models without native SGLang implementations or models with custom pipeline classes.

### Arguments

| Argument | Values | Description |
|----------|--------|-------------|
| `--backend` | `auto` (default), `sglang`, `diffusers` | `auto`: prefer native SGLang, fallback to diffusers. `sglang`: force native (fails if unavailable). `diffusers`: force vanilla diffusers pipeline. |
| `--diffusers-attention-backend` | `flash`, `_flash_3_hub`, `sage`, `xformers`, `native` | Attention backend for diffusers pipelines. See [diffusers attention backends](https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends). |
| `--trust-remote-code` | flag | Required for models with custom pipeline classes (e.g., Ovis). |
| `--vae-tiling` | flag | Enable VAE tiling for large image support (decodes tile-by-tile). |
| `--vae-slicing` | flag | Enable VAE slicing for lower memory usage (decodes slice-by-slice). |
| `--dit-precision` | `fp16`, `bf16`, `fp32` | Precision for the diffusion transformer. |
| `--vae-precision` | `fp16`, `bf16`, `fp32` | Precision for the VAE. |

### Example: Running Ovis-Image-7B

[Ovis-Image-7B](https://huggingface.co/AIDC-AI/Ovis-Image-7B) is a 7B text-to-image model optimized for high-quality text rendering.

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

### Extra Diffusers Arguments

For pipeline-specific parameters not exposed via CLI, use `diffusers_kwargs` in a config file:

```json
{
    "model_path": "AIDC-AI/Ovis-Image-7B",
    "backend": "diffusers",
    "prompt": "A beautiful landscape",
    "diffusers_kwargs": {
        "cross_attention_kwargs": {"scale": 0.5}
    }
}
```

```bash
sglang generate --config config.json
```

### Cache-DiT Acceleration

Users who use the diffusers backend can also leverage Cache-DiT acceleration and load custom cache configs from a YAML file to boost performance of diffusers pipelines. See the [Cache-DiT Acceleration](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html) documentation for details.
