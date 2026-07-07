<div align="center"  style="display:block; margin:auto;">
<img src=https://github.com/lm-sys/lm-sys.github.io/releases/download/test/sgl-diffusion-logo.png width="80%"/>
</div>

**SGLang diffusion is an inference framework for accelerated image/video generation.**

SGLang diffusion features an end-to-end unified pipeline for accelerating diffusion models. It is designed to be modular and extensible, allowing users to easily add new models and optimizations.

## Key Features

SGLang Diffusion has the following features:
  - Broad model support: Wan series, FastWan series, Hunyuan, LTX-2, Qwen-Image, Qwen-Image-Edit, Flux, Z-Image, GLM-Image
  - Fast inference speed: enpowered by highly optimized kernel from sgl-kernel and efficient scheduler loop
  - Ease of use: OpenAI-compatible api, CLI, and python sdk support
  - Multi-platform support:
    - NVIDIA GPUs (H100, H200, A100, B200, 4090)
    - AMD GPUs (MI300X, MI325X)
    - Ascend NPU (A2, A3)
    - Apple Silicon (M-series via MPS)
    - Moore Threads GPUs (MTT S5000)

### AMD/ROCm Support

SGLang Diffusion supports AMD Instinct GPUs through ROCm. On AMD platforms, we use the Triton attention backend and leverage AITER kernels for optimized layernorm and other operations. See the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation) for setup instructions.

### Moore Threads/MUSA Support

SGLang Diffusion supports Moore Threads GPUs (MTGPU) through the MUSA software stack. On MUSA platforms, we use FlashAttention (FA3) when available; also supports Sage Attention when installed; otherwise falls back to the Torch SDPA backend. See the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation) for setup instructions.

### Apple MPS Support

SGLang Diffusion supports Apple Silicon (M-series) via the MPS backend. Since Triton is Linux-only, all Triton kernels are replaced with PyTorch-native fallbacks on MPS. Norm operations can be optionally accelerated with MLX fused Metal kernels (`SGLANG_USE_MLX=1`). See the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation) for setup instructions.

## Getting Started

```bash
uv pip install 'sglang[diffusion]' --prerelease=allow
```

For more installation methods (e.g. pypi, uv, docker, ROCm/AMD, MUSA/Moore Threads), check the [installation guide](https://docs.sglang.io/docs/sglang-diffusion/installation).

## Inference

Here's a minimal example to generate a video using the default settings:

```python
from sglang.multimodal_gen import DiffGenerator

def main():
    # Create a diff generator from a pre-trained model
    generator = DiffGenerator.from_pretrained(
        model_path="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        num_gpus=1,  # Adjust based on your hardware
    )

    # Generate the video
    video = generator.generate(
        sampling_params_kwargs=dict(
            prompt="A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest.",
            return_frames=True,  # Also return frames from this call (defaults to False)
            output_path="my_videos/",  # Controls where videos are saved
            save_output=True
        )
    )

if __name__ == '__main__':
    main()
```

Or, more simply, with the CLI:

```bash
sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
    --text-encoder-cpu-offload --pin-cpu-memory \
    --prompt "A curious raccoon" \
    --save-output
```

### LoRA support

Apply LoRA adapters via `--lora-path`:

```bash
sglang generate \
  --model-path Qwen/Qwen-Image-Edit-2511 \
  --lora-path prithivMLmods/Qwen-Image-Edit-2511-Anime \
  --prompt "Transform into anime." \
  --image-path "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png" \
  --save-output
```

For more usage examples (e.g. OpenAI compatible API, server mode), check the [CLI reference](https://docs.sglang.io/docs/sglang-diffusion/api/cli).

### Pi0.5 action policies

Pi0.5/OpenPI/LeRobot action policies are registered as native `multimodal_gen` pipelines:

```bash
sglang serve lerobot/pi05_base --model-type diffusion
```

The registered LeRobot checkpoints resolve to `Pi05Pipeline` automatically. The action HTTP API is served under `/v1/actions/generations` with metadata at `/v1/actions/metadata`. Robot clients can use the OpenPI-compatible msgpack websocket adapter at `/openpi/policy`. The native path includes request-local `PrefixContext`, exact full-prefix cache, split prefix/action execution, action denoise CUDA graph support, and the SGLang-owned PaliGemma prefix encoder plus Gemma action expert execution.

See the [Pi0.5 VLA action policy cookbook](../../../docs_new/cookbook/vla/OpenPI/Pi0.5.mdx) for request examples, cache behavior, memory switches, and validation commands.

## Contributing

All contributions are welcome. The contribution guide is available [here](https://docs.sglang.io/docs/sglang-diffusion/contributing).

## Acknowledgement

We learnt and reused code from the following projects:

- [FastVideo](https://github.com/hao-ai-lab/FastVideo.git). The major components of this repo are based on a fork of FastVideo on Sept. 24, 2025.
- [xDiT](https://github.com/xdit-project/xDiT). We used the parallelism library from it.
- [diffusers](https://github.com/huggingface/diffusers) We used the pipeline design from it.
