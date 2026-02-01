<div align="center"  style="display:block; margin:auto;">
<img src=https://github.com/lm-sys/lm-sys.github.io/releases/download/test/sgl-diffusion-logo.png width="80%"/>
</div>

**SGLang diffusion is an inference framework for accelerated image/video generation.**

SGLang diffusion features an end-to-end unified pipeline for accelerating diffusion models. It is designed to be modular and extensible, allowing users to easily add new models and optimizations.

## Key Features

SGLang Diffusion has the following features:
  - Broad model support: Wan series, FastWan series, Hunyuan, Qwen-Image, Qwen-Image-Edit, Flux, Z-Image, GLM-Image
  - Fast inference speed: enpowered by highly optimized kernel from sgl-kernel and efficient scheduler loop
  - Ease of use: OpenAI-compatible api, CLI, and python sdk support
  - Multi-platform support: NVIDIA GPUs (H100, H200, A100, B200, 4090) and AMD GPUs (MI300X, MI325X)

### AMD/ROCm Support

SGLang Diffusion supports AMD Instinct GPUs through ROCm. On AMD platforms, we use the Triton attention backend and leverage AITER kernels for optimized layernorm and other operations. See the [ROCm installation guide](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/install_rocm.md) for setup instructions.

## Getting Started

```bash
uv pip install 'sglang[diffusion]' --prerelease=allow
```

For more installation methods (e.g. pypi, uv, docker), check [install.md](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/install.md). ROCm/AMD users should follow the [ROCm quickstart](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/install_rocm.md) that includes the additional kernel builds and attention backend settings we validated on MI300X.


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

For more usage examples (e.g. OpenAI compatible API, server mode), check [cli.md](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/cli.md).

## Contributing

All contributions are welcome. The contribution guide is available [here](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/contributing.md).

## Acknowledgement

We learnt and reused code from the following projects:

- [FastVideo](https://github.com/hao-ai-lab/FastVideo.git). The major components of this repo are based on a fork of FastVideo on Sept. 24, 2025.
- [xDiT](https://github.com/xdit-project/xDiT). We used the parallelism library from it.
- [diffusers](https://github.com/huggingface/diffusers) We used the pipeline design from it.
