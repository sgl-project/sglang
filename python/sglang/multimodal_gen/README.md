<div align="center">
<img src=assets/logos/logo.svg width="30%"/>
</div>

**sgl-diffusion is an inference framework for accelerated image/video generation.**

sgl-diffusion features an end-to-end unified pipeline for accelerating diffusion models. It is designed to be modular and extensible, allowing users to easily add new optimizations and techniques.

## Key Features

sgl-diffusion has the following features:

- State-of-the-art performance optimizations for inference
    - [Video Sparse Attention](https://arxiv.org/pdf/2505.13389)
    - [Sliding Tile Attention](https://arxiv.org/pdf/2502.04507)
    - [TeaCache](https://arxiv.org/pdf/2411.19108)
    - [Sage Attention](https://arxiv.org/abs/2410.02367)
    - USP
    - CFG Parallel
- Diverse hardware and OS support
    - Supported hardware: H100, H200, A100, B200, 4090
    - Supported OS: Linux, Windows, MacOS

## Getting Started

```bash
uv pip install sglang[.diffusion] --prerelease=allow
```

For more information, check the [docs](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/install.md).


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

    # Provide a prompt for your video
    prompt = "A curious raccoon peers through a vibrant field of yellow sunflowers, its eyes wide with interest."

    # Generate the video
    video = generator.generate(
        prompt,
        return_frames=True,  # Also return frames from this call (defaults to False)
        output_path="my_videos/",  # Controls where videos are saved
        save_output=True
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

For more information, check the [docs](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen/docs/cli.md).

## Contributing

All contributions are welcome.

## Acknowledgement

We learnt and reused code from the following projects:

- [FastVideo](https://github.com/hao-ai-lab/FastVideo.git). The major components of this repo are based on a fork of FastVide on Sept. 24, 2025.
- [xDiT](https://github.com/xdit-project/xDiT). We used the parallelism library from it.
- [diffusers](https://github.com/huggingface/diffusers) We used the pipeline design from it.
