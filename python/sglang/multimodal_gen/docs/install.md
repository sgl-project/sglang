# Install sgl-diffusion

You can install sgl-diffusion using one of the methods below.

This page primarily applies to common NVIDIA GPU platforms.

## Method 1: With pip or uv

It is recommended to use uv for a faster installation:

```bash
pip install --upgrade pip
pip install uv
uv pip install sgl-diffusion --prerelease=allow
```

## Method 2: From source

```bash
# Use the latest release branch
git clone -b v0.5.4 https://github.com/sgl-project/sgl-diffusion.git
cd sgl-diffusion

# Install the Python packages
pip install --upgrade pip
pip install -e "."
```

**Quick fixes for common problems:**

- If you want to develop sgl-diffusion, it is recommended to use Docker. The Docker image is `lmsysorg/sgl-diffusion:latest`.

## Method 3: Using Docker

The Docker images are available on Docker Hub at [lmsysorg/sgl-diffusion](), built from the [Dockerfile](https://github.com/sgl-project/sgl-diffusion/tree/main/docker).
Replace `<secret>` below with your HuggingFace Hub [token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sgl-diffusion:latest \
    sglang generate --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
          --use-fsdp-inference \
          --text-encoder-cpu-offload --pin-cpu-memory \
          --prompt "A curious raccoon" \
          --save-output
```
