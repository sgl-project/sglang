# Install sgl-diffusion

You can install sgl-diffusion using one of the methods below.

This page primarily applies to common NVIDIA GPU platforms.

## Method 1: With pip or uv

It is recommended to use uv for a faster installation:

```bash
pip install --upgrade pip
pip install uv
uv pip install sglang[.diffusion] --prerelease=allow
```

## Method 2: From source

```bash
# Use the latest release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
pip install -e "python/.[diffusion]"

# With uv
uv pip install --prerelease=allow  -e "python/.[diffusion]"
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
    lmsysorg/sglang:diffusion \
    sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```
