# Install SGLang-Diffusion

You can install SGLang-Diffusion using one of the methods below.

## Standard Installation (NVIDIA GPUs)

### Method 1: With pip or uv

It is recommended to use uv for a faster installation:

```bash
pip install --upgrade pip
pip install uv
uv pip install "sglang[diffusion]" --prerelease=allow
```

### Method 2: From source

```bash
# Use the latest release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
pip install -e "python[diffusion]"

# With uv
uv pip install -e "python[diffusion]" --prerelease=allow
```

### Method 3: Using Docker

The Docker images are available on Docker Hub at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang), built from the [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile).
Replace `<secret>` below with your HuggingFace Hub [token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
docker run --gpus all \
    --shm-size 32g \
    -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    --ipc=host \
    lmsysorg/sglang:dev \
    zsh -c '\
        echo "Installing diffusion dependencies..." && \
        pip install -e "python[diffusion]" && \
        echo "Starting SGLang-Diffusion..." && \
        sglang generate \
            --model-path black-forest-labs/FLUX.1-dev \
            --prompt "A logo With Bold Large text: SGL Diffusion" \
            --save-output \
    '
```

## Platform-Specific: ROCm (AMD GPUs)

For AMD Instinct GPUs (e.g., MI300X), you can use the ROCm-enabled Docker image:

```bash
docker run --device=/dev/kfd --device=/dev/dri --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env HF_TOKEN=<secret> \
  lmsysorg/sglang:v0.5.5.post2-rocm700-mi30x \
  sglang generate --model-path black-forest-labs/FLUX.1-dev --prompt "A logo With Bold Large text: SGL Diffusion" --save-output
```

For detailed ROCm system configuration and installation from source, see [AMD GPUs](../../platforms/amd_gpu.md).

## Platform-Specific: MUSA (Moore Threads GPUs)

For Moore Threads GPUs (MTGPU) with the MUSA software stack:

```bash
# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

## Platform-Specific: Ascend NPU

For Ascend NPU, please follow the [NPU installation guide](../platforms/ascend_npu.md).

Quick test:

```bash
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```
