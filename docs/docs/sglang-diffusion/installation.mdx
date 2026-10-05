---
title: Install SGLang Diffusion
description: Run SGLang's built-in diffusion engine with Docker, or install the diffusion extra with pip, uv, or from source.
---
SGLang Diffusion is SGLang's built-in image and video generation engine. It ships in the same [repository](https://github.com/sgl-project/sglang) and **`sglang` Python package**, using `sglang generate` and `sglang serve`.

Docker is the recommended setup for Linux GPU deployments. The official NVIDIA images include SGLang, its diffusion dependencies, and the optimized kernel stack; no separate Python installation or `pip install` inside the container is needed for the standard diffusion runtime. Optional backends and model-specific dependencies are documented in the [cookbook](/cookbook/diffusion/intro).

For Python installation, use `sglang[diffusion]`: `[diffusion]` adds optional dependencies to the same package, not a separate engine.

## Install on NVIDIA GPUs

<Tabs>
<Tab title="Docker (recommended)" id="method-3-using-docker">

Use Linux with a supported NVIDIA GPU, a driver compatible with the image's CUDA version, [Docker Engine](https://docs.docker.com/engine/install/), and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Configure Docker's GPU runtime before continuing. See the [SGLang installation guide](/docs/get-started/install) for current CUDA requirements.

The example uses the latest stable release. For reproducible deployments, replace `latest` with a release tag or image digest from [Docker Hub](https://hub.docker.com/r/lmsysorg/sglang/tags). For models or features not yet released, choose a [nightly image](https://hub.docker.com/r/lmsysorg/sglang/tags?name=nightly) containing the required changes, or install from source.

```bash
docker pull lmsysorg/sglang:latest
```

Start a container with persistent model and output directories. The working directory is `/outputs`, so generated files remain in the host's `outputs` directory after the container exits. The server port is published only on the host's loopback interface.

```bash
mkdir -p "$HOME/.cache/huggingface" "$PWD/outputs"
docker run --rm -it --gpus all \
  --ipc=host \
  -p 127.0.0.1:30000:30000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$PWD/outputs:/outputs" \
  -w /outputs \
  lmsysorg/sglang:latest bash
```

For gated or private models, export your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) as `HF_TOKEN` on the host and add `--env HF_TOKEN` before the image name. Mount any local input media or checkpoints into the container and use their container paths in requests.

Run the commands in [Generate or serve](#generate-or-serve) inside this container. For development tools and source changes, see the [development environment guide](/docs/developer_guide/development_guide_using_docker#setup-docker-container).

</Tab>
<Tab title="pip / uv" id="method-1-with-pip-or-uv">

Use this option for an existing Python environment. Follow the [Python and CUDA prerequisites](/docs/get-started/install), then install the diffusion extra in a virtual environment:

```bash
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install "sglang[diffusion]" --prerelease=allow
```

With pip, use `pip install "sglang[diffusion]" --pre` in your activated environment.

</Tab>
<Tab title="From source" id="method-2-from-source">

Use source installation to develop SGLang or try changes that are not in a published image. Follow the [Python and CUDA prerequisites](/docs/get-started/install). The commands below install the current `main`; check out a release tag or commit before installation when you need a fixed revision.

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install uv
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e "python[diffusion]" --prerelease=allow
```

The diffusion implementation lives in [`python/sglang/multimodal_gen`](https://github.com/sgl-project/sglang/tree/main/python/sglang/multimodal_gen).

</Tab>
</Tabs>

## Generate or serve

Run these commands inside the container or your activated Python environment. For a one-off image:

```bash
sglang generate --model-path Qwen/Qwen-Image \
  --prompt "A beautiful sunset over the mountains" \
  --save-output
```

Alternatively, start an HTTP server. Binding to `0.0.0.0` makes the published container port reachable from the host:

```bash
sglang serve --model-path Qwen/Qwen-Image --host 0.0.0.0 --port 30000
```

Choose a model and GPU configuration from the [cookbook](/cookbook/diffusion/intro). See the [OpenAI-compatible API](/docs/sglang-diffusion/api/openai_api) to send requests.

## AMD GPUs (ROCm)

Docker is also recommended for AMD GPUs. Use a ROCm image matched to your GPU and host driver, not the NVIDIA `latest` image. Follow [AMD GPUs](/docs/hardware-platforms/amd_gpu) for the current image tags, device passthrough flags, and source installation options.

## Moore Threads GPUs (MUSA)

For Moore Threads GPUs (MTGPU) with the MUSA software stack, follow the platform guide first. If the source tree still requires the alternate platform `pyproject` fallback, keep a backup of the default file before switching:

```bash Command
# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Install the Python packages
pip install --upgrade pip
mv python/pyproject.toml python/pyproject.toml.bak
cp python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_musa]"
```

## Intel XPU

For Intel Data Center GPU Max or Arc GPUs, follow the Docker instructions in the [XPU installation guide](/docs/hardware-platforms/xpu). The Dockerfile already includes diffusion dependencies.

## Ascend NPU

For Ascend NPU, please follow the [NPU installation guide](/docs/hardware-platforms/ascend-npus/getting-started/installation).

Quick test:

```bash Command
sglang generate --model-path black-forest-labs/FLUX.1-dev \
    --prompt "A logo With Bold Large text: SGL Diffusion" \
    --save-output
```

## Platform-Specific: Apple MPS

For Apple MPS, install natively on macOS using the source instructions below; the Linux Docker image is not an MPS runtime. If the source tree still requires the alternate platform `pyproject` fallback, keep a backup of the default file before switching:

```bash Command
# Install ffmpeg
brew install ffmpeg

# Install uv
brew install uv

# Clone the repository
git clone https://github.com/sgl-project/sglang.git
cd sglang

# Create and activate a virtual environment
uv venv -p 3.12 sglang-diffusion
source sglang-diffusion/bin/activate

# Install the Python packages
uv pip install --upgrade pip
mv python/pyproject.toml python/pyproject.toml.bak
cp python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
```

SGLang Diffusion uses PyTorch MPS. The `all_mps` extra also installs the SRT
MLX backend dependencies; `SGLANG_USE_MLX` applies only to SRT serving.
