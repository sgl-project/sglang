# AMD Configuration and Setup for SGLang

## Table of Contents

- [Introduction](#introduction)
- [System Configure](#system-configure)
  - [MI300X](#mi300x)
- [Installing SGLang](#installing-sglang)
  - [Install from Source](#install-to-source)
  - [Install Using Docker](#install-using-docker)

## Introduction

This document describes how to set up an AMD-based environment for [SGLang](https://docs.sglang.ai). If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues) on the SGLang repository.

## System Configuration

When using AMD GPUs (such as MI300X), certain system-level optimizations help ensure stable performance.

### MI300X Notes

AMD provides official documentation for MI300X optimization and system tuning:

- [AMD MI300X Tuning Guides](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
  - [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html)
  - [AMD Instinct MI300X System Optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html)
  - [AMD Instinct MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)

> **Tip:** We strongly recommend reading the entire `System Optimization` guide to fully configure your system.

Below are a few key settings to confirm or enable:

1. **Update GRUB Settings**
   In `/etc/default/grub`, append the following to `GRUB_CMDLINE_LINUX`:
   ```text
   pci=realloc=off iommu=pt
   ```
   Afterward, run `sudo update-grub` (or your distro’s equivalent) and reboot.
2. **Disable NUMA Auto-Balancing**
   ```bash
   sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
   ```
   - You can automate or verify this change using [this helpful script](https://github.com/ROCm/triton/blob/rocm_env/scripts/amd/env_check.sh)

Again, please go through the entire documentation to confirm your system is using the recommended configuration.

## Installing SGLang

For general installation instructions, see the official [SGLang Installation Docs](https://docs.sglang.ai/start/install.html). Below are the AMD-specific steps summarized for convenience.

### Install from Source

```bash
# Use the last release branch
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install sgl-kernel --force-reinstall --no-deps
pip install -e "python[all_hip]"
```

### Install Using Docker

> **NOTE:** Replace `<secret>` below with your huggingface hub [token](https://huggingface.co/docs/hub/en/security-tokens)

1. Build the Docker Image

```bash
docker build -t sglang_image -f Dockerfile.rocm .
```

2. Create a Convenient Alias

```bash
alias drun='docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME/dockerx:/dockerx \
    -v /data:/data'
```

3. Launch the Server

```bash
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path NousResearch/Meta-Llama-3.1-8B \
    --host 0.0.0.0 \
    --port 30000
```

4. In a Different Terminal, Run a Benchmark

```bash
drun sglang_image \
    python3 -m sglang.bench_serving \
    --backend sglang \
    --dataset-name random \
    --num-prompts 4000 \
    --random-input 128 \
    --random-output 128
```

With your AMD system properly configured and SGLang installed, you can now fully leverage AMD hardware to power SGLang’s machine learning capabilities.
