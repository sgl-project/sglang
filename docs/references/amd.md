# SGLang on AMD

## Introduction

This document describes how to set up an AMD-based environment for [SGLang](https://github.com/sgl-project/sglang). If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues) on the SGLang repository.

## System Configure

When using AMD GPUs (such as MI300X), certain system-level optimizations help ensure stable performance. Here we take MI300X as an example. AMD provides official documentation for MI300X optimization and system tuning:

- [AMD MI300X Tuning Guides](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
  - [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html)
  - [AMD Instinct MI300X System Optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html)
  - [AMD Instinct MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)

**NOTE:** We strongly recommend reading theses docs entirely guide to fully utilize your system.

Below are a few key settings to confirm or enable:

### Update GRUB Settings

In `/etc/default/grub`, append the following to `GRUB_CMDLINE_LINUX`:

```text
pci=realloc=off iommu=pt
```

Afterward, run `sudo update-grub` (or your distro’s equivalent) and reboot.

### Disable NUMA Auto-Balancing

```bash
sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'
```

You can automate or verify this change using [this helpful script](https://github.com/ROCm/triton/blob/rocm_env/scripts/amd/env_check.sh).

Again, please go through the entire documentation to confirm your system is using the recommended configuration.

## Installing SGLang

For general installation instructions, see the official [SGLang Installation Docs](https://docs.sglang.ai/start/install.html). Below are the AMD-specific steps summarized for convenience.

### Install from Source

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang

pip install --upgrade pip
pip install sgl-kernel --force-reinstall --no-deps
pip install -e "python[all_hip]"
```

### Install Using Docker (Recommended)

1. Build the docker image.

```bash
docker build -t sglang_image -f Dockerfile.rocm .
```

2. Create a convenient alias.

```bash
alias drun='docker run -it --rm --network=host --privileged --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -v $HOME/dockerx:/dockerx \
    -v /data:/data'
```

If you are using RDMA, please note that:

1. `--network host` and `--privileged` are required by RDMA. If you don't need RDMA, you can remove them.
2. You may need to set `NCCL_IB_GID_INDEX` if you are using RoCE, for example: `export NCCL_IB_GID_INDEX=3`.

3. Launch the server.

**NOTE:** Replace `<secret>` below with your [huggingface hub token](https://huggingface.co/docs/hub/en/security-tokens).

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

4. To verify the utility, you can run a benchmark in another terminal or refer to [other docs](https://docs.sglang.ai/backend/openai_api_completions.html) to send requests to the engine.

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

## Examples

### Running DeepSeek-V3

The only difference in running DeepSeek-V3 is when starting the server. Here's an example command:

```bash
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \ # <- here
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```

### Running Llama3.1

Running Llama3.1 is nearly identical. The only difference is in the model specified when starting the server, shown by the following example command:

```bash
drun -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    --env "HF_TOKEN=<secret>" \
    sglang_image \
    python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \ # <- here
    --tp 8 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 30000
```

### Warmup Step

When the server displays "The server is fired up and ready to roll!", it means the startup is successful.
