# AMD GPUs

This document describes how run SGLang on AMD GPUs. If you encounter issues or have questions, please [open an issue](https://github.com/sgl-project/sglang/issues).

## System Configuration

When using AMD GPUs (such as MI300X), certain system-level optimizations help ensure stable performance. Here we take MI300X as an example. AMD provides official documentation for MI300X optimization and system tuning:

- [AMD MI300X Tuning Guides](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
- [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference/vllm-benchmark.html)
- [AMD Instinct MI300X System Optimization](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html)
- [AMD Instinct MI300X Workload Optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html)
- [Supercharge DeepSeek-R1 Inference on AMD Instinct MI300X](https://rocm.blogs.amd.com/artificial-intelligence/DeepSeekR1-Part2/README.html)

**NOTE:** We strongly recommend reading these docs and guides entirely to fully utilize your system.

Below are a few key settings to confirm or enable for SGLang:

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

## Install SGLang

You can install SGLang using one of the methods below.

### Install from Source

```bash
# Use the last release branch
git clone -b v0.5.5.post3 https://github.com/sgl-project/sglang.git
cd sglang

# Compile sgl-kernel
pip install --upgrade pip
cd sgl-kernel
python setup_rocm.py install

# Install sglang python package
cd ..
rm -rf python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[all_hip]"
```

### Install Using Docker (Recommended)

The docker images are available on Docker Hub at [lmsysorg/sglang](https://hub.docker.com/r/lmsysorg/sglang/tags), built from [rocm.Dockerfile](https://github.com/sgl-project/sglang/tree/main/docker).

The steps below show how to build and use an image.

1. Build the docker image.
   If you use pre-built images, you can skip this step and replace `sglang_image` with the pre-built image names in the steps below.

   ```bash
   docker build -t sglang_image -f rocm.Dockerfile .
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
     - `--network host` and `--privileged` are required by RDMA. If you don't need RDMA, you can remove them.
     - You may need to set `NCCL_IB_GID_INDEX` if you are using RoCE, for example: `export NCCL_IB_GID_INDEX=3`.

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

4. To verify the utility, you can run a benchmark in another terminal or refer to [other docs](https://docs.sglang.ai/basic_usage/openai_api_completions.html) to send requests to the engine.

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

The only difference when running DeepSeek-V3 is in how you start the server. Here's an example command:

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

[Running DeepSeek-R1 on a single NDv5 MI300X VM](https://techcommunity.microsoft.com/blog/azurehighperformancecomputingblog/running-deepseek-r1-on-a-single-ndv5-mi300x-vm/4372726) could also be a good reference.

### Running Llama3.1

Running Llama3.1 is nearly identical to running DeepSeek-V3. The only difference is in the model specified when starting the server, shown by the following example command:

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

When the server displays `The server is fired up and ready to roll!`, it means the startup is successful.
