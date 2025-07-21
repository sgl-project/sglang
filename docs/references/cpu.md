# SGLang on CPU

The document addresses how to set up the [SGLang](https://github.com/sgl-project/sglang) environment and run LLM inference on CPU servers.
Specifically, SGLang is well optimized on the CPUs equipped with Intel® AMX® Instructions,
which are 4th generation or newer Intel® Xeon® Scalable Processors.

## Optimized Model List

A list of popular LLMs are optimized and run efficiently on CPU,
including the most notable open-source models like Llama series, Qwen series,
and the phenomenal high-quality reasoning model DeepSeek-R1.

| Model Name | BF16 | w8a8_int8 | FP8 |
|:---:|:---:|:---:|:---:|
| DeepSeek-R1 |   | [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8) | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [RedHatAI/Llama-3.2-3B-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8) |   |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8) |   |
| QwQ-32B |   | [RedHatAI/QwQ-32B-quantized.w8a8](https://huggingface.co/RedHatAI/QwQ-32B-quantized.w8a8) |   |
| DeepSeek-Distilled-Llama |   | [RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8](https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8) |   |
| Qwen3-235B |   |   | [Qwen/Qwen3-235B-A22B-FP8](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8) |

**Note:** The model identifiers listed in the table above
have been verified on 6th Gen Intel® Xeon® P-core platforms.

## Installation

### Install Using Docker

It is recommended to use Docker for setting up the SGLang environment.
A [Dockerfile](https://github.com/sgl-project/sglang/blob/main/docker/Dockerfile.xeon) is provided to facilitate the installation.
Replace `<secret>` below with your [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens).

```bash
# Clone the SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/docker

# Build the docker image
docker build -t sglang-cpu:main -f Dockerfile.xeon .

# Initiate a docker container
docker run \
    -it \
    --privileged \
    --ipc=host \
    --network=host \
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<secret>" \
    sglang-cpu:main /bin/bash
```

### Install From Source

If you'd prefer to install SGLang in a bare metal environment,
the command list is as below.
It is worth noting that the environment variable `SGLANG_USE_CPU_ENGINE=1`
is required to enable SGLang service with CPU engine.

```bash
# Create and activate a conda environment
conda create -n sgl-cpu python=3.12 -y
conda activate sgl-cpu

# Optional: Set PyTorch CPU as primary pip install channel to avoid installing CUDA version
pip config set global.index-url https://download.pytorch.org/whl/cpu
pip config set global.extra-index-url https://pypi.org/simple

# Check if some conda related environment variables have been set
env | grep -i conda
# The following environment variable settings are required
# if they have not been set properly
export CONDA_EXE=$(which conda)
export CONDA_ROOT=${CONDA_EXE}/../..
export CONDA_PREFIX=${CONDA_ROOT}/envs/sgl-cpu
export PATH=${PATH}:${CONDA_ROOT}/bin:${CONDA_ROOT}/condabin

# Clone the SGLang code
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout <YOUR-DESIRED-VERSION>

# Install SGLang dependent libs, and build SGLang main package
pip install --upgrade pip setuptools
conda install -y libsqlite==3.48.0 gperftools tbb libnuma numactl
pip install intel-openmp
pip install -e "python[all_cpu]"

# Build the CPU backend kernels
cd sgl-kernel
cp pyproject_cpu.toml pyproject.toml
pip install -v .

# Other required environment variables
# Recommend to set these in ~/.bashrc in order not to set every time in a new terminal
export SGLANG_USE_CPU_ENGINE=1
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libtbbmalloc.so.2
```

## Launch of the Serving Engine

Example command to launch SGLang serving:

```bash
python -m sglang.launch_server   \
    --model <MODEL_ID_OR_PATH>   \
    --trust-remote-code          \
    --disable-overlap-schedule   \
    --device cpu                 \
    --host 0.0.0.0               \
    --tp 6
```

Notes:

1. For running W8A8 quantized models, please add the flag `--quantization w8a8_int8`.

2. The flag `--tp 6` specifies that tensor parallelism will be applied using 6 ranks (TP6).
    The number of TP specified is how many TP ranks will be used during the execution.
    In a CPU platform, a TP rank means a sub-NUMA cluster (SNC).
    Usually we can get the SNC information (How many available) from Operation System.
    User can specify TP to be no more than the total available SNCs in current system.

    If the specified TP rank number differs from the total SNC count,
    the system will automatically utilize the first `n` SNCs.
    Note that `n` cannot exceed the total SNC number, doing so will result in an error.

    To specify the cores to be used, we need to explicitly set the environment variable `SGLANG_CPU_OMP_THREADS_BIND`.
    For example, if we want to run the SGLang service using the first 40 cores of each SNC on a Xeon® 6980P server,
    which has 43-43-42 cores on the 3 SNCs of a socket, we should set:

    ```bash
    export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167|171-210|214-253"
    ```

3. A warmup step is automatically triggered when the service is started.
The server is ready when you see the log `The server is fired up and ready to roll!`.

## Benchmarking with Requests

You can benchmark the performance via the `bench_serving` script.
Run the command in another terminal.

```bash
python -m sglang.bench_serving   \
    --dataset-name random        \
    --random-input-len 1024      \
    --random-output-len 1024     \
    --num-prompts 1              \
    --request-rate inf           \
    --random-range-ratio 1.0
```

The detail explanations of the parameters can be looked up by the command:

```bash
python -m sglang.bench_serving -h
```

Additionally, the requests can be formed with
[OpenAI Completions API](https://docs.sglang.ai/backend/openai_api_completions.html)
and sent via the command line (e.g. using `curl`) or via your own script.

## Example: Running DeepSeek-R1

An example command to launch service for W8A8 DeepSeek-R1 on a Xeon® 6980P server

```bash
python -m sglang.launch_server                 \
    --model meituan/DeepSeek-R1-Channel-INT8   \
    --trust-remote-code                        \
    --disable-overlap-schedule                 \
    --device cpu                               \
    --quantization w8a8_int8                   \
    --host 0.0.0.0                             \
    --mem-fraction-static 0.8                  \
    --max-total-token 65536                    \
    --tp 6
```

Similarly, an example command to launch service for FP8 DeepSeek-R1 would be

```bash
python -m sglang.launch_server                 \
    --model deepseek-ai/DeepSeek-R1            \
    --trust-remote-code                        \
    --disable-overlap-schedule                 \
    --device cpu                               \
    --host 0.0.0.0                             \
    --mem-fraction-static 0.8                  \
    --max-total-token 65536                    \
    --tp 6
```

Then you can test with `bench_serving` command or construct your own command or script
following [the benchmarking example](#benchmarking-with-requests).
