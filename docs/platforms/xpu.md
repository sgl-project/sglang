# XPU

The document addresses how to set up the [SGLang](https://github.com/sgl-project/sglang) environment and run LLM inference on Intel GPU, [see more context about Intel GPU support within PyTorch ecosystem](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html).

Specifically, SGLang is optimized for [Intel® Arc™ Pro B-Series Graphics](https://www.intel.com/content/www/us/en/ark/products/series/242616/intel-arc-pro-b-series-graphics.html) and [
Intel® Arc™ B-Series Graphics](https://www.intel.com/content/www/us/en/ark/products/series/240391/intel-arc-b-series-graphics.html).

## Optimized Model List

A list of LLMs have been optimized on Intel GPU, and more are on the way:

| Model Name | BF16 |
|:---:|:---:|
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| Qwen2.5-1.5B |   [Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) |

**Note:** The model identifiers listed in the table above
have been verified on [Intel® Arc™ B580 Graphics](https://www.intel.com/content/www/us/en/products/sku/241598/intel-arc-b580-graphics/specifications.html).

## Installation

### Install From Source

Currently SGLang XPU only supports installation from source. Please refer to ["Getting Started on Intel GPU"](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html) to install XPU dependency.

```bash
# Create and activate a conda environment
conda create -n sgl-cpu python=3.12 -y
conda activate sgl-cpu

# Set PyTorch CPU as primary pip install channel to avoid installing the larger CUDA-enabled version and prevent potential runtime issues.
pip3 install torch==2.8.0 torchvision torchaudio pytorch-triton-xpu==3.4.0 --index-url https://download.pytorch.org/whl/xpu

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

# Use dedicated toml file
cp python/pyproject_xpu.toml python/pyproject.toml
# Install SGLang dependent libs, and build SGLang main package
pip install --upgrade pip setuptools
conda install -y libsqlite==3.48.0 gperftools tbb libnuma numactl
pip install -e "python[all_xpu]"

# Other required environment variables
# Recommend to set these in ~/.bashrc in order not to set every time in a new terminal
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so:${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libtbbmalloc.so.2
```

### Install Using Docker

The docker for XPU is under active development. Please stay tuned.

## Launch of the Serving Engine

Example command to launch SGLang serving:

```bash
python -m sglang.launch_server       \
    --model <MODEL_ID_OR_PATH>       \
    --trust-remote-code              \
    --disable-overlap-schedule       \
    --device xpu                     \
    --host 0.0.0.0                   \
    --tp 2                           \   # using multi GPUs
    --attention-backend intel_xpu    \   # using intel optimized XPU attention backend
    --page-size                      \   # intel_xpu attention backend supports [32, 64, 128]
```

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
[OpenAI Completions API](https://docs.sglang.ai/basic_usage/openai_api_completions.html)
and sent via the command line (e.g. using `curl`) or via your own script.
