# SGLang on CPU

The document addresses how to set up the [SGLang](https://github.com/sgl-project/sglang) environment and run LLM inference on CPU servers.
Specifically, the model service is well optimized on the CPUs equipped with Intel® AMX® Instructions,
which are 4th generation or newer Intel® Xeon® Scalable Processors.

## Optimized Model List

A list of popular LLMs are optimized and run efficiently on CPU,
including the most notable open-source models like Llama series, Qwen series,
and the phenomenal high-quality reasoning model DeepSeek-R1.

| Model Name | BF16 | w8a8_int8 | FP8 |
|:---:|:---:|:---:|:---:|
| DeepSeek-R1 |   | [meituan/DeepSeek-R1-Channel-INT8](https://huggingface.co/meituan/DeepSeek-R1-Channel-INT8) | [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| Llama-3.2-1B | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | [RedHatAI/Llama-3.2-1B-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-1B-Instruct-quantized.w8a8) |   |
| Llama-3.2-3B | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | [RedHatAI/Llama-3.2-3B-quantized.w8a8](https://huggingface.co/RedHatAI/Llama-3.2-3B-Instruct-quantized.w8a8) |   |
| Llama-3.1-8B | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8](https://huggingface.co/RedHatAI/Meta-Llama-3.1-8B-quantized.w8a8) |   |
| QwQ-32B |   | [RedHatAI/QwQ-32B-quantized.w8a8](https://huggingface.co/RedHatAI/QwQ-32B-quantized.w8a8) |   |
| DeepSeek-Distilled-Llama |   | [RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8](https://huggingface.co/RedHatAI/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8) |   |

**Note:** In the above table, if the model identifier is listed in the grid,
it means the model is verified.

## Installation

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

If you'd prefer to install SGLang in a bare metal environment,
please take the command list in the Dockerfile as reference.
Please note that the environmental variable `SGLANG_USE_CPU_ENGINE=1`
is required to be set to enable SGLang service with CPU engine.

## Launch of the Serving Engine

An example command for LLM serving engine launching would be:

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
    In general, the number of TP ranks should correspond to the total number of sub-NUMA clusters (SNCs) on the server.
    For instance, TP6 is suitable for a server with 2 sockets configured with SNC3.

    If the desired TP rank number differs from the total SNC count, the system will automatically
    utilize the first `n` SNCs, provided `--tp n` is set and `n` is less than the total SNC count.
    Note that `n` cannot exceed the total SNC number, doing so will result in an error.

    To specify the cores to be used, we need to explicitly set the environment variable `SGLANG_CPU_OMP_THREADS_BIND`.
    For example, if we want to run the SGLang service using the first 40 cores of each SNC on a Xeon® 6980P server,
    which has 43-43-42 cores per socket (reserving the remaining 16 cores for other tasks), we should set:

    ```bash
    export SGLANG_CPU_OMP_THREADS_BIND="0-39|43-82|86-125|128-167|171-210|214-253"
    ```

    Additionally, include `--tp 6` in the `launch_server` command.

3. A warmup step is automatically triggered when the service is started.
When `The server is fired up and ready to roll!` is echoed,
the server is ready to handle the incoming requests.

## Benchmarking with Requests

We can benchmark the performance via the `bench_serving` script.
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

The example W8A8 model service launching command in the container on a Xeon® 6980P server would be

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

Similarly, the example FP8 model service launching command would be

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

Then we can test with `bench_serving` command or construct our own command or script
following [the benchmarking example](#benchmarking-with-requests).
