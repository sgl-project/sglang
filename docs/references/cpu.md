# SGLang on CPU

The document addresses how to set up the [SGLang](https://github.com/sgl-project/sglang) environment and run LLM inference on CPU servers.
Specifically, the model service is well optimized on the CPUs equipped with Intel® AMX® Instructions,
which are the 4th or newer Gen of Intel® Xeon® Scalable Processors.

## Optimized Model List

A list of popular LLMs are optimized and run efficiently on CPU,
including the most notable open-source models like Llama series, Qwen series,
and the phenomenal high-quality reasoning model DeepSeek-R1.

| Model Name | BF16 | w8a8_int8 | w4a16 | FP8 |
|:---:|:---:|:---:|:---:|
| DeepSeek-R1 |   | meituan/DeepSeek-R1-Channel-INT8 |   | deepseek-ai/DeepSeek-R1 |
| Llama-3.2-3B | meta-llama/Llama-3.2-3B-Instruct | RedHatAI/Llama-3.2-3B-quantized.w8a8 | AMead10/Llama-3.2-3B-Instruct-AWQ |   |

**Note:** In the above table, if the model ID is exhibited in the grid,
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
    -v /dev/shm:/dev/shm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 30000:30000 \
    -e "HF_TOKEN=<secret>" \
    sglang-cpu:main /bin/bash
```

If you'd prefer to install SGLang in a bare metal environment,
please take the command list in the Dockerfile as reference.

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

1. For running INT8 quantized models, please add the flag `--quantization w8a8_int8`.

2. The flag `--tp 6` indicates that we will apply tensor parallel with 6 ranks (TP6).
In general the TP rank number should be in line with the total number of sub-numa clusters (SNCs) on the server
(e.g. TP6 should be applied on a server having 2 sockets with SNC3 configuration).

    If the desired TP rank number is not the same with total SNC number, an explicit setting of env variable
    `SGLANG_CPU_OMP_THREADS_BIND` is needed. For example, if we want to run TP3 on the 1st socket of the server
    with 120cc x 2 sockets, which has totally 6 SNCs, we need to set

    ```bash
    export SGLANG_CPU_OMP_THREADS_BIND="0-39|40-79|80-119"
    ```

    and set `--tp 3` in the `launch_server` command.

3. An warmup step is automatically triggered when the service is started.
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
