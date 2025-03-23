# NVIDIA Dynamo

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) is a high-throughput low-latency inference framework designed for serving generative AI and reasoning models in multi-node distributed environments.
This doc ports the examples from the [original repo](https://github.com/ai-dynamo/dynamo/blob/main/docs/guides/dynamo_run.md) to SGLang.

## Setup

Please note that you need Ubuntu 24.04 with a x86_64 CPU.

To ensure compatibility we recommend to use `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` as base image and run it on [docker](https://hub.docker.com/r/nvidia/cuda).
You will furthermore need the rust package manager [cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html) installed.

### Install dynamo

```
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0
python3 -m venv venv
source venv/bin/activate

pip install ai-dynamo[all]
```

### Install SGLang

```
pip install pip
pip install sgl-kernel --force-reinstall --no-deps
pip install "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/
```

### Install rust packages

Clone [the repo](https://github.com/ai-dynamo/dynamo), cd into it and run:

```
cargo build --features sglang
```

## Inference

### Simple chat CLI

In your terminal execute:

```
dynamo run out=sglang Qwen/Qwen2.5-3B-Instruct
```

You can than interact with the model in your terminal:

```
✔ User · What is the capital of France?
The capital of France is Paris.
```

### Server

Start a server:

```
dynamo run in=http out=sglang Qwen/Qwen2.5-3B-Instruct
```

You will recieve a message informing you if the server is up and running.

```
2025-03-22T20:32:52.029318Z  INFO dynamo_llm::engines::sglang::worker: Waiting for sglang0 to signal that it's ready
2025-03-22T20:33:13.629792Z  INFO dynamo_llm::engines::sglang::worker: sglang0 is ready
2025-03-22T20:33:14.477283Z  INFO dynamo_llm::http::service::service_v2: Starting HTTP service on: 0.0.0.0:8080 address="0.0.0.0:8080"
```

You can than send a request to the server by executing the following code in a separate terminal

```
curl -d '{"model": "Qwen2.5-3B-Instruct", "max_completion_tokens": 2049, "messages":[{"role":"user", "content": "What is the capital of South Africa?" }]}' -H 'Content-Type: application/json' http://localhost:8080/v1/chat/completions
```

and recieve the output

```
{"id":"chatcmpl-f56db541-b7f6-461c-baf8-0182208e760d","choices":[{"index":0,"message":{"content":"The capital of South Africa is Pretoria. However, it's important to note that Pretoria is often referred to as the administrative capital. The legislative capital is Cape Town, where the Parliament of South Africa is located. The de facto capital, where the President's official residence and office are located, is often considered to be Johannesburg.","refusal":null,"tool_calls":null,"role":"assistant","function_call":null,"audio":null},"finish_reason":null,"logprobs":null}],"created":1742675773,"model":"Qwen2.5-3B-Instruct","service_tier":null,"system_fingerprint":null,"object":"chat.completion","usage":{"prompt_tokens":37,"completion_tokens":67,"total_tokens":0,"prompt_tokens_details":null,"completion_tokens_details":null}}
```

### Multi Node

You may also run multi node inference by executing the following:

```
dynamo-run in=http out=sglang --model-path ~/llm_models/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 8 --num-nodes 2 --node-rank 0 --dist-init-addr 10.217.98.122:9876
```
```
dynamo-run in=none out=sglang --model-path ~/llm_models/DeepSeek-R1-Distill-Llama-70B/ --tensor-parallel-size 8 --num-nodes 2 --node-rank 1 --dist-init-addr 10.217.98.122:9876
```

### Batch Inference

To run batch inference please prepare  `input.jsonl` file with content like this:

```
{"text": "What is the capital of France?"}
{"text": "What is the capital of Spain?"}
```

Run batch inference:

```
dynamo-run in=batch:prompts.jsonl out=llamacpp <model>
```

This will create an `output.jsonl` with the inference results:

```
{"text":"What is the capital of France?","response":"The capital of France is Paris.","tokens_in":7,"tokens_out":7,"elapsed_ms":1566}
{"text":"What is the capital of Spain?","response":".The capital of Spain is Madrid.","tokens_in":7,"tokens_out":7,"elapsed_ms":855}
```
