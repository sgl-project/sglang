# MindSpore Models

## Introduction

SGLang support run MindSpore framework models, this doc guide users to run mindspore models with SGLang.

## Requirements

MindSpore with SGLang current only support Ascend Npu device, users need first install Ascend CANN software packages.
The CANN software packages can download from the [Ascend Official Websites](https://www.hiascend.com). The version depends on the MindSpore version [MindSpore Installation](https://www.mindspore.cn/install)

## Supported Models

Currently, the following models are supported:

- **Qwen3**: Dense models supported. MoE models coming soon.
- *More models coming soon...*

## Installation

> **Note**: Currently, MindSpore models are provided by an independent package `sgl-mindspore`, which needs to be installed separately.

```shell
git clone https://github.com/chz34/sgl-mindspore.git
cd sgl-mindspore
pip install -e .
```

You will need to install the following packages, due to the support of tensor conversion through `dlpack` on 3rd devices, the minimum version of  `PyTorch` is 2.7.1

```shell
pip install mindspore
pip install "torch>=2.7.1"
pip install "torch_npu>=2.7.1"
pip install triton_ascend
```

```shell
pip install -e "python[all_npu]"
```

## Run Model

Current SGLang-MindSpore support Qwen3 dense model, this doc uses Qwen3-8B as example.

### Offline infer

Use the following script for offline infer:

```python
import sglang as sgl

# Initialize the engine with MindSpore backend
llm = sgl.Engine(
    model_path="/path/to/your/model",  # Local model path
    device="npu",                      # Use NPU device
    model_impl="mindspore",            # MindSpore implementation
    attention_backend="ascend",        # Attention backend
    tp_size=1,                         # Tensor parallelism size
    dp_size=1                          # Data parallelism size
)

# Generate text
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

sampling_params = {"temperature": 0.01, "top_p": 0.9}
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Generated: {output['text']}")
    print("---")
```

### Start server

Launch a server with MindSpore backend:

```bash
# Basic server startup
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1 \
    --dp-size 1
```

For distributed server with multiple nodes:

```bash
# Multi-node distributed server
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --dist-init-addr 127.0.0.1:29500 \
    --nnodes 2 \
    --node-rank 0 \
    --tp-size 4 \
    --dp-size 2
```

## Troubleshooting

#### Debug Mode

Enable sglang debug logging by log-level argument.

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --log-level DEBUG
```

Enable mindspore info and debug logging by setting environments.

```bash
export GLOG_v=1  # INFO
export GLOG_v=0  # DEBUG
```

#### Explicitly select devices

Use the following environment variable to explicitly select the devices to use.

```shell
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  # to set device
```

#### Some communication environment issues

In case of some environment with special communication environment, users need set some environment variables.

```shell
export MS_ENABLE_LCCL=off # current not support LCCL communication mode in SGLang-MindSpore
```

#### Some dependencies of protobuf

In case of some environment with special protobuf version, users need set some environment variables to avoid binary version mismatch.

```shell
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # to avoid protobuf binary version mismatch
```

## Support
For MindSpore-specific issues:

- Refer to the [MindSpore documentation](https://www.mindspore.cn/)
