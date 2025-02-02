# Step-by-Step Guide to Use SGLang on NVIDIA Jetson Orin platform

This is a replicate from https://github.com/shahizat/SGLang-Jetson, thanks to the support from [shahizat](https://github.com/shahizat).
## Prerequisites

Before starting, ensure the following:

- [**NVIDIA Jetson AGX Orin Devkit**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) is set up with **JetPack 6.1** or later.
- **CUDA Toolkit** and **cuDNN** are installed.
- Verify that the Jetson AGX Orin is in **high-performance mode**:
  ```bash
  sudo nvpmodel -m 0
  ```
- A custom PyPI index hosted at https://pypi.jetson-ai-lab.dev/jp6/cu126, tailored for NVIDIA Jetson Orin platforms and CUDA 12.6.

To install torch from this index:
  ```bash
pip install torch --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
 ```
* * * * *
## Installation
Installation guide for FlashInfer and SGLang please refer to [[Installation Guide]](https://docs.sglang.ai/start/install.html)
* * * * *

Running Inference with FlashInfer Backend
-----------------------------------------

Launch the server:
```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --device cuda \
  --dtype half \
  --attention-backend flashinfer \
  --mem-fraction-static 0.8 \
  --context-length 8192 
```
The quantization and  limited context length`--dtype half --context-length 8192` are due to the limited computational resources in [Nvidia jetson kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/). \
`--dtype half` refer to the arg in [server_args.py](https://github.com/sgl-project/sglang/blob/959dca4fc7d720b8885e74761f7b098bed2bdeb7/python/sglang/srt/server_args.py#L347) \
`--context-length 8192` same as above. 

Run a sample inference script:\
    Create a Python script (e.g., `inference.py`) with the following content:\
    Please refer to the documentation of [Chat completions in SGLang doc](https://docs.sglang.ai/backend/openai_api_completions.html#Usage)


```
Performance metrics
```bash
[2025-01-26 21:32:18 TP0] Decode batch. #running-req: 1, #token: 1351, token usage: 0.01, gen throughput (token/s): 11.19, #queue-req: 0
```
* * * * *

Running Inference with other Attension Backend
-------------------------------------------

Launch the server:
```bash
python -m sglang.launch_server \
      --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
      --device cuda \
      --mem-fraction-static 0.8 \
      --context-length 8192
```
Torch Native Backend \
 `--attention-backend torch_native `

Triton Backend\
`--attention-backend triton`

* * * * *
Running quantization with TorchAO
-------------------------------------
Launch the server with the best configuration:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --device cuda \
    --dtype bfloat16 \
    --attention-backend flashinfer \
    --mem-fraction-static 0.8 \
    --context-length 8192 \
    --torchao-config int4wo-128
```
This enables TorchAO's int4 weight-only quantization with 128-group size.
The usage of `--torchao-config int4wo-128` is a result of memory efficiency. 

```bash
[2025-01-27 00:06:47 TP0] Decode batch. #running-req: 1, #token: 115, token usage: 0.00, gen throughput (token/s): 30.84, #queue-req:
```

* * * * *
Structured output with XGrammar
-------------------------------
Please refer to [SGLang doc structured output](https://docs.sglang.ai/backend/structured_outputs.html)
* * * * *

References
----------

-   [SGLang Official Documentation](https://docs.sglang.ai/index.html)

-   [FlashInfer GitHub Repository](https://github.com/flashinfer-ai/flashinfer)

-   [SGLang GitHub Repository](https://github.com/sgl-project/sglang)

-   [NVIDIA Jetson AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)
