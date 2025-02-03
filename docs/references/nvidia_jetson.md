# Apply SGLang on NVIDIA Jetson Orin

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
Please refer to [Installation Guide](https://docs.sglang.ai/start/install.html) to install FlashInfer and SGLang.
* * * * *

Running Inference
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
The quantization and limited context length (`--dtype half --context-length 8192`) are due to the limited computational resources in [Nvidia jetson kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/). A detailed explanation can be found in [Server Arguments](https://docs.sglang.ai/backend/server_arguments.html).

After launching the engine, refer to [Chat completions](https://docs.sglang.ai/backend/openai_api_completions.html#Usage) to test the usability.
* * * * *
Running quantization with TorchAO
-------------------------------------
TorchAO is suggested to NVIDIA Jetson Orin.
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
This enables TorchAO's int4 weight-only quantization with a 128-group size. The usage of `--torchao-config int4wo-128` is also for memory efficiency.


* * * * *
Structured output with XGrammar
-------------------------------
Please refer to [SGLang doc structured output](https://docs.sglang.ai/backend/structured_outputs.html).
* * * * *

Thanks to the support from [shahizat](https://github.com/shahizat).

References
----------
-   [NVIDIA Jetson AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)
