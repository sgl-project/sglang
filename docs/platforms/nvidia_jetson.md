# NVIDIA Jetson Orin

## Prerequisites

Before starting, ensure the following:

- [**NVIDIA Jetson AGX Orin Devkit**](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/) is set up with **JetPack 6.1** or later.
- **CUDA Toolkit** and **cuDNN** are installed.
- Verify that the Jetson AGX Orin is in **high-performance mode**:
```bash
sudo nvpmodel -m 0
```
* * * * *
## Installing and running SGLang with Jetson Containers
Clone the jetson-containers github repository:
```
git clone https://github.com/dusty-nv/jetson-containers.git
```
Run the installation script:
```
bash jetson-containers/install.sh
```
Build the container image:
```
jetson-containers build sglang
```
Run the container:
```
jetson-containers run $(autotag sglang)
```
Or you can also manually run a container with this command:
```
docker run --runtime nvidia -it --rm --network=host IMAGE_NAME
```
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
The quantization and limited context length (`--dtype half --context-length 8192`) are due to the limited computational resources in [Nvidia jetson kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/). A detailed explanation can be found in [Server Arguments](../backend/server_arguments.md).

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
Please refer to [SGLang doc structured output](../advanced_features/structured_outputs.ipynb).
* * * * *

Thanks to the support from [Nurgaliyev Shakhizat](https://github.com/shahizat), [Dustin Franklin](https://github.com/dusty-nv) and [Johnny Núñez Cano](https://github.com/johnnynunez).

References
----------
-   [NVIDIA Jetson AGX Orin Documentation](https://developer.nvidia.com/embedded/jetson-agx-orin)
