## Flashinfer Mode

[flashinfer](https://github.com/flashinfer-ai/flashinfer) is a kernel library for LLM serving.
It can be used in SGLang runtime to accelerate attention computation.

### Install flashinfer

You can install flashinfer via pip as follows for CUDA 12.1.

```bash
pip install flashinfer -i https://flashinfer.ai/whl/cu121/
```

You can look for other CUDA versions in https://github.com/flashinfer-ai/flashinfer?tab=readme-ov-file#installation. If there is no desire version for your environment,
please build it from source (the compilation takes a long time).

### Run a Server With Flashinfer Mode

Add `--model-mode flashinfer` argument to enable flashinfer when launching a server.

Example:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --model-mode flashinfer
```
