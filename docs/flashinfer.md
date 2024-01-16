## Flashinfer Mode

[flashinfer](https://github.com/flashinfer-ai/flashinfer) is a kernel library for LLM serving.
It can be used in SGLang runtime to accelerate attention computation.

### Install flashinfer

Note: The compilation can take a very long time.

```bash
git submodule update --init --recursive
pip install 3rdparty/flashinfer/python
```

### Run a Server With Flashinfer Mode

Add `--model-mode flashinfer` argument to enable flashinfer when launching a server.

Example:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --model-mode flashinfer
```
