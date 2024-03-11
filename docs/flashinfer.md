## Flashinfer Mode

[flashinfer](https://github.com/flashinfer-ai/flashinfer) is a kernel library for LLM serving.
It can be used in SGLang runtime to accelerate attention computation.

### Install flashinfer

See https://docs.flashinfer.ai/installation.html.

### Run a Server With Flashinfer Mode

Add `--enable-flashinfer` argument to enable flashinfer when launching a server.

Example:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --enable-flashinfer
```
