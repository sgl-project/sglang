## Flashinfer Mode

[`flashinfer`](https://github.com/flashinfer-ai/flashinfer) is a kernel library for LLM serving; we use it here to support our attention computation.

### Install flashinfer

Note: The compilation can take a very long time.

```bash
git submodule update --init --recursive
pip install 3rdparty/flashinfer/python
```

### Run Sever With Flashinfer Mode

Add through `--model_mode` argument from the command line.

Example:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000 --model-mode flashinfer
```
