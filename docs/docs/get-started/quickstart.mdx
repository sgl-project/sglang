---
title: "Quickstart"
description: "Start a model server and verify it with one request."
---

## Install and start

These commands target Linux with an NVIDIA GPU and a CUDA 13-compatible driver. See [Installation](/docs/get-started/install) for environment requirements and other setup options.

<Tabs>
<Tab title="Docker">

With Docker and NVIDIA Container Toolkit configured, pull the image. Skip this command if you have already pulled it:

```bash
docker pull lmsysorg/sglang:latest
```

Start the server on the GPU host. Replace `MODEL_PATH` with your Hugging Face model ID (for example, `Qwen/Qwen3-0.6B`):

```bash
docker run --rm --gpus all --ipc=host \
  -p 30000:30000 \
  lmsysorg/sglang:latest \
  sglang serve MODEL_PATH --host 0.0.0.0 --port 30000
```

</Tab>
<Tab title="uv">

With uv installed, create a Python environment and install SGLang. Skip this block if you already have an environment with SGLang installed:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --prerelease=allow sglang
```

Start the server in that environment. Replace `MODEL_PATH` with your Hugging Face model ID (for example, `Qwen/Qwen3-0.6B`):

```bash
sglang serve MODEL_PATH --host 0.0.0.0 --port 30000
```

</Tab>
</Tabs>

For additional model-specific launch arguments, see the [Cookbook](/cookbook/intro). Wait for `The server is fired up and ready to roll!`.

## Send a request

In another terminal on the same host, send a request to verify the server. Use the same model ID for `MODEL_PATH`:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MODEL_PATH",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

Check `choices[0].message.content` for the answer. Press `Ctrl+C` in the server terminal to stop.

See [Examples](/docs/get-started/examples) for a complete 27B deployment, streaming, and tool calls.
