# SGLang Diffusion WebUI User Guide

SGLang Diffusion WebUI provides an intuitive Gradio-based interface for image and video generation, supporting parameter
tuning and real-time previews.

## Prerequisites

The WebUI runs on Gradio. To get started, install Gradio first:

```bash
pip install gradio==6.1.0
```

## Launch WebUI Service

SGLang Diffusion now includes an integrated WebUI. Simply add the `--webui` parameter when starting the service.

### Launch Text-to-Image Service

```bash
sglang serve --model-path black-forest-labs/FLUX.1-dev --num-gpus 1 --webui --webui-port 2333
```

### Launch Text-to-Video Service

```bash
sglang serve --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers --num-gpus 1 --webui --webui-port 2333
```

### Launch Image-to-Image Service
```bash
sglang serve --model-path Qwen/Qwen-Image-Edit-2511 --num-gpus 1 --webui --webui-port 2333
```

### Launch Image-to-Video Service
```bash
sglang serve --model-path Wan-AI/Wan2.2-TI2V-5B-Diffusers --num-gpus 1 --webui --webui-port 2333
```

### Launch MiniMax H3

MiniMax H3 uses a native joint video/audio request contract. Select the weight
partition at server startup:

```bash
# Serves text-to-video-with-audio (t2va) and first/last-frame-to-video-with-audio (fl2va).
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant fl2va \
  --num-gpus 4 --ulysses-degree 4 --webui --webui-port 2333

# Serves reference-to-video-with-audio (ref2va).
sglang serve --model-path MiniMaxAI/MiniMax-H3 --model-variant ref2va \
  --num-gpus 4 --ulysses-degree 4 --webui --webui-port 2333
```

The WebUI exposes H3's `task`, conditioning media, target short edge/aspect
ratio/duration, joint denoising steps, video/audio flow shifts, and seed. H3 is CFG-distilled, so the generic negative prompt,
guidance scales, manual FPS/frame count, width/height, and TeaCache controls do
not apply. H3 output is fixed at 24 FPS, with its frame count and canvas derived
from the target.

## Port Forwarding

Once the WebUI service is running, you need to use **SSH port forwarding** to securely access the remote service from
your local machine.

In most cases: Your IDE (like VS Code, Cursor, etc.) can handle this automatically. Check your IDE's remote development
or port forwarding features. Otherwise, execute this command manually.

```bash
ssh -L ${WEBUI_PORT}:localhost:${WEBUI_PORT} user_name@machine_name
```

Learn more about port forwarding: [Port Forwarding](https://en.wikipedia.org/wiki/Port_forwarding).

## Interface Instructions

You can view your model path and task name directly in the UI. We'd appreciate any feedback you'd like to share.

Once launched, access the interface at `http://localhost:${WEBUI_PORT}` in your browser.
