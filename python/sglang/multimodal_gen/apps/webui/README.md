# SGLang Diffusion WebUI User Guide

SGLang Diffusion WebUI provides an intuitive Gradio-based interface for image and video generation, supporting parameter tuning and real-time previews.

## Prerequisites
The WebUI runs on Gradio. To get started, install Gradio first:
```bash
pip install gradio==6.1.0
```

## Launch WebUI Service
SGLang Diffusion now includes an integrated WebUI. Simply add the `--webui` parameter when starting the service.

### Launch Text-to-Image Service
```bash
SERVER_ARGS=(
  --model-path black-forest-labs/FLUX.1-dev
  --num-gpus 2
)
WEBUI_PORT=2333
sglang serve "${SERVER_ARGS[@]}" --webui --webui-port ${WEBUI_PORT}
```
### Launch Text-to-Video Service
```bash
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --num-gpus 2
)
WEBUI_PORT=2333
sglang serve "${SERVER_ARGS[@]}" --webui --webui-port ${WEBUI_PORT}
```

## Port Forwarding
Once the WebUI service is running, you need to use **SSH port forwarding** to securely access the remote service from your local machine.

In most cases:​ Your IDE (like VS Code, Cursor, etc.) can handle this automatically. Check your IDE's remote development or port forwarding features. Otherwise, execute this command manually.

```bash
ssh -L ${WEBUI_PORT}:localhost:${WEBUI_PORT} user_name@machine_name
```
Learn more about port forwarding:​ [Port Forwarding](https://en.wikipedia.org/wiki/Port_forwarding).


## Interface Instructions

1. Task mode is automatically determined by the `num_frames` parameter:
  - num_frames = 1: Text-to-Image mode
  - num_frames > 1: Text-to-Video mode
2. After generation, manually click:
  - Image output: View generated images
  - Video output: View generated videos

Once launched, access the interface at `http://localhost:${WEBUI_PORT}` in your browser.
