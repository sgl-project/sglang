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
sglang serve black-forest-labs/FLUX.1-dev --num-gpus 1 --webui --webui-port 2333
```

### Launch Text-to-Video Service

```bash
sglang serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --num-gpus 1 --webui --webui-port 2333
```

### Launch Image-to-Image Service
```bash
sglang serve --model-path Qwen/Qwen-Image-Edit-2511 --num-gpus 1 --webui --webui-port 2333
```

### Launch Image-to-Video Service
```bash
sglang serve Wan-AI/Wan2.2-TI2V-5B-Diffusers --num-gpus 1 --webui --webui-port 2333
```

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
