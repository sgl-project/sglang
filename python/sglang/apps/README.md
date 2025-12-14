# SGLang diffusion webui user guide

The SGLang-diffusion webui provides a web interface for image and video generation, implemented using Gradio library.


## 1. Launch server within webui
When launching http server within webui, the http server will be launched from another subprocess to release control of main process.
Finally, the webui will occupy the terminal.

### webui example script for t2i and t2v

```bash
## text to image
SERVER_ARGS=(
  --model-path black-forest-labs/FLUX.1-dev
  --num-gpus 2
)
sglang serve "${SERVER_ARGS[@]}" --webui --webui-port 2333

## text to video
SERVER_ARGS=(
  --model-path Wan-AI/Wan2.2-T2V-A14B-Diffusers
  --num-gpus 2
)
sglang serve "${SERVER_ARGS[@]}" --webui --webui-port 2333
```

## 2. With a pre-launched http server
  *WIP*
