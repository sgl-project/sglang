# ComfyUI SGLDiffusion Plugin

A ComfyUI plugin for integrating with SGLang Diffusion server, supporting image and video generation capabilities.

## Installation

1. **Install SGLang**: Follow the [Installation Guide](https://docs.sglang.io/docs/sglang-diffusion/installation) to install `sglang[diffusion]`.
2. **Install Plugin**: Copy this entire directory (`ComfyUI_SGLDiffusion`) to your ComfyUI `custom_nodes/` folder.
3. **Restart ComfyUI**: Restart ComfyUI to load the plugin.

## Usage

The plugin supports two modes of operation: **Server Mode** (via HTTP API) and **Integrated Mode** (tight integration with ComfyUI).

### Supported Models
- **Z-Image**: High-speed image generation models (e.g., `Z-Image-Turbo`)
- **FLUX**: State-of-the-art text-to-image models (e.g., `FLUX.1-dev`)
- **Qwen-Image**: Multi-modal image generation models (e.g., `Qwen-Image`,`Qwen-Image-2512`). *Note: Image editing support is currently experimental and may have some issues.*
- **MiniMax-H3**: Joint video-and-audio generation, server mode only (`SGLDiffusion Generate H3`)

### Mode 1: Server Mode (HTTP API)
Connect to a standalone SGLang Diffusion server.

1. **Start SGLang Diffusion Server**: Ensure the server is running and accessible.
2. **Connect to Server**: Use the `SGLDiffusion Server Model` node to connect (default: `http://localhost:3000/v1`).
3. **Generate Content**:
   - `SGLDiffusion Generate Image`: For text-to-image and image editing.
   - `SGLDiffusion Generate Video`: For text-to-video and image-to-video.
4. **LoRA Support**: Use `SGLDiffusion Server Set LoRA` and `SGLDiffusion Server Unset LoRA`.

### Mode 2: Integrated Mode (Tight Integration)
Leverage SGLang's high-performance sampling directly within ComfyUI while using ComfyUI's front-end nodes (CLIP, VAE, etc.).

1. **Load Model**: Use the `SGLDiffusion UNET Loader` node to load your diffusion model.
2. **Configure Options**: Use the `SGLDiffusion Options` node to set runtime parameters like `num_gpus`, `tp_size`, `model_type`, or `enable_torch_compile`.
3. **Sample**: Connect the loaded model to standard ComfyUI samplers. SGLang will handle the sampling process efficiently.
4. **LoRA Support**: Use the `SGLDiffusion LoRA Loader` for native LoRA integration.

## Adding a Model

Pick the mode before writing code; the wrong one costs several hundred lines
of weight mapping that buys nothing.

Take **Server Mode** when the model needs conditioning ComfyUI cannot supply
(audio, reference materials, task routing), emits more than one modality, or
has its own request contract. Reproducing that inside ComfyUI would duplicate
stages the server already runs.

- If the request fits the existing `generate_image` / `generate_video` fields,
  there is nothing to write — point the existing nodes at the server.
- If the model has extra request fields, pass them via `extra_fields`. The
  request schemas accept unknown keys, so `core/server_api.py` needs no
  per-model change.
- Add a node in `nodes.py` only to surface those inputs as ComfyUI widgets.
  `SGLDiffusionGenerateH3` is the worked example.

Take **Integrated Mode** only when the model denoises a single latent tensor
that ComfyUI already knows how to build and decode, so its KSampler can drive
the loop unchanged. Each model then needs:

- `runtime/pipelines/comfyui_<model>_pipeline.py` mapping ComfyUI's
  single-file checkpoint layout onto the native module tree (350-690 lines in
  the existing three)
- an executor in `executors/` adapting latent layout and conditioning to `Req`
- entries in both `pipeline_class_dict` and `executor_class_dict` in
  `core/generator.py`

## Example Workflows

Reference workflow files are provided in the `workflows/` directory:

- **`flux_sgld_sp.json`**: Multi-GPU (Sequence Parallelism) workflow for FLUX models. High-performance inference across multiple cards.
- **`qwen_image_sgld.json`**: Qwen-Image generation with LoRA support. Optimized for multi-modal image tasks.
- **`z-image_sgld.json`**: High-speed image generation using Z-Image.
- **`sgld_text2img.json`**: Server-mode text-to-image generation with LoRA support.
- **`sgld_image2video.json`**: Server-mode image-to-video generation.

For other workflows supporting the models, you can easily use SGLang by replacing the official `UNET Loader` node with the `SGLDUNETLoader` node. Similarly, for LoRA support, replace the official LoRA loader with the `SGLDiffusion LoRA Loader`.

To use these workflows:
1. Open ComfyUI.
2. Load the workflow JSON file from the `workflows/` directory.
3. Adjust the parameters and model paths as needed.
4. Run the workflow.

## Current Implementation

This plugin provides a high-performance backend for diffusion models in ComfyUI. By leveraging SGLang's optimized kernels and parallelization techniques (Tensor Parallelism, TeaCache, etc.), it significantly accelerates the sampling process, especially for large models like FLUX.
