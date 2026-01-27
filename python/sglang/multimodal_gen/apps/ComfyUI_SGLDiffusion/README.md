# ComfyUI SGLDiffusion Plugin

A ComfyUI plugin for integrating with SGLang Diffusion server, supporting image and video generation capabilities.

## Installation

1. **Install SGLang**: Follow the [Installation Guide](../../docs/install.md) to install `sglang[diffusion]`.
2. **Install Plugin**: Copy this entire directory (`ComfyUI_SGLDiffusion`) to your ComfyUI `custom_nodes/` folder.
3. **Restart ComfyUI**: Restart ComfyUI to load the plugin.

## Usage

The plugin supports two modes of operation: **Server Mode** (via HTTP API) and **Integrated Mode** (tight integration with ComfyUI).

### Supported Models
- **Z-Image**: High-speed image generation models (e.g., `Z-Image-Turbo`)
- **FLUX**: State-of-the-art text-to-image models (e.g., `FLUX.1-dev`)
- **Qwen-Image**: Multi-modal image generation models (e.g., `Qwen-Image`,`Qwen-Image-2512`). *Note: Image editing support is currently experimental and may have some issues.*

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
