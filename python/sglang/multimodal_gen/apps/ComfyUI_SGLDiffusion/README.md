# ComfyUI SGLDiffusion Plugin

A ComfyUI plugin for integrating with SGLang Diffusion server, supporting image and video generation capabilities.

## Installation

1. Copy this entire directory (`ComfyUI_SGLDiffusion`) to your ComfyUI `custom_nodes/` folder
2. Restart ComfyUI to load the plugin

## Usage

### Prerequisites

Before using this plugin, you need to start the SGLang Diffusion server. The plugin connects to the server via HTTP API calls.

### Basic Workflow

1. **Start SGLang Diffusion Server**: Ensure the SGLang Diffusion server is running and accessible
2. **Connect to Server**: Use the `SGLDiffusion Server Model` node to connect to your server (default: `http://localhost:3000/v1`)
3. **Generate Content**: Use the generation nodes to create images or videos:
   - `SGLDiffusion Generate Image`: For text-to-image and image editing
   - `SGLDiffusion Generate Video`: For text-to-video and image-to-video
4. **Optional**: Use `SGLDiffusion Set LoRA` to load LoRA adapters for style customization, use `SGLDiffusion Unset LoRA` to remove LoRA

### Example Workflows

Reference workflow files are provided in the `workflows/` directory:

- **`sgld_text2img.json`**: Text-to-image generation with LoRA support
- **`sgld_image2video.json`**: Image-to-video generation

To use these workflows:
1. Open ComfyUI
2. Load the workflow JSON file from the `workflows/` directory
3. Adjust the server URL and API key in the `SGLDiffusion Server Model` node if needed
4. Modify prompts and parameters as desired
5. Run the workflow

## Current Implementation

Currently, this plugin uses a server-based approach where it makes HTTP API calls to a running SGLang Diffusion server. This requires:

- The SGLang Diffusion server to be running separately
- Network connectivity between ComfyUI and the server
- Proper server configuration and API key setup

## Future Improvements

Future versions will integrate more tightly with ComfyUI, utilizing ComfyUI's models and most of its nodes, while leveraging SGLD specifically for the computationally intensive sampling process. This will provide a more seamless and efficient workflow.
