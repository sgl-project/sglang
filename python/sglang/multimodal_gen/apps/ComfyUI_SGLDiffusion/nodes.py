"""
ComfyUI nodes for SGLang Diffusion integration.
Provides nodes for connecting to SGLang Diffusion server and generating images/videos.
"""

import base64
import io
import os
import time
import torch

from .server_api import SGLDiffusionServerAPI
from .utils import get_image_path, is_empty_image, convert_b64_to_tensor_image, convert_video_to_comfy_video
import folder_paths


class SGLDiffusionServerModel:
    """Node to load and manage SGLang Diffusion server connection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": (
                    "STRING",
                    {
                        "default": "http://localhost:3000/v1",
                        "multiline": False,
                    },
                ),
                "api_key": (
                    "STRING",
                    {
                        "default": "sk-proj-1234567890",
                        "multiline": False,
                    },
                ),
            }
        }

    RETURN_TYPES = ("SGLD_CLIENT", "STRING")
    RETURN_NAMES = ("sgld_client", "model_info")
    FUNCTION = "load_server"
    CATEGORY = "SGLDiffusion"

    def load_server(self, base_url: str, api_key: str):
        """Initialize OpenAI client for SGLang Diffusion server."""
        client = SGLDiffusionServerAPI(base_url=base_url, api_key=api_key)
        try:
            model_info = client.get_model_info()
            # Format model_info as a readable string
            info_lines = ["=== SGLDiffusion Model Info ==="]
            for key, value in model_info.items():
                info_lines.append(f"{key}: {value}")
            model_info_str = "\n".join(info_lines)
        except Exception as e:
            model_info_str = f"Failed to get model info: {str(e)}"
        return (client, model_info_str)


class SGLDiffusionGenerateImage:
    """Node to generate images using SGLang Diffusion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sgld_client": ("SGLD_CLIENT",),
                "positve_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Text prompt for image generation",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Negative prompt to avoid certain elements",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": "input image to use for editing",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1024,
                        "min": -1,
                        "max": 2**32 - 1,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 256,
                        "max": 4096,
                        "step": 64,
                    },
                ),
                "enable_teacache": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    def generate_image(
        self,
        sgld_client: SGLDiffusionServerAPI,
        positve_prompt: str,
        negative_prompt: str = "",
        image: torch.Tensor = None,
        seed: int = 1024,
        steps: int = 6,
        cfg: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        enable_teacache: bool = False,
    ):
        """Generate image using SGLang Diffusion API."""
        if not positve_prompt:
            raise ValueError("Prompt cannot be empty")

        size = f"{width}x{height}"

        # Prepare request parameters
        request_params = {
            "prompt": positve_prompt,
            "size": size,
            "response_format": "b64_json",
        }

        # Add optional parameters if provided
        if negative_prompt:
            request_params["negative_prompt"] = negative_prompt
        if cfg is not None:
            request_params["guidance_scale"] = cfg
        if steps is not None:
            request_params["num_inference_steps"] = steps
        if seed is not None and seed >= 0:
            request_params["seed"] = seed
        if enable_teacache:
            request_params["enable_teacache"] = True
        if image is not None:
            # If the image is empty, use the size of the image to generate the image
            if is_empty_image(image):
                width, height = image.shape[2], image.shape[1]
                size = f"{width}x{height}"
                request_params["size"] = size
            else:
                request_params["image_path"] = get_image_path(image)

        # Call API
        try:
            response = sgld_client.generate_image(**request_params)
        except Exception as e:
            raise RuntimeError(f"Failed to generate image: {str(e)}")

        # Decode base64 image
        if not response['data'] or not response['data'][0]['b64_json']:
            raise RuntimeError("No image data in response")
        image_data = response['data'][0]['b64_json']
        image = convert_b64_to_tensor_image(image_data)

        return (image,)


class SGLDiffusionGenerateVideo:
    """Node to generate videos using SGLang Diffusion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sgld_client": ("SGLD_CLIENT",),
                "positve_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Text prompt for video generation",
                    },
                ),
            },
            "optional": {
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Negative prompt to avoid certain elements",
                    },
                ),
                "image": (
                    "IMAGE",
                    {
                        "default": None,
                        "tooltip": "input image to use for image-to-video",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1024,
                        "min": -1,
                        "max": 2**32 - 1,
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 6,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 1.0,
                        "max": 20.0,
                        "step": 0.1,
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 256,
                        "max": 4096,
                        "step": 1,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 720,
                        "min": 256,
                        "max": 4096,
                        "step": 1,
                    },
                ),
                "num_frames": (
                    "INT",
                    {
                        "default": 120,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                    },
                ),
                "fps": (
                    "INT",
                    {
                        "default": 24,
                        "min": 1,
                        "max": 60,
                        "step": 1,
                    },
                ),
                "seconds": (
                    "INT",
                    {
                        "default": 5,
                        "min": 1,
                        "max": 60,
                        "step": 1,
                    },
                ),
                "enable_teacache": (
                    "BOOLEAN",
                    {
                        "default": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "generate_video"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    def generate_video(
        self,
        sgld_client: SGLDiffusionServerAPI,
        positve_prompt: str,
        negative_prompt: str = "",
        image: torch.Tensor = None,
        seed: int = 1024,
        steps: int = 6,
        cfg: float = 7.0,
        width: int = 1280,
        height: int = 720,
        num_frames: int = 120,
        fps: int = 24,
        seconds: int = 5,
        enable_teacache: bool = False,
    ):
        """Generate video using SGLang Diffusion API."""
        if not positve_prompt:
            raise ValueError("Prompt cannot be empty")

        size = f"{width}x{height}"
        output_dir = folder_paths.get_temp_directory()

        # Prepare request parameters
        request_params = {
            "prompt": positve_prompt,
            "size": size,
            "seconds": seconds,
            "fps": fps,
            "output_path": output_dir,
        }

        # Add optional parameters if provided
        if negative_prompt:
            request_params["negative_prompt"] = negative_prompt
        if cfg is not None:
            request_params["guidance_scale"] = cfg
        if steps is not None:
            request_params["num_inference_steps"] = steps
        if seed is not None and seed >= 0:
            request_params["seed"] = seed
        if enable_teacache:
            request_params["enable_teacache"] = True
        if num_frames is not None:
            request_params["num_frames"] = num_frames
        if image is not None:
            # If the image is empty, use the size of the image to generate the video
            if is_empty_image(image):
                width, height = image.shape[2], image.shape[1]
                size = f"{width}x{height}"
                request_params["size"] = size
            else:
                request_params["input_reference"] = get_image_path(image)

        # Call API
        try:
            response = sgld_client.generate_video(**request_params)
            video_path = response.get("file_path", "")
            video = convert_video_to_comfy_video(video_path, height, width)
        except Exception as e:
            raise RuntimeError(f"Failed to generate video: {str(e)}")

        return (video, video_path)


class SGLDiffusionSetLora:
    """Node to set LoRA adapter for SGLang Diffusion server."""
    def __init__(self):
        self.target = "all"
        self.sgld_client = None

    def __del__(self):
        if self.sgld_client:
            self.sgld_client.unset_lora(target=self.target)
            self.sgld_client = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sgld_client": ("SGLD_CLIENT",),
                "lora_name": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The name of the LoRA adapter",
                    },
                ),
            },
            "optional": {
                "lora_nickname": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "The nickname of the LoRA adapter",
                    },
                ),
                "target": (
                    [
                        "all",
                        "transformer",
                        "transformer_2",
                        "critic",
                    ],
                    {
                        "default": "all",
                        "tooltip": "Which transformer(s) to apply the LoRA to",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SGLD_CLIENT",)
    RETURN_NAMES = ("sgld_client",)
    FUNCTION = "set_lora"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    def set_lora(
        self,
        sgld_client: SGLDiffusionServerAPI,
        lora_name: str = "",
        lora_nickname: str = "",
        target: str = "all",
    ):
        """Set LoRA adapter using SGLang Diffusion API."""
        if lora_nickname == "":
            lora_nickname = lora_name[:-4]

        self.sgld_client = sgld_client
        self.target = target

        # Prepare request parameters
        request_params = {
            "lora_nickname": lora_nickname,
            "lora_path": lora_name,
            "target": target,
        }

        # Call API
        try:
            response = self.sgld_client.set_lora(**request_params)
            return (self.sgld_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to set LoRA adapter: {str(e)}")


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SGLDiffusionServerModel": SGLDiffusionServerModel,
    "SGLDiffusionGenerateImage": SGLDiffusionGenerateImage,
    "SGLDiffusionGenerateVideo": SGLDiffusionGenerateVideo,
    "SGLDiffusionSetLora": SGLDiffusionSetLora,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SGLDiffusionServerModel": "SGLDiffusion Server Model",
    "SGLDiffusionGenerateImage": "SGLDiffusion Generate Image",
    "SGLDiffusionGenerateVideo": "SGLDiffusion Generate Video",
    "SGLDiffusionSetLora": "SGLDiffusion Set LoRA",
}

