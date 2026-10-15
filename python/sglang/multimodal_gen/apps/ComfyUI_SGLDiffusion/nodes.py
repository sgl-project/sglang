"""
ComfyUI nodes for SGLang Diffusion integration.
Provides nodes for connecting to SGLang Diffusion server and generating images/videos.
"""

import base64
import json
import os
import urllib.parse
import uuid

import comfy.model_management
import comfy.utils
import folder_paths
import torch
from comfy_api.input import VideoInput

from .core import SGLDiffusionGenerator, SGLDiffusionServerAPI
from .options import SGLDOptions


def _enable_gguf_in_diffusion_models() -> None:
    """Let SGLDUNETLoader list ``.gguf`` DiTs. Quantization lives in the file."""
    entry = folder_paths.folder_names_and_paths.get("diffusion_models")
    if entry is not None and len(entry) >= 2 and isinstance(entry[1], set):
        entry[1].add(".gguf")


_enable_gguf_in_diffusion_models()
from .utils import (
    convert_b64_to_tensor_image,
    convert_video_to_comfy_video,
    extract_audio,
    get_image_path,
    get_video_path,
    is_empty_image,
    make_image_grid,
    parse_step_list,
    resolve_video_path,
)

# The server clamps n to this range (image_api.py).
MAX_SERVER_IMAGES = 10


def _decode_image_batch(response: dict, num_images: int) -> torch.Tensor:
    """Decode the server's images, in order, into one ComfyUI IMAGE batch."""
    items = [item.get("b64_json") for item in response.get("data") or []]
    if not items or not all(items):
        raise RuntimeError("No image data in response")
    if len(items) < num_images:
        raise RuntimeError(
            f"Requested {num_images} images but the server returned {len(items)}"
        )
    images = [convert_b64_to_tensor_image(b64) for b64 in items[:num_images]]
    shapes = {tuple(image.shape[1:]) for image in images}
    if len(shapes) > 1:
        raise RuntimeError(
            f"Server returned images of different sizes {sorted(shapes)}; "
            "they cannot be combined into one IMAGE batch"
        )
    return torch.cat(images, dim=0)


class SGLDH3DistilledSigmas:
    """The checkpoint's trained rectified-flow grid, independent of scheduler presets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_type": (["fast_h3", "vdn_h3"],)}}

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "build"
    CATEGORY = "SGLDiffusion"

    def build(self, model_type):
        from .executors.minimax_h3 import distilled_h3_sigmas

        return (torch.tensor(distilled_h3_sigmas(model_type)),)


def _comfy_job_hooks() -> dict:
    """Wire a server-mode job into ComfyUI's progress bar and Cancel button."""
    pbar = comfy.utils.ProgressBar(100)
    return {
        "progress_callback": pbar.update_absolute,
        "check_interrupt": comfy.model_management.throw_exception_if_processing_interrupted,
    }


class SGLDOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "model_type": (
                    [
                        "auto-detect",
                        "qwen_image",
                        "qwen_image_edit",
                        "qwen_image21",
                        "flux",
                        "lumina2",
                        "minimax_h3",
                        "fast_h3",
                        "vdn_h3",
                    ],
                    {"default": "auto-detect"},
                ),
                "enable_torch_compile": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "num_gpus": ("INT", {"default": 1, "min": 1, "step": 1}),
                "tp_size": ("INT", {"default": -1, "min": -1, "step": 1}),
                "sp_degree": ("INT", {"default": -1, "min": -1, "step": 1}),
                "ulysses_degree": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "step": 1,
                    },
                ),
                "ring_degree": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "step": 1,
                    },
                ),
                "dp_size": ("INT", {"default": 1, "min": 1, "step": 1}),
                "dp_degree": ("INT", {"default": 1, "min": 1, "step": 1}),
                "enable_cfg_parallel": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "attention_backend": (
                    "STRING",
                    {"default": ""},
                ),
                "dit_layerwise_offload": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "enable_cache_dit": (
                    "BOOLEAN",
                    {"default": False},
                ),
                "quantization": (
                    "STRING",
                    {"default": ""},
                ),
                "runtime_model_path": ("STRING", {"default": ""}),
                "attention_backend_config": ("STRING", {"default": ""}),
                "transformer_weights_path": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": False,
                    },
                ),
            },
        }

    RETURN_TYPES = ("SGLD_OPTIONS",)
    RETURN_NAMES = ("sgld_options",)
    FUNCTION = "create_options"
    CATEGORY = "SGLDiffusion"

    def create_options(
        self,
        model_type: str = "auto-detect",
        enable_torch_compile: bool = False,
        num_gpus: int = 1,
        tp_size: int = -1,
        sp_degree: int = -1,
        ulysses_degree: int = -1,
        ring_degree: int = -1,
        dp_size: int = 1,
        dp_degree: int = 1,
        enable_cfg_parallel: bool = False,
        attention_backend: str = "",
        dit_layerwise_offload: bool = False,
        enable_cache_dit: bool = False,
        quantization: str = "",
        transformer_weights_path: str = "",
        runtime_model_path: str = "",
        attention_backend_config: str = "",
    ):
        """
        Build a dictionary of SGLang Diffusion runtime options.
        """
        # Convert -1 to None for optional parameters (matching ServerArgs defaults)
        ulysses_degree = None if ulysses_degree == -1 else ulysses_degree
        ring_degree = None if ring_degree == -1 else ring_degree
        tp_size = None if tp_size == -1 else tp_size
        sp_degree = None if sp_degree == -1 else sp_degree
        attention_backend = None if attention_backend == "" else attention_backend
        # dp_degree is a leftover alias; ServerArgs only has dp_size.
        if dp_degree not in (None, 1) and dp_size in (None, 1):
            dp_size = dp_degree

        options = {
            "model_type": model_type,
            "enable_torch_compile": enable_torch_compile,
            "num_gpus": num_gpus,
            "tp_size": tp_size,
            "sp_degree": sp_degree,
            "ulysses_degree": ulysses_degree,
            "ring_degree": ring_degree,
            "dp_size": dp_size,
            "enable_cfg_parallel": enable_cfg_parallel,
            "attention_backend": attention_backend,
            "dit_layerwise_offload": dit_layerwise_offload,
        }
        if enable_cache_dit:
            options["enable_cache_dit"] = True
        quantization = (quantization or "").strip()
        if quantization:
            options["quantization"] = quantization
        transformer_weights_path = (transformer_weights_path or "").strip()
        if transformer_weights_path:
            # Same selector as `sglang serve --transformer-weights-path`:
            # local .gguf, owner/repo/path/file.gguf, or owner/repo:QUANT.
            options["transformer_weights_path"] = transformer_weights_path

        for key, value in (
            ("runtime_model_path", runtime_model_path),
            # Same format as --attention-backend-config; ServerArgs parses it.
            ("attention_backend_config", attention_backend_config),
        ):
            if value.strip():
                options[key] = value.strip()

        # Strip None to keep payload clean
        options = {k: v for k, v in options.items() if v is not None}
        return (options,)


class SGLDLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": (
                    "FLOAT",
                    {"default": 1.0, "min": 0, "max": 10, "step": 0.01},
                ),
                "nickname": ("STRING", {"default": ""}),
                "target": (
                    ["all", "transformer", "transformer_2", "critic"],
                    {"default": "all"},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"

    CATEGORY = "SGLDiffusion"

    def load_lora(
        self, model, lora_name, strength_model=1.0, nickname="", target="all"
    ):
        """Load LoRA adapter using SGLang Diffusion API."""
        lora_path = folder_paths.get_full_path("loras", lora_name)
        assert model is not None
        bi = model.clone()
        nickname = nickname if nickname != "" else str("lora" + str(uuid.uuid4()))
        # set lora in the model
        bi.patches[nickname] = (lora_path, strength_model, target)

        # prepare input for the SGLang Diffusion API
        lora_input = {
            "lora_nickname": [],
            "lora_path": [],
            "strength": [],
            "target": [],
        }
        for nickname, lora_info in bi.patches.items():
            lora_input["lora_nickname"].append(nickname)
            lora_input["lora_path"].append(lora_info[0])
            lora_input["strength"].append(lora_info[1])
            lora_input["target"].append(lora_info[2])

        targets = set(lora_input["target"])
        if "all" in targets and len(targets) > 1:
            # set_lora groups adapters per target string; "all" overlapping a
            # named target makes the worker keep only one group on that module.
            raise ValueError(
                "Chained SGLDLoraLoader nodes mix target 'all' with "
                f"{sorted(targets - {'all'})}; use the same target on every node"
            )
        # The shared worker is bound to the selected MODEL at sampling time.
        bi.model_options["sgld_lora_input"] = lora_input
        return (bi,)


class SGLDUNETLoader:
    def __init__(self):
        self.generator = SGLDiffusionGenerator.shared()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
            },
            "optional": {
                "sgld_options": ("SGLD_OPTIONS",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "SGLDiffusion"

    def load_unet(self, unet_name, weight_dtype, sgld_options: dict = None):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)

        model = self.generator.load_model(
            unet_path, model_options=model_options, sgld_options=sgld_options
        )
        return (model,)


# Dropdowns are built in INPUT_TYPES, before any client exists; they list the
# adapters of the server the user most recently connected to.
_last_client = None
LORA_NONE = "none"
LORA_SLOTS = 4


def _loaded_lora_choices() -> list:
    if _last_client is None:
        return [LORA_NONE]
    try:
        listing = _last_client.list_loras(timeout=3)
    except Exception:
        return [LORA_NONE]
    return [LORA_NONE] + [a["nickname"] for a in listing.get("loaded_adapters", [])]


class SGLDNativeModelLoader:
    """Native SGLang model directories (FastH3, VDN-H3) that have no ComfyUI single file."""

    def __init__(self):
        self.generator = SGLDiffusionGenerator.shared()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model_dir": ("STRING", {"default": ""})},
            "optional": {"sgld_options": ("SGLD_OPTIONS",)},
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "SGLDiffusion"

    def load(self, model_dir, sgld_options: dict = None):
        model_dir = os.path.abspath(os.path.expanduser(model_dir.strip()))
        if not os.path.isfile(os.path.join(model_dir, "model_index.json")):
            raise ValueError(f"{model_dir} is not a native model directory")
        return (self.generator.load_model(model_dir, sgld_options=sgld_options),)


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
            },
            "optional": {
                "image_timeout": (
                    "INT",
                    {
                        "default": 300,
                        "min": 1,
                        "max": 3600,
                        "tooltip": "Seconds to wait for an image "
                        "generate/edit request (e.g. a large job, or "
                        "torch.compile warmup on the first request, can "
                        "need more than 300s)",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SGLD_CLIENT", "STRING")
    RETURN_NAMES = ("sgld_client", "model_info")
    FUNCTION = "load_server"
    CATEGORY = "SGLDiffusion"

    def load_server(self, base_url: str, api_key: str, image_timeout: int = 300):
        """Initialize OpenAI client for SGLang Diffusion server."""
        global _last_client
        client = SGLDiffusionServerAPI(
            base_url=base_url, api_key=api_key, image_timeout=image_timeout
        )
        _last_client = client
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
                "positive_prompt": (
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
                "enhance_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Rewrite the prompt on the server (needs "
                        "--prompt-enhancer-config); the result is returned "
                        "as revised_prompt",
                    },
                ),
                "num_images": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": MAX_SERVER_IMAGES,
                        "tooltip": "Images to generate in one request, returned as "
                        "one IMAGE batch. Image i uses seed + i.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "revised_prompt")
    FUNCTION = "generate_image"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    def generate_image(
        self,
        sgld_client: SGLDiffusionServerAPI,
        positive_prompt: str,
        negative_prompt: str = "",
        image: torch.Tensor = None,
        seed: int = 1024,
        steps: int = 6,
        cfg: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        enable_teacache: bool = False,
        enhance_prompt: bool = False,
        num_images: int = 1,
    ):
        """Generate image using SGLang Diffusion API."""
        if not positive_prompt:
            raise ValueError("Prompt cannot be empty")
        if not 1 <= num_images <= MAX_SERVER_IMAGES:
            raise ValueError(
                f"num_images must be between 1 and {MAX_SERVER_IMAGES}, got {num_images}"
            )

        size = f"{width}x{height}"

        # Prepare request parameters
        request_params = {
            "prompt": positive_prompt,
            "size": size,
            "n": num_images,
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
        if enhance_prompt:
            request_params["enhance_prompt"] = True
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

        image = _decode_image_batch(response, num_images)
        revised_prompt = response["data"][0].get("revised_prompt") or positive_prompt

        return (image, revised_prompt)


class SGLDiffusionCheckpointSweep:
    """Render one prompt and seed across training checkpoints and tile the results.

    Each checkpoint is loaded into the running server with
    /update_weights_from_disk, so no restart is needed between steps. The
    server's original weights are restored afterwards.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sgld_client": ("SGLD_CLIENT",),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "checkpoint_template": (
                    "STRING",
                    {
                        "default": "/path/to/run/checkpoint-{step}",
                        "tooltip": "Directory of one checkpoint, readable by the "
                        "server and laid out like the served model. {step} is "
                        "replaced by each step below",
                    },
                ),
                "checkpoint_steps": (
                    "STRING",
                    {
                        "default": "500, 1000, 1500, 2000, 2500, "
                        "3000, 3500, 4000, 4500, 5000",
                        "tooltip": "Comma-separated training steps, laid out "
                        "left to right, top to bottom",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 0,
                        "max": 2**32 - 1,
                        "tooltip": "Fixed so only the weights differ between cells",
                    },
                ),
            },
            "optional": {
                "negative_prompt": ("STRING", {"default": ""}),
                "sampling_steps": ("INT", {"default": 6, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "columns": ("INT", {"default": 5, "min": 1, "max": 16}),
                "restore_original": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Reload the weights the server started with "
                        "when the sweep ends, even if it fails",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("grid", "images", "log")
    FUNCTION = "sweep"
    CATEGORY = "SGLDiffusion"

    def sweep(
        self,
        sgld_client: SGLDiffusionServerAPI,
        positive_prompt: str,
        checkpoint_template: str,
        checkpoint_steps: str,
        seed: int,
        negative_prompt: str = "",
        sampling_steps: int = 6,
        cfg: float = 7.0,
        width: int = 1024,
        height: int = 1024,
        columns: int = 5,
        restore_original: bool = True,
    ):
        if not positive_prompt:
            raise ValueError("Prompt cannot be empty")
        if "{step}" not in checkpoint_template:
            raise ValueError("checkpoint_template must contain {step}")
        if seed < 0:
            raise ValueError(
                "seed must be >= 0; a random seed makes cells incomparable"
            )
        steps = parse_step_list(checkpoint_steps)

        # Read before the first swap: the worker rewrites its model_path on each update.
        original_path = sgld_client.get_model_path() if restore_original else None
        request = self._request(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            steps=sampling_steps,
            cfg=cfg,
            width=width,
            height=height,
        )
        images, log = [], []
        try:
            for step in steps:
                path = checkpoint_template.replace("{step}", str(step))
                try:
                    sgld_client.update_weights_from_disk(path)
                except Exception as e:
                    raise RuntimeError(f"Checkpoint step {step} ({path}): {e}")
                images.append(self._render(sgld_client, step=step, request=request))
                log.append(f"step {step}: {path}")
        finally:
            if original_path is not None:
                try:
                    sgld_client.update_weights_from_disk(original_path)
                    log.append(f"restored: {original_path}")
                except Exception as e:
                    log.append(f"WARNING: could not restore {original_path}: {e}")

        batch = torch.cat(images, dim=0)
        return (make_image_grid(batch, columns), batch, "\n".join(log))

    @staticmethod
    def _request(*, prompt, negative_prompt, seed, steps, cfg, width, height):
        request = {
            "prompt": prompt,
            "size": f"{width}x{height}",
            "response_format": "b64_json",
            "guidance_scale": cfg,
            "num_inference_steps": steps,
            "seed": seed,
        }
        if negative_prompt:
            request["negative_prompt"] = negative_prompt
        return request

    @staticmethod
    def _render(sgld_client, *, step, request):
        try:
            response = sgld_client.generate_image(**request)
        except Exception as e:
            raise RuntimeError(f"Failed to generate at step {step}: {e}")
        if not response["data"] or not response["data"][0]["b64_json"]:
            raise RuntimeError(f"No image data in response at step {step}")
        return convert_b64_to_tensor_image(response["data"][0]["b64_json"])


class SGLDiffusionGenerateVideo:
    """Node to generate videos using SGLang Diffusion."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sgld_client": ("SGLD_CLIENT",),
                "positive_prompt": (
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
                "video": (
                    "VIDEO",
                    {
                        "tooltip": "source clip for video-to-video "
                        "(restyle or extend an existing video)",
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
                "enhance_prompt": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Rewrite the prompt on the server (needs "
                        "--prompt-enhancer-config); the result is returned "
                        "as revised_prompt",
                    },
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING", "STRING")
    RETURN_NAMES = ("video", "video_path", "revised_prompt")
    FUNCTION = "generate_video"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    def generate_video(self, **kwargs):
        """Generate video using SGLang Diffusion API."""
        return self._generate(**kwargs)

    def _generate(
        self,
        sgld_client: SGLDiffusionServerAPI,
        positive_prompt: str,
        negative_prompt: str = "",
        image: torch.Tensor = None,
        video: VideoInput = None,
        seed: int = 1024,
        steps: int = 6,
        cfg: float = 7.0,
        width: int = 1280,
        height: int = 720,
        num_frames: int = 120,
        fps: int = 24,
        seconds: int = 5,
        enable_teacache: bool = False,
        enhance_prompt: bool = False,
        extra_fields: dict = None,
    ):
        if not positive_prompt:
            raise ValueError("Prompt cannot be empty")

        size = f"{width}x{height}"
        output_dir = folder_paths.get_temp_directory()

        # Prepare request parameters
        request_params = {
            "prompt": positive_prompt,
            "size": size,
            "seconds": seconds,
            "fps": fps,
            "output_path": output_dir,
        }
        if extra_fields:
            request_params["extra_fields"] = extra_fields

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
        if enhance_prompt:
            request_params["enhance_prompt"] = True
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
        if video is not None:
            request_params["video_path"] = get_video_path(video)

        # Call API
        try:
            response = sgld_client.generate_video(
                **request_params, **_comfy_job_hooks()
            )
            video_path = resolve_video_path(
                response.get("file_path"), response.get("url")
            )
            # The server may have resolved a different size than requested
            # (e.g. to match an image-to-video input's aspect ratio).
            resolved_size = response.get("size", "")
            if resolved_size:
                width, height = (int(v) for v in resolved_size.split("x"))
            video = convert_video_to_comfy_video(video_path, height, width)
        except comfy.model_management.InterruptProcessingException:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to generate video: {str(e)}")

        return video, video_path, response.get("revised_prompt") or positive_prompt


class SGLDiffusionGenerateAudioVideo(SGLDiffusionGenerateVideo):
    """Generate a video and hand back its audio track as a ComfyUI AUDIO.

    One node for every server model that muxes audio into its MP4 (MiniMax-H3,
    LTX-2, MOVA, Kandinsky-6, ...). Model-specific request fields go through
    `extra_request_json`, so a new model needs no new node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["optional"]["extra_request_json"] = (
            "STRING",
            {
                "default": "",
                "multiline": True,
                "tooltip": "JSON object merged into the /v1/videos request, "
                'e.g. {"flow_shift": 12.0}. Overrides the fields above.',
            },
        )
        return types

    RETURN_TYPES = ("VIDEO", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("video", "audio", "video_path", "revised_prompt")
    FUNCTION = "generate_audio_video"

    def generate_audio_video(self, extra_request_json: str = "", **kwargs):
        extra_fields = None
        if extra_request_json.strip():
            try:
                extra_fields = json.loads(extra_request_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"extra_request_json is not valid JSON: {e}")
            if not isinstance(extra_fields, dict):
                raise ValueError("extra_request_json must be a JSON object")

        video, video_path, revised_prompt = self._generate(
            extra_fields=extra_fields, **kwargs
        )
        return (video, extract_audio(video_path), video_path, revised_prompt)


class SGLDiffusionGenerateH3:
    """Node to generate joint video and audio with MiniMax-H3.

    H3 denoises a packed video+audio sequence in one pass and routes its
    conditioning by task rather than by a single reference slot, so it needs
    its own request shape (`task` / `conditions` / `target`) that the generic
    video node does not model. The returned MP4 carries both streams.
    """

    TASKS = ["t2va", "fl2va", "ref2va"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sgld_client": ("SGLD_CLIENT",),
                "positive_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Text prompt. Reference material is addressed "
                        "positionally as <Picture 1>, <Video 1>, <Audio 1>",
                    },
                ),
                "task": (
                    cls.TASKS,
                    {
                        "default": "t2va",
                        "tooltip": "t2va: text only. fl2va: first/last keyframes. "
                        "ref2va: image, video, and audio references",
                    },
                ),
            },
            "optional": {
                "first_frame": (
                    "IMAGE",
                    {"tooltip": "fl2va: becomes the clip's first frame"},
                ),
                "last_frame": (
                    "IMAGE",
                    {"tooltip": "fl2va: becomes the clip's last frame"},
                ),
                "reference_image": (
                    "IMAGE",
                    {
                        "tooltip": "ref2va: guides identity and style; not "
                        "preserved as an endpoint frame"
                    },
                ),
                "reference_video": (
                    "STRING",
                    {"default": "", "tooltip": "ref2va: path or URL to a video"},
                ),
                "reference_audio": (
                    "STRING",
                    {"default": "", "tooltip": "ref2va: path or URL to audio"},
                ),
                "negative_prompt": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 1101, "min": -1, "max": 2**32 - 1}),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "short_edge": ("INT", {"default": 768, "min": 256, "max": 1536}),
                "aspect_ratio": (
                    ["16:9", "9:16", "1:1", "auto"],
                    {"default": "16:9"},
                ),
                "duration_seconds": (
                    "FLOAT",
                    {"default": 5.0, "min": 4.0, "max": 15.0, "step": 0.5},
                ),
                "flow_shift": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 30.0}),
                "audio_flow_shift": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 30.0},
                ),
            },
        }

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "video_path")
    FUNCTION = "generate"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    @staticmethod
    def _image_material_uri(image: torch.Tensor) -> str:
        """Embed the image inline as a base64 data URI.

        A `file://` path only works when the SGLD server runs on this same
        machine; the server's H3 material loader accepts `data:` URIs
        directly (no upload endpoint needed), so this works locally and
        remotely alike.
        """
        path = get_image_path(image)
        try:
            with open(path, "rb") as fh:
                encoded = base64.b64encode(fh.read()).decode("ascii")
        finally:
            os.remove(path)
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _is_local_server(base_url: str) -> bool:
        host = urllib.parse.urlparse(base_url).hostname or ""
        return host in ("localhost", "127.0.0.1", "::1")

    @classmethod
    def _remote_material_uri(cls, value: str, kind: str, base_url: str) -> str:
        """Remote URLs pass through. A local path only works when the SGLD
        server is also on this machine: H3's server-side `/v1/videos`
        endpoint has no multipart slot for the video/audio conditions this
        node sends through `extra_fields.conditions`, unlike the image
        conditions above (which sidestep that by embedding bytes inline).
        Raise clearly instead of building a `file://` URI that will
        silently fail against any server that isn't local.
        """
        if value.startswith(("http://", "https://")):
            return value
        if cls._is_local_server(base_url):
            return f"file://{os.path.abspath(value)}"
        raise ValueError(
            f"{kind} must be an http(s) URL when the SGLD server "
            f"({base_url}) isn't on this machine; got a local path "
            f"({value!r}), which that server has no way to read or upload."
        )

    def generate(
        self,
        sgld_client: SGLDiffusionServerAPI,
        positive_prompt: str,
        task: str,
        first_frame: torch.Tensor = None,
        last_frame: torch.Tensor = None,
        reference_image: torch.Tensor = None,
        reference_video: str = "",
        reference_audio: str = "",
        negative_prompt: str = "",
        seed: int = 1101,
        steps: int = 50,
        short_edge: int = 768,
        aspect_ratio: str = "16:9",
        duration_seconds: float = 5.0,
        flow_shift: float = 12.0,
        audio_flow_shift: float = 3.0,
    ):
        """Build H3's task-shaped request and submit it through the server API."""
        if not positive_prompt:
            raise ValueError("Prompt cannot be empty")

        # 1. keyframes carry a frame_index and are preserved as endpoints;
        #    references are semantic material and keep request order, because
        #    the prompt addresses them positionally per modality
        conditions = []
        if first_frame is not None:
            conditions.append(
                {
                    "type": "image",
                    "uri": self._image_material_uri(first_frame),
                    "role": "keyframe",
                    "frame_index": 0,
                }
            )
        if last_frame is not None:
            conditions.append(
                {
                    "type": "image",
                    "uri": self._image_material_uri(last_frame),
                    "role": "keyframe",
                    "frame_index": -1,
                }
            )
        if reference_image is not None:
            conditions.append(
                {
                    "type": "image",
                    "uri": self._image_material_uri(reference_image),
                    "role": "reference",
                }
            )
        if reference_video:
            conditions.append(
                {
                    "type": "video",
                    "uri": self._remote_material_uri(
                        reference_video, "reference_video", sgld_client.base_url
                    ),
                    "role": "reference",
                }
            )
        if reference_audio:
            conditions.append(
                {
                    "type": "audio",
                    "uri": self._remote_material_uri(
                        reference_audio, "reference_audio", sgld_client.base_url
                    ),
                    "role": "reference",
                }
            )

        # 2. reject wiring the server would reject anyway, but name the input
        #    the user has to change
        if task == "fl2va" and not (first_frame is not None or last_frame is not None):
            raise ValueError("fl2va requires first_frame, last_frame, or both")
        if task == "ref2va" and not conditions:
            raise ValueError(
                "ref2va requires at least one of reference_image, "
                "reference_video, or reference_audio"
            )
        if task == "t2va" and conditions:
            raise ValueError("t2va takes no conditioning inputs; pick another task")

        # 3. `target` resolves the aligned canvas and frame count; the `size`
        #    the server API always sends is unused by H3
        extra_fields = {
            "task": task,
            "conditions": conditions,
            "target": {
                "short_edge": short_edge,
                "aspect_ratio": aspect_ratio,
                "duration_seconds": duration_seconds,
            },
            "flow_shift": flow_shift,
            "audio_flow_shift": audio_flow_shift,
        }

        request_params = {
            "prompt": positive_prompt,
            "seconds": int(duration_seconds),
            "num_inference_steps": steps,
            "output_path": folder_paths.get_temp_directory(),
            "extra_fields": extra_fields,
        }
        if negative_prompt:
            request_params["negative_prompt"] = negative_prompt
        if seed >= 0:
            request_params["seed"] = seed

        try:
            response = sgld_client.generate_video(
                **request_params, **_comfy_job_hooks()
            )
            video_path = resolve_video_path(
                response.get("file_path"), response.get("url")
            )
        except comfy.model_management.InterruptProcessingException:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to generate MiniMax-H3 video: {str(e)}")

        # H3 aligns the canvas server-side, so the resolved size is only known
        # from the response; short_edge and aspect_ratio cannot reconstruct it
        resolved_size = response.get("size", "")
        if resolved_size:
            width, height = (int(v) for v in resolved_size.split("x"))
        else:
            width = height = short_edge
        video = convert_video_to_comfy_video(video_path, height, width)

        return (video, video_path)


class SGLDiffusionServerSetLora:
    """Node to set LoRA adapter for SGLang Diffusion server."""

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
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": "LoRA merge strength; the server's own "
                        "default is 1.0",
                    },
                ),
                "merge_mode": (
                    ["server default", "auto", "merge", "dynamic"],
                    {"default": "server default"},
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
        strength: float = 1.0,
        merge_mode: str = "server default",
    ):
        """Set LoRA adapter using SGLang Diffusion API."""
        if lora_nickname == "":
            lora_nickname = os.path.splitext(lora_name)[0]

        # Prepare request parameters
        request_params = {
            "lora_nickname": lora_nickname,
            "lora_path": lora_name,
            "target": target,
            "strength": strength,
            "merge_mode": None if merge_mode == "server default" else merge_mode,
        }

        # Call API
        try:
            sgld_client.set_lora(**request_params)
            return (sgld_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to set LoRA adapter: {str(e)}")


class SGLDiffusionServerListLoras:
    """Node to browse the LoRA adapters loaded on the SGLang Diffusion server."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sgld_client": ("SGLD_CLIENT",)}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("nicknames", "status")
    FUNCTION = "list_loras"
    CATEGORY = "SGLDiffusion"

    def list_loras(self, sgld_client: SGLDiffusionServerAPI):
        listing = sgld_client.list_loras()
        nicknames = [a["nickname"] for a in listing.get("loaded_adapters", [])]
        return ("\n".join(nicknames), json.dumps(listing, indent=2))


class SGLDiffusionServerLoraStack:
    """Node to apply several LoRA adapters at once, each with its own strength.

    A slot's dropdown lists adapters already loaded on the server (refresh after
    connecting). To load a new one, leave the dropdown on "none" and give its
    path or HF repo id in the slot's path field.
    """

    MERGE_MODES = ["merge", "auto", "dynamic", "server default"]

    @classmethod
    def INPUT_TYPES(cls):
        choices = _loaded_lora_choices()
        optional = {}
        for i in range(1, LORA_SLOTS + 1):
            optional[f"lora_{i}"] = (choices, {"default": LORA_NONE})
            optional[f"lora_path_{i}"] = (
                "STRING",
                {
                    "default": "",
                    "tooltip": "Path or HF repo id; only needed to load an "
                    "adapter the server does not have yet",
                },
            )
            optional[f"strength_{i}"] = (
                "FLOAT",
                {"default": 1.0, "min": 0, "max": 10, "step": 0.01},
            )
        optional["merge_mode"] = (
            cls.MERGE_MODES,
            {
                "default": "merge",
                "tooltip": "Stacking several LoRAs on one target needs merge",
            },
        )
        optional["target"] = (
            ["all", "transformer", "transformer_2", "critic"],
            {"default": "all"},
        )
        return {"required": {"sgld_client": ("SGLD_CLIENT",)}, "optional": optional}

    @classmethod
    def VALIDATE_INPUTS(cls, lora_1=None, lora_2=None, lora_3=None, lora_4=None):
        # The dropdown can be stale or empty; the server is the authority.
        return True

    RETURN_TYPES = ("SGLD_CLIENT", "STRING")
    RETURN_NAMES = ("sgld_client", "summary")
    FUNCTION = "set_lora_stack"
    CATEGORY = "SGLDiffusion"

    def set_lora_stack(
        self,
        sgld_client: SGLDiffusionServerAPI,
        merge_mode: str = "merge",
        target: str = "all",
        **slots,
    ):
        adapters = self._collect_adapters(slots)
        nicknames = [nickname for nickname, _, _ in adapters]
        if len(set(nicknames)) != len(nicknames):
            raise ValueError(f"The same LoRA is selected twice: {nicknames}")
        if len(adapters) > 1 and merge_mode == "dynamic":
            raise ValueError("Stacking several LoRAs needs merge_mode merge or auto")

        try:
            sgld_client.set_lora(
                lora_nickname=nicknames,
                lora_path=[path for _, path, _ in adapters],
                target=target,
                strength=[strength for _, _, strength in adapters],
                merge_mode=None if merge_mode == "server default" else merge_mode,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to set LoRA stack: {str(e)}")

        summary = ", ".join(f"{name}@{strength:g}" for name, _, strength in adapters)
        return (sgld_client, summary)

    @staticmethod
    def _collect_adapters(slots: dict) -> list:
        adapters = []
        for i in range(1, LORA_SLOTS + 1):
            name = slots.get(f"lora_{i}", LORA_NONE)
            path = slots.get(f"lora_path_{i}", "").strip()
            if name == LORA_NONE and not path:
                continue
            if name == LORA_NONE:
                name = os.path.splitext(os.path.basename(path.rstrip("/")))[0]
            adapters.append((name, path or None, slots.get(f"strength_{i}", 1.0)))
        if not adapters:
            raise ValueError("Pick at least one LoRA")
        return adapters


class SGLDiffusionServerUnsetLora:
    """Node to unset LoRA adapter for SGLang Diffusion server."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sgld_client": ("SGLD_CLIENT",),
            },
            "optional": {
                "target": (
                    [
                        "all",
                        "transformer",
                        "transformer_2",
                        "critic",
                    ],
                    {
                        "default": "all",
                        "tooltip": "Which transformer(s) to unset the LoRA from",
                    },
                ),
            },
        }

    RETURN_TYPES = ("SGLD_CLIENT",)
    RETURN_NAMES = ("sgld_client",)
    FUNCTION = "unset_lora"
    CATEGORY = "SGLDiffusion"
    OUTPUT_NODE = False

    def unset_lora(
        self,
        sgld_client: SGLDiffusionServerAPI,
        target: str = "all",
    ):
        """Unset LoRA adapter using SGLang Diffusion API."""
        try:
            sgld_client.unset_lora(target=target)
            return (sgld_client,)
        except Exception as e:
            raise RuntimeError(f"Failed to unset LoRA adapter: {str(e)}")


# Register nodes
NODE_CLASS_MAPPINGS = {
    "SGLDH3DistilledSigmas": SGLDH3DistilledSigmas,
    "SGLDiffusionServerModel": SGLDiffusionServerModel,
    "SGLDiffusionGenerateImage": SGLDiffusionGenerateImage,
    "SGLDiffusionCheckpointSweep": SGLDiffusionCheckpointSweep,
    "SGLDiffusionGenerateVideo": SGLDiffusionGenerateVideo,
    "SGLDiffusionGenerateAudioVideo": SGLDiffusionGenerateAudioVideo,
    "SGLDiffusionGenerateH3": SGLDiffusionGenerateH3,
    "SGLDiffusionServerSetLora": SGLDiffusionServerSetLora,
    "SGLDiffusionServerListLoras": SGLDiffusionServerListLoras,
    "SGLDiffusionServerLoraStack": SGLDiffusionServerLoraStack,
    "SGLDiffusionServerUnsetLora": SGLDiffusionServerUnsetLora,
    "SGLDUNETLoader": SGLDUNETLoader,
    "SGLDNativeModelLoader": SGLDNativeModelLoader,
    "SGLDOptions": SGLDOptions,
    "SGLDLoraLoader": SGLDLoraLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SGLDiffusionServerModel": "SGLDiffusion Server Model",
    "SGLDiffusionGenerateImage": "SGLDiffusion Generate Image",
    "SGLDiffusionCheckpointSweep": "SGLDiffusion Checkpoint Sweep",
    "SGLDiffusionGenerateVideo": "SGLDiffusion Generate Video",
    "SGLDiffusionGenerateAudioVideo": "SGLDiffusion Generate Audio-Video",
    "SGLDiffusionGenerateH3": "SGLDiffusion Generate MiniMax-H3",
    "SGLDiffusionServerSetLora": "SGLDiffusion Server Set LoRA",
    "SGLDiffusionServerListLoras": "SGLDiffusion Server List LoRAs",
    "SGLDiffusionServerLoraStack": "SGLDiffusion Server LoRA Stack",
    "SGLDiffusionServerUnsetLora": "SGLDiffusion Server Unset LoRA",
    "SGLDUNETLoader": "SGLDiffusion UNET Loader",
    "SGLDNativeModelLoader": "SGLDiffusion Native Model Loader",
    "SGLDOptions": "SGLDiffusion Options",
    "SGLDLoraLoader": "SGLDiffusion LoRA Loader",
}
