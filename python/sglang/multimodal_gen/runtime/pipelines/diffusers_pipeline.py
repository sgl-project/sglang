# SPDX-License-Identifier: Apache-2.0
"""
Diffusers backend pipeline wrapper.

This module provides a wrapper that allows running any diffusers-supported model
through sglang's infrastructure using vanilla diffusers pipelines.
"""

import argparse
import inspect
import re
import warnings
from io import BytesIO
from typing import Any

import numpy as np
import requests
import torch
import torchvision.transforms as T
from diffusers import DiffusionPipeline
from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.pipeline_executor import (
    PipelineExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.executors.sync_executor import (
    SyncExecutor,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import PipelineStage
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class DiffusersExecutionStage(PipelineStage):
    """Pipeline stage that wraps diffusers pipeline execution."""

    def __init__(self, diffusers_pipe: DiffusionPipeline):
        super().__init__()
        self.diffusers_pipe = diffusers_pipe

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Execute the diffusers pipeline."""

        kwargs = self._build_pipeline_kwargs(batch, server_args)

        # Filter kwargs to only those supported by the pipeline, warn about ignored args
        kwargs, _ = self._filter_pipeline_kwargs(kwargs)

        # Request tensor output for cleaner handling
        if "output_type" not in kwargs:
            kwargs["output_type"] = "pt"

        with torch.no_grad(), warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            try:
                output = self.diffusers_pipe(**kwargs)
            except TypeError as e:
                # Some pipelines don't support output_type="pt"
                if "output_type" in str(e):
                    kwargs.pop("output_type", None)
                    output = self.diffusers_pipe(**kwargs)
                else:
                    raise

        batch.output = self._extract_output(output)
        if batch.output is not None:
            batch.output = self._postprocess_output(batch.output)

        return batch

    def _filter_pipeline_kwargs(
        self, kwargs: dict, *, strict: bool = False
    ) -> tuple[dict, list[str]]:
        """Filter kwargs to those accepted by the pipeline's __call__.

        Args:
            kwargs: Arguments to filter
            strict: If True, raise ValueError on unsupported args; otherwise warn

        Returns:
            Tuple of (filtered_kwargs, ignored_keys)
        """
        try:
            sig = inspect.signature(self.diffusers_pipe.__call__)
        except (ValueError, TypeError):
            return kwargs, []

        params = sig.parameters
        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        if accepts_var_kwargs:
            return kwargs, []

        valid = set(params.keys()) - {"self"}

        filtered = {}
        ignored = []
        for k, v in kwargs.items():
            if k in valid:
                filtered[k] = v
            else:
                ignored.append(k)

        if ignored:
            pipe_name = type(self.diffusers_pipe).__name__
            msg = (
                f"Pipeline '{pipe_name}' does not support: {', '.join(sorted(ignored))}. "
                "These arguments will be ignored."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)

        return filtered, ignored

    def _extract_output(self, output: Any) -> torch.Tensor | None:
        """Extract tensor output from pipeline result."""
        for attr in ["images", "frames", "video", "sample", "pred_original_sample"]:
            if not hasattr(output, attr):
                continue

            data = getattr(output, attr)
            if data is None:
                continue

            result = self._convert_to_tensor(data)
            if result is not None:
                logger.debug(
                    "Extracted output from '%s': shape=%s, dtype=%s",
                    attr,
                    result.shape,
                    result.dtype,
                )
                return result

        logger.warning("Could not extract output from pipeline result")
        return None

    def _convert_to_tensor(self, data: Any) -> torch.Tensor | None:
        """Convert various data formats to a tensor."""
        if isinstance(data, torch.Tensor):
            return data

        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
            # (B, H, W, C) -> (B, C, H, W) or (B, T, H, W, C) -> (B, C, T, H, W)
            if tensor.ndim == 4:
                tensor = tensor.permute(0, 3, 1, 2)
            elif tensor.ndim == 5:
                tensor = tensor.permute(0, 4, 1, 2, 3)
            return tensor

        if hasattr(data, "mode"):  # PIL Image
            return T.ToTensor()(data)

        if isinstance(data, list) and len(data) > 0:
            return self._convert_list_to_tensor(data)

        return None

    def _convert_list_to_tensor(self, data: list) -> torch.Tensor | None:
        """Convert a list of items to a tensor."""
        first = data[0]

        # Nested list (e.g., [[frame1, frame2, ...]] for video batches)
        if isinstance(first, list) and len(first) > 0:
            data = first
            first = data[0]

        if hasattr(first, "mode"):  # PIL images
            tensors = [T.ToTensor()(img) for img in data]
            stacked = torch.stack(tensors)
            if len(tensors) > 1:
                return stacked.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
            return stacked[0]

        if isinstance(first, torch.Tensor):
            stacked = torch.stack(data)
            if len(data) > 1:
                return stacked.permute(1, 0, 2, 3)
            return stacked[0]

        if isinstance(first, np.ndarray):
            tensors = [torch.from_numpy(arr).float() for arr in data]
            if tensors[0].max() > 1.0:
                tensors = [t / 255.0 for t in tensors]
            if tensors[0].ndim == 3:
                tensors = [t.permute(2, 0, 1) for t in tensors]
            stacked = torch.stack(tensors)
            if len(data) > 1:
                return stacked.permute(1, 0, 2, 3)
            return stacked[0]

        return None

    def _postprocess_output(self, output: torch.Tensor) -> torch.Tensor:
        """Post-process output tensor to ensure valid values and correct shape."""
        output = output.cpu().float()

        # Handle NaN or Inf values
        if torch.isnan(output).any() or torch.isinf(output).any():
            logger.warning("Output contains invalid values, fixing...")
            output = torch.nan_to_num(output, nan=0.5, posinf=1.0, neginf=0.0)

        # Normalize to [0, 1] range if needed
        min_val, max_val = output.min().item(), output.max().item()
        if min_val < -0.5 or max_val > 1.5:
            output = (output + 1) / 2

        output = output.clamp(0, 1)

        # Ensure correct shape for downstream processing
        output = self._fix_output_shape(output)

        logger.debug("Final output tensor shape: %s", output.shape)
        return output

    def _fix_output_shape(self, output: torch.Tensor) -> torch.Tensor:
        """Fix tensor shape for downstream processing.

        Expected: (B, C, H, W) for images or (B, C, T, H, W) for videos.
        """
        if output.dim() == 5:
            # Video: (B, T, C, H, W) -> (B, C, T, H, W)
            return output.permute(0, 2, 1, 3, 4)

        if output.dim() == 4:
            if output.shape[0] == 1 or output.shape[1] in [1, 3, 4]:
                return output  # Already (B, C, H, W)
            # (T, C, H, W) -> (1, C, T, H, W)
            return output.unsqueeze(0).permute(0, 2, 1, 3, 4)

        if output.dim() == 3:
            c, h, w = output.shape
            if c > 4 and w <= 4:
                output = output.permute(2, 0, 1)
            if output.shape[0] == 1:
                output = output.repeat(3, 1, 1)
            return output.unsqueeze(0)

        if output.dim() == 2:
            return output.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

        return output

    def _build_pipeline_kwargs(self, batch: Req, server_args: ServerArgs) -> dict:
        """Build kwargs dict for diffusers pipeline call."""
        kwargs = {}

        if batch.prompt is not None:
            kwargs["prompt"] = batch.prompt

        if batch.negative_prompt:
            kwargs["negative_prompt"] = batch.negative_prompt

        if batch.num_inference_steps is not None:
            kwargs["num_inference_steps"] = batch.num_inference_steps

        if batch.guidance_scale is not None:
            kwargs["guidance_scale"] = batch.guidance_scale

        if batch.true_cfg_scale is not None:
            kwargs["true_cfg_scale"] = batch.true_cfg_scale

        if batch.height is not None:
            kwargs["height"] = batch.height

        if batch.width is not None:
            kwargs["width"] = batch.width

        if batch.num_frames is not None and batch.num_frames > 1:
            kwargs["num_frames"] = batch.num_frames

        # Generator for reproducibility
        if batch.generator is not None:
            kwargs["generator"] = batch.generator
        elif batch.seed is not None:
            device = self._get_pipeline_device()
            kwargs["generator"] = torch.Generator(device=device).manual_seed(batch.seed)

        # Image input for img2img or inpainting
        image = self._load_input_image(batch)
        if image is not None:
            kwargs["image"] = image

        if batch.num_outputs_per_prompt > 1:
            kwargs["num_images_per_prompt"] = batch.num_outputs_per_prompt

        # Extra diffusers-specific kwargs
        if batch.extra:
            diffusers_kwargs = batch.extra.get("diffusers_kwargs", {})
            if diffusers_kwargs:
                kwargs.update(diffusers_kwargs)

        return kwargs

    def _get_pipeline_device(self) -> str:
        """Get the device the pipeline is running on."""
        for attr in ["unet", "transformer", "vae"]:
            component = getattr(self.diffusers_pipe, attr, None)
            if component is not None:
                try:
                    return next(component.parameters()).device
                except StopIteration:
                    pass
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_input_image(self, batch: Req) -> Image.Image | None:
        """Load input image from batch."""
        # Check for PIL image in condition_image or pixel_values
        if batch.condition_image is not None and isinstance(
            batch.condition_image, Image.Image
        ):
            return batch.condition_image
        if batch.pixel_values is not None and isinstance(
            batch.pixel_values, Image.Image
        ):
            return batch.pixel_values

        if not batch.image_path:
            return None

        try:
            if batch.image_path.startswith(("http://", "https://")):
                response = requests.get(batch.image_path, timeout=30)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert("RGB")
            return Image.open(batch.image_path).convert("RGB")
        except Exception as e:
            logger.error("Failed to load image from %s: %s", batch.image_path, e)
            return None


class DiffusersPipeline(ComposedPipelineBase):
    """
    Pipeline wrapper that uses vanilla diffusers pipelines.

    This allows running any diffusers-supported model through sglang's infrastructure
    without requiring native sglang implementation.
    """

    pipeline_name = "DiffusersPipeline"
    is_video_pipeline = False
    _required_config_modules: list[str] = []

    def __init__(
        self,
        model_path: str,
        server_args: ServerArgs,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        executor: PipelineExecutor | None = None,
    ):
        self.server_args = server_args
        self.model_path = model_path
        self._stages: list[PipelineStage] = []
        self._stage_name_mapping: dict[str, PipelineStage] = {}
        self.modules: dict[str, Any] = {}
        self.memory_usages: dict[str, float] = {}
        self.post_init_called = False
        self.executor = executor or SyncExecutor(server_args=server_args)

        logger.info("Loading diffusers pipeline from %s", model_path)
        self.diffusers_pipe = self._load_diffusers_pipeline(model_path, server_args)
        self._detect_pipeline_type()

    def _load_diffusers_pipeline(self, model_path: str, server_args: ServerArgs) -> Any:
        """Load the diffusers pipeline.

        Optimizations applied:
        - device_map: Loads models directly to GPU, warming up CUDA caching allocator
          to avoid small tensor allocations during inference.
        - Parallel shard loading: When using device_map with accelerate, model shards
          are loaded in parallel for faster initialization.
        """

        original_model_path = model_path  # Keep original for custom_pipeline
        model_path = maybe_download_model(model_path)
        self.model_path = model_path

        dtype = self._get_dtype(server_args)
        logger.info("Loading diffusers pipeline with dtype=%s", dtype)

        # Build common kwargs for from_pretrained
        load_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": server_args.trust_remote_code,
            "revision": server_args.revision,
        }

        # Add quantization config if provided (e.g., BitsAndBytesConfig for 4/8-bit)
        config = server_args.pipeline_config
        if config is not None:
            quant_config = getattr(config, "quantization_config", None)
            if quant_config is not None:
                load_kwargs["quantization_config"] = quant_config
                logger.info(
                    "Using quantization config: %s", type(quant_config).__name__
                )

        try:
            pipe = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
        except AttributeError as e:
            if "has no attribute" in str(e):
                # Custom pipeline class not in diffusers - try loading with custom_pipeline
                logger.info(
                    "Pipeline class not found in diffusers, trying custom_pipeline from repo..."
                )
                try:
                    custom_kwargs = {
                        **load_kwargs,
                        "custom_pipeline": original_model_path,
                    }
                    custom_kwargs["trust_remote_code"] = True
                    pipe = DiffusionPipeline.from_pretrained(
                        model_path, **custom_kwargs
                    )
                except Exception as e2:
                    match = re.search(r"has no attribute (\w+)", str(e))
                    class_name = match.group(1) if match else "unknown"
                    raise RuntimeError(
                        f"Pipeline class '{class_name}' not found in diffusers and no custom pipeline.py in repo. "
                        f"Try: pip install --upgrade diffusers (some pipelines require latest version). "
                        f"Original error: {e}"
                    ) from e2
            else:
                raise
        except Exception as e:
            # Only retry with float32 for dtype-related errors
            if "dtype" in str(e).lower() or "float" in str(e).lower():
                logger.warning(
                    "Failed with dtype=%s, falling back to float32: %s", dtype, e
                )
                load_kwargs["torch_dtype"] = torch.float32
                pipe = DiffusionPipeline.from_pretrained(model_path, **load_kwargs)
            else:
                raise

        pipe = pipe.to(get_local_torch_device())
        # Apply VAE memory optimizations from pipeline config
        self._apply_vae_optimizations(pipe, server_args)
        # Apply attention backend if specified
        self._apply_attention_backend(pipe, server_args)
        # Apply cache-dit acceleration if configured
        pipe = self._apply_cache_dit(pipe, server_args)
        logger.info("Loaded diffusers pipeline: %s", pipe.__class__.__name__)
        return pipe

    def _apply_vae_optimizations(self, pipe: Any, server_args: ServerArgs) -> None:
        """Apply VAE memory optimizations (tiling, slicing) from pipeline config."""
        config = server_args.pipeline_config
        if config is None:
            return

        # VAE slicing: decode latents slice-by-slice for lower peak memory
        # https://huggingface.co/docs/diffusers/optimization/memory#vae-slicing
        if getattr(config, "vae_slicing", False):
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
                logger.info("Enabled VAE slicing for lower memory usage")

        # VAE tiling: decode latents tile-by-tile for large images
        # https://huggingface.co/docs/diffusers/optimization/memory#vae-tiling
        if getattr(config, "vae_tiling", False):
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
                logger.info("Enabled VAE tiling for large image support")

    def _apply_attention_backend(self, pipe: Any, server_args: ServerArgs) -> None:
        """Apply attention backend setting from pipeline config or server_args.

        See: https://huggingface.co/docs/diffusers/main/en/optimization/attention_backends
        Available backends: flash, _flash_3_hub, sage, xformers, native, etc.
        """
        backend = server_args.attention_backend

        if backend is None:
            config = server_args.pipeline_config
            if config is not None:
                backend = getattr(config, "diffusers_attention_backend", None)

        if backend is None:
            return

        backend = backend.lower()
        sglang_backends = {e.name.lower() for e in AttentionBackendEnum} | {
            "fa3",
            "fa4",
        }
        if backend in sglang_backends:
            logger.debug(
                "Skipping diffusers attention backend '%s' because it matches a "
                "SGLang backend name. Use diffusers backend names when running "
                "the diffusers backend.",
                backend,
            )
            return

        for component_name in ["transformer", "unet"]:
            component = getattr(pipe, component_name, None)
            if component is not None and hasattr(component, "set_attention_backend"):
                try:
                    component.set_attention_backend(backend)
                    logger.info(
                        "Set attention backend '%s' on %s", backend, component_name
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to set attention backend '%s' on %s: %s",
                        backend,
                        component_name,
                        e,
                    )

    def _apply_cache_dit(self, pipe: Any, server_args: ServerArgs) -> Any:
        """Enable cache-dit for diffusers pipeline if configured."""
        cache_dit_config = server_args.cache_dit_config
        if not cache_dit_config:
            return pipe

        try:
            import cache_dit
        except ImportError as e:
            raise RuntimeError(
                "cache-dit is required for --cache-dit-config. "
                "Install it with `pip install cache-dit`."
            ) from e

        if not hasattr(cache_dit, "load_configs"):
            raise RuntimeError(
                "cache-dit>=1.2.0 is required for --cache-dit-config. "
                "Please upgrade cache-dit."
            )

        try:
            cache_options = cache_dit.load_configs(cache_dit_config)
        except Exception as e:
            raise ValueError(
                "Failed to load cache-dit config. Provide a YAML/JSON path (or a dict "
                "supported by cache-dit>=1.2.0)."
            ) from e

        try:
            pipe = cache_dit.enable_cache(pipe, **cache_options)
        except Exception:
            # cache-dit is an external integration and can raise a variety of errors.
            logger.exception("Failed to enable cache-dit for diffusers pipeline")
            raise

        logger.info("Enabled cache-dit for diffusers pipeline")
        return pipe

    def _get_dtype(self, server_args: ServerArgs) -> torch.dtype:
        """
        Determine the dtype to use for model loading.
        """
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if hasattr(server_args, "pipeline_config") and server_args.pipeline_config:
            dit_precision = server_args.pipeline_config.dit_precision
            if dit_precision == "fp16":
                dtype = torch.float16
            elif dit_precision == "bf16":
                dtype = torch.bfloat16
            elif dit_precision == "fp32":
                dtype = torch.float32

        return dtype

    def _detect_pipeline_type(self):
        """Detect if this is an image or video pipeline."""
        pipe_class_name = self.diffusers_pipe.__class__.__name__.lower()
        video_indicators = ["video", "animat", "cogvideo", "wan", "hunyuan"]
        self.is_video_pipeline = any(ind in pipe_class_name for ind in video_indicators)
        logger.debug(
            "Detected pipeline type: %s",
            "video" if self.is_video_pipeline else "image",
        )

    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Skip sglang's module loading - diffusers handles it."""
        return {"diffusers_pipeline": self.diffusers_pipe}

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Create the execution stage wrapping the diffusers pipeline."""
        self.add_stage(
            stage_name="diffusers_execution",
            stage=DiffusersExecutionStage(self.diffusers_pipe),
        )

    def initialize_pipeline(self, server_args: ServerArgs):
        """Initialize the pipeline."""
        pass

    def post_init(self) -> None:
        """Post initialization hook."""
        if self.post_init_called:
            return
        self.post_init_called = True
        self.initialize_pipeline(self.server_args)
        self.create_pipeline_stages(self.server_args)

    def add_stage(self, stage_name: str, stage: PipelineStage):
        """Add a stage to the pipeline."""
        self._stages.append(stage)
        self._stage_name_mapping[stage_name] = stage
        setattr(self, stage_name, stage)

    @property
    def stages(self) -> list[PipelineStage]:
        """List of stages in the pipeline."""
        return self._stages

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        """Execute the pipeline on the given batch."""
        if not self.post_init_called:
            self.post_init()
        return self.executor.execute(self.stages, batch, server_args)

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        pipeline_config: str | PipelineConfig | None = None,
        args: argparse.Namespace | None = None,
        required_config_modules: list[str] | None = None,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
        **kwargs,
    ) -> "DiffusersPipeline":
        """Load a pipeline from a pretrained model using diffusers backend."""
        kwargs["model_path"] = model_path
        server_args = ServerArgs.from_kwargs(**kwargs)

        pipe = cls(
            model_path,
            server_args,
            required_config_modules=required_config_modules,
            loaded_modules=loaded_modules,
        )
        pipe.post_init()
        return pipe

    def get_module(self, module_name: str, default_value: Any = None) -> Any:
        """Get a module by name."""
        if module_name == "diffusers_pipeline":
            return self.diffusers_pipe
        return self.modules.get(module_name, default_value)


EntryClass = DiffusersPipeline
