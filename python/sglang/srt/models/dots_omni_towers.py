"""In-process vision/audio towers for dots.note.omni.

The converted checkpoint stores the towers in ``new_ve`` and ``new_ae`` next
to the language-model shards.  This module adapts the existing native tower
implementations to SGLang without starting the legacy gRPC encoder service.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import types
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def _install_encoder_source_compat():
    """Load the source implementation without its serving-only package setup."""
    source_root = Path(
        os.getenv(
            "DOTS_OMNI_ENCODER_REPO",
            "/cpfs/user/qianwu/mm_encoder_server",
        )
    )
    if not source_root.is_dir():
        raise RuntimeError(
            "dots.note.omni native towers require the encoder implementation at "
            f"{source_root}. Set DOTS_OMNI_ENCODER_REPO to override it."
        )
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)

    # Importing python.multimodal_processors normally executes the encoder
    # server registry and its internal logging/metrics dependencies.  The model
    # only needs the native kernels, so expose a narrow namespace package.
    package_name = "python.multimodal_processors"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(source_root / "python" / "multimodal_processors")]
        sys.modules[package_name] = package

    if "python.log" not in sys.modules:
        log_module = types.ModuleType("python.log")
        log_module.logger = logger
        sys.modules["python.log"] = log_module

    if "python.metrics" not in sys.modules:
        metrics_package = types.ModuleType("python.metrics")
        metrics_package.__path__ = []
        sys.modules["python.metrics"] = metrics_package
    if "python.metrics.metric_impl" not in sys.modules:
        metrics_module = types.ModuleType("python.metrics.metric_impl")
        metrics_module.record_mm_image_process_error_num = lambda *args, **kwargs: None
        metrics_module.record_mm_request_step_latency = lambda *args, **kwargs: None
        sys.modules["python.metrics.metric_impl"] = metrics_module

    # The encoder source predates the fp8 kernel namespace move.
    import sglang.kernels.ops.quantization.fp8_kernel as fp8_kernel

    sys.modules.setdefault(
        "sglang.srt.layers.quantization.fp8_kernel", fp8_kernel
    )

    # The legacy ViT imports the functional Triton MoE runner from its old
    # location inside ``forward``.  Keep the legacy import stable while using
    # the current implementation and types.
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe

    old_fused_moe_name = "sglang.srt.layers.moe.fused_moe_triton.fused_moe"
    if old_fused_moe_name not in sys.modules:
        fused_moe_module = types.ModuleType(old_fused_moe_name)
        fused_moe_module.fused_moe = fused_moe
        sys.modules[old_fused_moe_name] = fused_moe_module

    # Current SGLang ships FA3 and leaves ``flash_attn`` as a namespace
    # package.  The legacy speech encoder imports the FA2 symbols
    # unconditionally before selecting FA3, so provide lazy compatibility
    # symbols; the FA3 branch is used on Hopper at runtime.
    import flash_attn

    def _fa3_func(*args, **kwargs):
        from sglang.kernels.ops.attention.flash_attention import (
            flash_attn_varlen_func,
        )

        kwargs.pop("dropout_p", None)
        return_attn_probs = kwargs.pop("return_attn_probs", False)
        return flash_attn_varlen_func(
            *args,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            return_softmax_lse=return_attn_probs,
            **kwargs,
        )

    def _fa3_varlen_func(*args, **kwargs):
        from sglang.kernels.ops.attention.flash_attention import (
            flash_attn_varlen_func,
        )

        kwargs.pop("dropout_p", None)
        return_attn_probs = kwargs.pop("return_attn_probs", False)
        return flash_attn_varlen_func(
            *args,
            return_softmax_lse=return_attn_probs,
            **kwargs,
        )

    if not hasattr(flash_attn, "flash_attn_func"):
        flash_attn.flash_attn_func = _fa3_func
    if not hasattr(flash_attn, "flash_attn_varlen_func"):
        flash_attn.flash_attn_varlen_func = _fa3_varlen_func
    return source_root


_ENCODER_SOURCE_ROOT = _install_encoder_source_compat()

from python.multimodal_processors.models.model_dots_moe_vit import (  # noqa: E402
    DotsMoEVitConfig,
    DotsMoEVitModel,
)
from python.multimodal_processors.models.model_omni_audio import (  # noqa: E402
    OmniAudioConfig,
    OmniAudioModel,
    compute_audio_token_length,
)


def _read_json(path: Path) -> dict:
    with path.open() as file:
        return json.load(file)


def _load_safetensor_state(model_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    index = _read_json(model_dir / "model.safetensors.index.json")
    shard_names = sorted(set(index["weight_map"].values()))
    state = {}
    for shard_name in shard_names:
        state.update(load_file(str(model_dir / shard_name), device="cpu"))
    return state


class DotsNoteOmniVisionEncoder(DotsMoEVitModel):
    """Native MoE ViT used by dots.note.omni."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        config = DotsMoEVitConfig(**_read_json(self.model_dir / "config.json"))
        super().__init__(config)
        self.to(torch.bfloat16)

    def load_converted_weights(self):
        missing, unexpected = self.load_state_dict(
            _load_safetensor_state(self.model_dir), strict=False
        )
        if missing:
            raise RuntimeError(f"Dots vision tower missing weights: {missing[:8]}")
        if unexpected:
            raise RuntimeError(
                f"Dots vision tower has unexpected weights: {unexpected[:8]}"
            )
        if getattr(self.config, "enable_torch_compile", False):
            self.compile_block_modules()


class DotsNoteOmniAudioEncoder(OmniAudioModel):
    """Native Dots speech encoder and adapter."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        config = OmniAudioConfig(**_read_json(self.model_dir / "config.json"))
        super().__init__(config)
        self.to(torch.bfloat16)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def load_converted_weights(self):
        missing, unexpected = self.load_state_dict(
            _load_safetensor_state(self.model_dir), strict=True
        )
        if missing or unexpected:
            raise RuntimeError(
                "Dots audio tower weight mismatch: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}"
            )


class DotsNoteOmniImagePreprocessor:
    """CPU image preprocessing matching the converted native ViT."""

    def __init__(self, model_dir: str):
        model_dir = Path(model_dir)
        config = _read_json(model_dir / "preprocessor_config.json")
        self.min_pixels = config["min_pixels"]
        self.max_pixels = config["max_pixels"]
        self.patch_size = config["patch_size"]
        self.temporal_patch_size = config["temporal_patch_size"]
        self.merge_size = config["merge_size"]
        self.pre_pixel_shuffle = config.get("pre_pixel_shuffle", True)
        self.image_mean = np.asarray(config["image_mean"], dtype=np.float32)
        self.image_std = np.asarray(config["image_std"], dtype=np.float32)
        image_detail_path = model_dir / "image_detail.json"
        self.image_detail_config = (
            _read_json(image_detail_path).get("image_details", {})
            if image_detail_path.is_file()
            else {}
        )

    @staticmethod
    def _round_by_factor(value: int, factor: int) -> int:
        return round(value / factor) * factor

    @staticmethod
    def _ceil_by_factor(value: float, factor: int) -> int:
        return math.ceil(value / factor) * factor

    @staticmethod
    def _floor_by_factor(value: float, factor: int) -> int:
        return math.floor(value / factor) * factor

    def _resized_size(
        self,
        width: int,
        height: int,
        min_pixels: int,
        max_pixels: int,
        target_height=None,
        target_width=None,
    ):
        height = target_height or height
        width = target_width or width
        factor = self.patch_size * self.merge_size
        if min(height, width) < factor // 4:
            raise ValueError(
                f"Image height/width must be at least {factor // 4}, "
                f"got {height}x{width}"
            )
        if max(height, width) / min(height, width) > 200:
            raise ValueError("Image aspect ratio must be smaller than 200")
        resized_h = max(factor, self._round_by_factor(height, factor))
        resized_w = max(factor, self._round_by_factor(width, factor))
        if resized_h * resized_w > max_pixels:
            beta = math.sqrt(height * width / max_pixels)
            resized_h = max(
                factor, self._floor_by_factor(height / beta, factor)
            )
            resized_w = max(
                factor, self._floor_by_factor(width / beta, factor)
            )
        elif resized_h * resized_w < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            resized_h = self._ceil_by_factor(height * beta, factor)
            resized_w = self._ceil_by_factor(width * beta, factor)
            if resized_h * resized_w > max_pixels:
                beta = math.sqrt(resized_h * resized_w / max_pixels)
                resized_h = max(
                    factor, self._floor_by_factor(resized_h / beta, factor)
                )
                resized_w = max(
                    factor, self._floor_by_factor(resized_w / beta, factor)
                )
        return resized_h, resized_w

    def _process_image(self, image, detail="auto"):
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL image, got {type(image)}")
        if image.mode == "RGBA":
            background = Image.new("RGB", image.size, (255, 255, 255))
            background.paste(image, mask=image.getchannel("A"))
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        detail_config = self.image_detail_config.get(detail, {})
        resized_h, resized_w = self._resized_size(
            *image.size,
            min_pixels=detail_config.get("min_pixels", self.min_pixels),
            max_pixels=detail_config.get("max_pixels", self.max_pixels),
            target_height=detail_config.get("target_height"),
            target_width=detail_config.get("target_width"),
        )
        image = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - self.image_mean) / self.image_std
        patches = array.transpose(2, 0, 1)[None]
        if patches.shape[0] == 1:
            patches = np.tile(
                patches, (self.temporal_patch_size, 1, 1, 1)
            )
        channel = patches.shape[1]
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h = resized_h // self.patch_size
        grid_w = resized_w // self.patch_size
        if self.pre_pixel_shuffle:
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
        else:
            patches = patches.reshape(
                grid_t,
                self.temporal_patch_size,
                channel,
                grid_h,
                self.patch_size,
                grid_w,
                self.patch_size,
            )
            patches = patches.transpose(0, 3, 5, 2, 1, 4, 6)
        pixel_values = torch.from_numpy(
            patches.reshape(
                grid_t * grid_h * grid_w,
                channel
                * self.temporal_patch_size
                * self.patch_size
                * self.patch_size,
            )
        )
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": torch.tensor([[grid_t, grid_h, grid_w]]),
        }

    def _get_image_token_str(self, token_count: int):
        return "<|img|>" + "<|imgpad|>" * token_count + "<|endofimg|>"

    def process_images(self, images: Iterable, details=None):
        images = list(images)
        details = details or ["auto"] * len(images)
        pixel_values = []
        grids = []
        token_strings = []
        for image, detail in zip(images, details):
            processed = self._process_image(image, detail)
            grid = processed["image_grid_thw"]
            token_count = int(grid.prod().item()) // self.merge_size**2
            pixel_values.append(processed["pixel_values"])
            grids.append(grid)
            token_strings.append(self._get_image_token_str(token_count))
        return pixel_values, grids, token_strings


def get_audio_token_count(num_samples: int, config: OmniAudioConfig) -> int:
    return compute_audio_token_length(
        num_samples,
        chunk_seconds=config.chunk_seconds,
        conv_temporal_stride=config.conv_temporal_stride,
        merge_factor=config.merge_factor,
    )


def get_audio_token_string(num_samples: int, config: OmniAudioConfig) -> str:
    count = get_audio_token_count(num_samples, config)
    return config.audio_comp_start + config.audio_comp_span * count + config.audio_comp_end
