# SPDX-License-Identifier: Apache-2.0
"""
Real-ESRGAN upscaling for SGLang diffusion pipelines.

Real-ESRGAN model code is vendored and adapted from:
  - https://github.com/xinntao/Real-ESRGAN  (BSD-3-Clause License)
  Copyright (c) 2021 xinntao

The ImageUpscaler wrapper and integration code are original work.
"""

import math
import os
import time
from hashlib import sha256
from typing import Optional
from urllib.parse import unquote, urlparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Default HuggingFace repo and filename for Real-ESRGAN weights
_DEFAULT_REALESRGAN_HF_REPO = "ai-forever/Real-ESRGAN"
_DEFAULT_REALESRGAN_FILENAME = "RealESRGAN_x4.pth"
_DEFAULT_REALESRGAN_FILENAMES_BY_SCALE = {
    2: "RealESRGAN_x2.pth",
    4: "RealESRGAN_x4.pth",
    8: "RealESRGAN_x8.pth",
}
_LOW_MEMORY_TILED_UPSCALE_FREE_BYTES = 2 * 1024**3
_REALESRGAN_TILE_SIZE = 256
_REALESRGAN_TILE_PAD = 32

# Module-level cache: model_path -> UpscalerModel instance
_MODEL_CACHE: dict[str, "UpscalerModel"] = {}
_RESOLVED_MODEL_PATH_CACHE: dict[str, str] = {}


def _default_model_path_for_scale(scale: int) -> str:
    filename = _DEFAULT_REALESRGAN_FILENAMES_BY_SCALE.get(
        int(scale),
        _DEFAULT_REALESRGAN_FILENAME,
    )
    return f"{_DEFAULT_REALESRGAN_HF_REPO}:{filename}"


# ---------------------------------------------------------------------------
# Vendored Real-ESRGAN architecture code
# (SRVGGNetCompact, ResidualDenseBlock, RRDB, RRDBNet)
# ---------------------------------------------------------------------------


class SRVGGNetCompact(nn.Module):
    """Compact VGG-style network for super resolution.

    Corresponds to ``realesr-animevideov3`` and ``realesr-general-x4v3``.
    Reference: xinntao/Real-ESRGAN (BSD-3-Clause).
    """

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        num_feat: int = 64,
        num_conv: int = 16,
        upscale: int = 4,
        act_type: str = "prelu",
    ):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        self.body = nn.ModuleList()
        # first conv
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        # first activation
        self.body.append(self._make_act(act_type, num_feat))
        # body convs + activations
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(self._make_act(act_type, num_feat))
        # last conv: maps to out_ch * upscale^2 for pixel shuffle
        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    @staticmethod
    def _make_act(act_type: str, num_feat: int) -> nn.Module:
        if act_type == "relu":
            return nn.ReLU(inplace=True)
        elif act_type == "prelu":
            return nn.PReLU(num_parameters=num_feat)
        elif act_type == "leakyrelu":
            return nn.LeakyReLU(negative_slope=0.1, inplace=True)
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        # residual addition with nearest upsampled input
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block used in RRDB (RealESRGAN_x4plus)."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """RRDB network for RealESRGAN_x4plus (heavier, higher quality for photos)."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
    ):
        super().__init__()
        self.scale = scale
        in_ch = num_in_ch
        if scale == 2:
            in_ch = num_in_ch * 4
        elif scale == 1:
            in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = F.pixel_unshuffle(x, 2)
        elif self.scale == 1:
            feat = F.pixel_unshuffle(x, 4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


# ---------------------------------------------------------------------------
# Architecture auto-detection
# ---------------------------------------------------------------------------


def _build_net_from_state_dict(state_dict: dict) -> nn.Module:
    """Detect architecture from checkpoint keys and return an unloaded network."""
    if "conv_first.weight" in state_dict:
        # RRDBNet (e.g., RealESRGAN_x4plus)
        num_feat = state_dict["conv_first.weight"].shape[0]
        in_channels = state_dict["conv_first.weight"].shape[1]
        if in_channels == 3:
            scale = 4
        elif in_channels == 12:
            scale = 2
        elif in_channels == 48:
            scale = 1
        else:
            raise ValueError(
                f"Unsupported RRDBNet conv_first input channels: {in_channels}"
            )
        num_block = sum(
            1
            for k in state_dict
            if k.startswith("body.") and k.endswith(".rdb1.conv1.weight")
        )
        num_grow_ch = state_dict["body.0.rdb1.conv1.weight"].shape[0]
        logger.info(
            "Detected RRDBNet: num_feat=%d, num_block=%d, num_grow_ch=%d, scale=%d",
            num_feat,
            num_block,
            num_grow_ch,
            scale,
        )
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=scale,
            num_feat=num_feat,
            num_block=num_block,
            num_grow_ch=num_grow_ch,
        )
    else:
        # SRVGGNetCompact (e.g., realesr-animevideov3)
        num_feat = state_dict["body.0.weight"].shape[0]
        # body layout: [first_conv, first_act, (conv, act)*num_conv, last_conv]
        # count 4-D weight tensors = first_conv + loop_convs + last_conv = num_conv + 2
        conv_keys = sorted(
            [
                k
                for k in state_dict
                if k.startswith("body.")
                and k.endswith(".weight")
                and state_dict[k].dim() == 4
            ],
            key=lambda k: int(k.split(".")[1]),
        )
        num_conv = len(conv_keys) - 2  # subtract first and last
        # upscale from last conv output channels: out_ch = num_out_ch * upscale^2
        last_out_ch = state_dict[conv_keys[-1]].shape[0]
        upscale = int(math.sqrt(last_out_ch / 3))
        logger.info(
            "Detected SRVGGNetCompact: num_feat=%d, num_conv=%d, upscale=%d",
            num_feat,
            num_conv,
            upscale,
        )
        return SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=num_feat,
            num_conv=num_conv,
            upscale=upscale,
            act_type="prelu",
        )


# ---------------------------------------------------------------------------
# UpscalerModel
# ---------------------------------------------------------------------------


class UpscalerModel:
    """Wraps a Real-ESRGAN network, provides load() and upscale() API."""

    def __init__(self, net: nn.Module, scale: int):
        self.net = net
        self.scale = scale  # the model's native upscaling factor (e.g. 4)

    @property
    def device(self) -> torch.device:
        return next(self.net.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.net.parameters()).dtype

    def _copy_input_to_device(self, frames: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(frames).to(self.device)

    def _preprocess_input_tensor(self, imgs_t: torch.Tensor) -> torch.Tensor:
        imgs_t = imgs_t.permute(0, 3, 1, 2).to(dtype=self.dtype).mul_(1.0 / 255.0)
        if self.device.type == "cuda":
            imgs_t = imgs_t.contiguous(memory_format=torch.channels_last)
        return imgs_t

    @staticmethod
    def _postprocess_output_tensor(out: torch.Tensor) -> torch.Tensor:
        out = out.permute(0, 2, 3, 1).clamp(0.0, 1.0).mul_(255.0)
        return out.to(torch.uint8).contiguous()

    @staticmethod
    def _copy_output_to_host(out: torch.Tensor) -> np.ndarray:
        return out.cpu().numpy()

    def _start_cuda_timer(self):
        if self.device.type != "cuda":
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        return start, end

    @staticmethod
    def _stop_cuda_timer(timer) -> None:
        if timer is not None:
            timer[1].record()

    @staticmethod
    def _cuda_elapsed_s(timer, fallback_s: float) -> float:
        if timer is None:
            return fallback_s
        timer[1].synchronize()
        return timer[0].elapsed_time(timer[1]) / 1000.0

    def _should_use_tiled_upscale(self, h: int, w: int) -> bool:
        if self.device.type != "cuda":
            return False
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        output_bytes = h * w * self.scale * self.scale * 3 * 4
        required_free_bytes = max(
            _LOW_MEMORY_TILED_UPSCALE_FREE_BYTES,
            output_bytes * 4,
        )
        return free_bytes < required_free_bytes

    def _upscale_tiled_to_cpu(
        self,
        img_t: torch.Tensor,
        tile_size: int = _REALESRGAN_TILE_SIZE,
        tile_pad: int = _REALESRGAN_TILE_PAD,
    ) -> torch.Tensor:
        _, channels, h, w = img_t.shape
        scale = self.scale
        output = torch.empty(
            (1, channels, h * scale, w * scale),
            dtype=torch.float32,
            device="cpu",
        )

        for y in range(0, h, tile_size):
            tile_h = min(tile_size, h - y)
            in_y0 = max(y - tile_pad, 0)
            in_y1 = min(y + tile_h + tile_pad, h)
            out_y0 = y * scale
            out_y1 = (y + tile_h) * scale
            crop_y0 = (y - in_y0) * scale
            crop_y1 = crop_y0 + tile_h * scale

            for x in range(0, w, tile_size):
                tile_w = min(tile_size, w - x)
                in_x0 = max(x - tile_pad, 0)
                in_x1 = min(x + tile_w + tile_pad, w)
                out_x0 = x * scale
                out_x1 = (x + tile_w) * scale
                crop_x0 = (x - in_x0) * scale
                crop_x1 = crop_x0 + tile_w * scale

                tile = img_t[..., in_y0:in_y1, in_x0:in_x1]
                out_tile = self.net(tile)
                out_tile = out_tile[..., crop_y0:crop_y1, crop_x0:crop_x1].float()
                output[..., out_y0:out_y1, out_x0:out_x1].copy_(out_tile.cpu())

        return output

    def upscale(self, frame: np.ndarray, outscale: float | None = None) -> np.ndarray:
        """Upscale a single HWC uint8 frame → HWC uint8 frame.

        Args:
            frame:    Input HWC uint8 numpy array.
            outscale: Desired final upscaling factor. If different from the
                      model's native scale, a cheap resize is applied after
                      the network output (same approach as the official
                      Real-ESRGAN ``inference_realesrgan.py --outscale``).
                      ``None`` means use the model's native scale as-is.
        """
        h, w = frame.shape[:2]
        img = frame.astype(np.float32) / 255.0
        img_t = (
            torch.from_numpy(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.dtype)
        )
        with torch.no_grad():
            if self._should_use_tiled_upscale(h, w):
                logger.info(
                    "Using tiled Real-ESRGAN upscale for low GPU memory: "
                    "frame=%dx%d, tile_size=%d, tile_pad=%d",
                    w,
                    h,
                    _REALESRGAN_TILE_SIZE,
                    _REALESRGAN_TILE_PAD,
                )
                out = self._upscale_tiled_to_cpu(img_t)
            else:
                try:
                    out = self.net(img_t)
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda":
                        raise
                    torch.cuda.empty_cache()
                    logger.warning(
                        "Real-ESRGAN full-frame upscale OOM; retrying with tiled upscale"
                    )
                    out = self._upscale_tiled_to_cpu(img_t)

        # If the desired outscale differs from the model's native scale,
        # resize to (h * outscale, w * outscale).
        if outscale is not None and outscale != self.scale:
            target_h = int(h * outscale)
            target_w = int(w * outscale)
            out = F.interpolate(
                out, size=(target_h, target_w), mode="bicubic", align_corners=False
            )

        out_np = out.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        return (out_np * 255.0).astype(np.uint8)

    def upscale_batch(
        self, frames: list[np.ndarray], outscale: float | None = None
    ) -> list[np.ndarray]:
        """Upscale same-resolution HWC uint8 frames in one batched forward pass."""
        if not frames:
            return []

        h, w = frames[0].shape[:2]
        if any(frame.shape[:2] != (h, w) for frame in frames):
            raise ValueError("All frames in a batch must have the same resolution")

        total_start_time = time.perf_counter()

        start_time = time.perf_counter()
        imgs = np.stack(frames, axis=0)
        stack_duration_s = time.perf_counter() - start_time

        start_time = time.perf_counter()
        h2d_timer = self._start_cuda_timer()
        imgs_t = self._copy_input_to_device(imgs)
        self._stop_cuda_timer(h2d_timer)
        h2d_wall_duration_s = time.perf_counter() - start_time

        start_time = time.perf_counter()
        input_preprocess_timer = self._start_cuda_timer()
        imgs_t = self._preprocess_input_tensor(imgs_t)
        self._stop_cuda_timer(input_preprocess_timer)
        input_preprocess_wall_duration_s = time.perf_counter() - start_time

        start_time = time.perf_counter()
        forward_timer = self._start_cuda_timer()
        with torch.inference_mode():
            out = self.net(imgs_t)
        self._stop_cuda_timer(forward_timer)
        forward_wall_duration_s = time.perf_counter() - start_time

        resize_timer = None
        resize_wall_duration_s = 0.0
        if outscale is not None and outscale != self.scale:
            start_time = time.perf_counter()
            resize_timer = self._start_cuda_timer()
            target_h = int(h * outscale)
            target_w = int(w * outscale)
            out = F.interpolate(
                out, size=(target_h, target_w), mode="bicubic", align_corners=False
            )
            self._stop_cuda_timer(resize_timer)
            resize_wall_duration_s = time.perf_counter() - start_time

        start_time = time.perf_counter()
        output_postprocess_timer = self._start_cuda_timer()
        out = self._postprocess_output_tensor(out)
        self._stop_cuda_timer(output_postprocess_timer)
        output_postprocess_wall_duration_s = time.perf_counter() - start_time

        start_time = time.perf_counter()
        output_d2h_timer = self._start_cuda_timer()
        out_np = self._copy_output_to_host(out)
        self._stop_cuda_timer(output_d2h_timer)
        output_d2h_wall_duration_s = time.perf_counter() - start_time

        start_time = time.perf_counter()
        outputs = [frame for frame in out_np]
        post_duration_s = time.perf_counter() - start_time

        h2d_duration_s = self._cuda_elapsed_s(h2d_timer, h2d_wall_duration_s)
        input_preprocess_duration_s = self._cuda_elapsed_s(
            input_preprocess_timer, input_preprocess_wall_duration_s
        )
        forward_duration_s = self._cuda_elapsed_s(
            forward_timer, forward_wall_duration_s
        )
        resize_duration_s = self._cuda_elapsed_s(resize_timer, resize_wall_duration_s)
        output_postprocess_duration_s = self._cuda_elapsed_s(
            output_postprocess_timer, output_postprocess_wall_duration_s
        )
        output_d2h_duration_s = self._cuda_elapsed_s(
            output_d2h_timer, output_d2h_wall_duration_s
        )
        total_duration_s = time.perf_counter() - total_start_time
        timing_source = "cuda_event" if self.device.type == "cuda" else "wall"
        logger.info(
            "RealESRGAN batch upscale: batch=%d input=%dx%d native_scale=%dx outscale=%s "
            "dtype=%s timing=%s total=%.3fs stack=%.3fs input_h2d=%.3fs "
            "input_pre=%.3fs forward=%.3fs resize=%.3fs output_post=%.3fs "
            "output_d2h=%.3fs python_post=%.3fs",
            len(frames),
            w,
            h,
            self.scale,
            outscale if outscale is not None else self.scale,
            self.dtype,
            timing_source,
            total_duration_s,
            stack_duration_s,
            h2d_duration_s,
            input_preprocess_duration_s,
            forward_duration_s,
            resize_duration_s,
            output_postprocess_duration_s,
            output_d2h_duration_s,
            post_duration_s,
        )
        return outputs


# ---------------------------------------------------------------------------
# ImageUpscaler public class
# ---------------------------------------------------------------------------


class ImageUpscaler:
    """
    Lazy-loaded Real-ESRGAN upscaler.

    Weights are downloaded and cached on first call to `.upscale()`.
    Supports both SRVGGNetCompact (lightweight, default) and RRDBNet (heavier).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        scale: int = 4,
        half_precision: bool = False,
    ):
        self._model_path = model_path
        self._scale = scale
        self._half_precision = half_precision

    def _ensure_model_loaded(self) -> UpscalerModel:
        """Download/load Real-ESRGAN weights, detect arch, and cache globally."""
        model_path = self._model_path or _default_model_path_for_scale(self._scale)

        # Resolve: local .pth pass-through, or HF repo → download single file
        resolved_path = _resolve_model_path(model_path)

        if resolved_path in _MODEL_CACHE:
            return _MODEL_CACHE[resolved_path]

        logger.info("Loading Real-ESRGAN weights from %s", resolved_path)
        try:
            state_dict = torch.load(
                resolved_path, map_location="cpu", weights_only=True
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Real-ESRGAN checkpoint from '{resolved_path}'. "
                f"The file may be corrupted or not a valid PyTorch checkpoint. "
                f"Original error: {e}"
            ) from e

        # Some checkpoints wrap weights under a 'params' or 'params_ema' key
        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        try:
            net = _build_net_from_state_dict(state_dict)
            net.load_state_dict(state_dict, strict=True)
        except (RuntimeError, KeyError) as e:
            raise RuntimeError(
                f"Real-ESRGAN weight file '{resolved_path}' is not compatible "
                f"with the supported architectures (SRVGGNetCompact / RRDBNet). "
                f"Please ensure you are using a valid Real-ESRGAN checkpoint. "
                f"Original error: {e}"
            ) from e
        net.eval()

        device = current_platform.get_local_torch_device()
        if self._half_precision:
            net = net.half()
        net = net.to(device)

        # Detect the model's native scale from network architecture
        native_scale = 4  # sensible default
        if hasattr(net, "upscale"):
            native_scale = net.upscale
        elif hasattr(net, "scale"):
            native_scale = net.scale

        model = UpscalerModel(net=net, scale=native_scale)
        _MODEL_CACHE[resolved_path] = model
        logger.info(
            "Real-ESRGAN model loaded on device: %s (native_scale=%dx, outscale=%s)",
            device,
            native_scale,
            f"{self._scale}x" if self._scale != native_scale else "native",
        )
        return model

    def upscale(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a list of HWC uint8 frames.

        Uses the model's native scale for super-resolution, then resizes to
        the desired ``outscale`` if it differs (cheap bicubic resize).
        """
        if not frames:
            return frames
        model = self._ensure_model_loaded()
        outscale = self._scale if self._scale != model.scale else None
        return [model.upscale(frame, outscale=outscale) for frame in frames]

    def upscale_batched(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale HWC uint8 frames with batched forwards grouped by resolution."""
        if not frames:
            return frames
        total_start_time = time.perf_counter()
        model = self._ensure_model_loaded()
        outscale = self._scale if self._scale != model.scale else None
        output_frames: list[np.ndarray | None] = [None] * len(frames)
        groups: dict[tuple[int, ...], list[int]] = {}
        for idx, frame in enumerate(frames):
            groups.setdefault(tuple(frame.shape), []).append(idx)

        for shape, indices in groups.items():
            logger.info(
                "RealESRGAN upscale group: frames=%d shape=%s indices=%s",
                len(indices),
                shape,
                indices,
            )
            group_frames = [frames[idx] for idx in indices]
            group_outputs = model.upscale_batch(group_frames, outscale=outscale)
            for idx, output in zip(indices, group_outputs):
                output_frames[idx] = output

        if any(frame is None for frame in output_frames):
            raise RuntimeError("RealESRGAN batch upscale did not produce all frames")

        total_duration_s = time.perf_counter() - total_start_time
        logger.info(
            "RealESRGAN batch_upscale_frames completed in %.3f seconds for %d frames across %d groups",
            total_duration_s,
            len(frames),
            len(groups),
        )
        return [frame for frame in output_frames if frame is not None]


# ---------------------------------------------------------------------------
# HF download helper
# ---------------------------------------------------------------------------


def _resolve_model_path(model_path: str) -> str:
    """Return a local .pth file path.

    Accepts:
    - An existing local file path (pass-through).
    - An http(s) URL to a .pth file, downloaded into the local cache.
    - A HuggingFace ``repo_id`` → downloads the default weight file
      (``RealESRGAN_x4.pth``).
    - A HuggingFace ``repo_id:filename`` → downloads *filename* from *repo_id*,
      allowing users to specify custom weight files hosted on HF.
    """
    cached_path = _RESOLVED_MODEL_PATH_CACHE.get(model_path)
    if cached_path is not None:
        return cached_path

    if os.path.isfile(model_path):
        _RESOLVED_MODEL_PATH_CACHE[model_path] = model_path
        return model_path

    parsed_url = urlparse(model_path)
    if parsed_url.scheme in ("http", "https"):
        filename = (
            os.path.basename(unquote(parsed_url.path)) or _DEFAULT_REALESRGAN_FILENAME
        )
        cache_dir = os.path.join(
            os.path.expanduser("~"), ".cache", "sglang", "realesrgan"
        )
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = sha256(model_path.encode("utf-8")).hexdigest()[:12]
        local_path = os.path.join(cache_dir, f"{cache_key}-{filename}")
        if not os.path.isfile(local_path):
            tmp_path = f"{local_path}.tmp"
            logger.info("Downloading Real-ESRGAN weights from URL %s", model_path)
            torch.hub.download_url_to_file(model_path, tmp_path, progress=False)
            os.replace(tmp_path, local_path)
        _RESOLVED_MODEL_PATH_CACHE[model_path] = local_path
        return local_path

    # Parse optional "repo_id:filename" syntax; fall back to default filename.
    if ":" in model_path and not model_path.startswith("/"):
        repo_id, filename = model_path.split(":", 1)
    else:
        repo_id = model_path
        filename = _DEFAULT_REALESRGAN_FILENAME

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required to download Real-ESRGAN weights. "
            "Install it with: pip install huggingface_hub"
        ) from e

    logger.info(
        "Downloading Real-ESRGAN weights from HF repo %s (file: %s)",
        repo_id,
        filename,
    )
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
    except Exception as e:
        raise FileNotFoundError(
            f"Failed to download Real-ESRGAN weights from HuggingFace repo "
            f"'{repo_id}' (file: '{filename}'). If you are using a custom "
            f"model, provide either a local .pth file path or use the "
            f"'repo_id:filename' format (e.g. 'my-org/my-esrgan:weights.pth'). "
            f"Original error: {e}"
        ) from e
    _RESOLVED_MODEL_PATH_CACHE[model_path] = local_path
    return local_path


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def upscale_frames(
    frames: list[np.ndarray],
    model_path: Optional[str] = None,
    scale: int = 4,
    half_precision: bool = False,
) -> list[np.ndarray]:
    """
    Convenience wrapper around ImageUpscaler.

    The model always runs at its native resolution (e.g. 4× for
    ``RealESRGAN_x4.pth``).  If *scale* differs from the native factor,
    a cheap bicubic resize is applied after the network output – the same
    approach used by the official Real-ESRGAN ``--outscale`` flag.

    Args:
        frames:         List of uint8 HWC numpy frames.
        model_path:     Local .pth file, HuggingFace repo ID, or
                        ``repo_id:filename`` for a custom weight file.
                        None → default ``ai-forever/Real-ESRGAN`` with
                        ``RealESRGAN_x4.pth``.
        scale:          Desired final upscaling factor (e.g. 2, 3, 4).
                        The 4× model is used internally; the output is
                        resized to match *scale* when it differs.
        half_precision: Use fp16 inference (faster on supported GPUs).

    Returns:
        List of upscaled uint8 HWC numpy frames.
    """
    upscaler = ImageUpscaler(
        model_path=model_path,
        scale=scale,
        half_precision=half_precision,
    )
    return upscaler.upscale(frames)


def batch_upscale_frames(
    frames: list[np.ndarray],
    model_path: Optional[str] = None,
    scale: int = 4,
) -> list[np.ndarray]:
    """
    Batched Real-ESRGAN upscaling for realtime video paths.

    The default ``upscale_frames`` API intentionally keeps its original
    per-frame behavior. Call this helper only when the caller can tolerate
    batched execution and same-shape grouping semantics.
    """
    upscaler = ImageUpscaler(
        model_path=model_path,
        scale=scale,
        half_precision=current_platform.is_cuda(),
    )
    return upscaler.upscale_batched(frames)
