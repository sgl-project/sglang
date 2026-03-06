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
from typing import Optional

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

# Module-level cache: model_path -> UpscalerModel instance
_MODEL_CACHE: dict[str, "UpscalerModel"] = {}


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
        num_block = sum(
            1
            for k in state_dict
            if k.startswith("body.") and k.endswith(".rdb1.conv1.weight")
        )
        num_grow_ch = state_dict["body.0.rdb1.conv1.weight"].shape[0]
        logger.info(
            "Detected RRDBNet: num_feat=%d, num_block=%d, num_grow_ch=%d",
            num_feat,
            num_block,
            num_grow_ch,
        )
        return RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
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
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(img_t)

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
        model_path = self._model_path or _DEFAULT_REALESRGAN_HF_REPO

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


# ---------------------------------------------------------------------------
# HF download helper
# ---------------------------------------------------------------------------


def _resolve_model_path(model_path: str) -> str:
    """Return a local .pth file path.

    Accepts:
    - An existing local file path (pass-through).
    - A HuggingFace ``repo_id`` → downloads the default weight file
      (``RealESRGAN_x4.pth``).
    - A HuggingFace ``repo_id:filename`` → downloads *filename* from *repo_id*,
      allowing users to specify custom weight files hosted on HF.
    """
    if os.path.isfile(model_path):
        return model_path

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
        model_path=model_path, scale=scale, half_precision=half_precision
    )
    return upscaler.upscale(frames)
