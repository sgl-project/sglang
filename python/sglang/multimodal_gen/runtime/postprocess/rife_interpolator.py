# SPDX-License-Identifier: Apache-2.0
"""
RIFE 4.22.lite frame interpolation for SGLang diffusion pipelines.

RIFE model code is vendored and adapted from:
  - https://github.com/hzwer/ECCV2022-RIFE  (MIT License)
  - https://github.com/hzwer/Practical-RIFE  (MIT License)
  Copyright (c) 2021 Zhewei Huang

The FrameInterpolator wrapper and integration code are original work.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Default HuggingFace repo for RIFE 4.22.lite weights
_DEFAULT_RIFE_HF_REPO = "elfgum/RIFE-4.22.lite"

# Module-level cache: model_path -> Model instance
_MODEL_CACHE: dict[str, "Model"] = {}


# ---------------------------------------------------------------------------
# Vendored RIFE 4.22.lite model code
# (IFBlock, IFNet_HDv3 backbone, Model wrapper)
# ---------------------------------------------------------------------------


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Warp tenInput by tenFlow using grid_sample."""
    # Build base grid for the current size
    tenHorizontal = (
        torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=tenFlow.device)
        .view(1, 1, 1, tenFlow.shape[3])
        .expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
    )
    tenVertical = (
        torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=tenFlow.device)
        .view(1, 1, tenFlow.shape[2], 1)
        .expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
    )
    tenGrid = torch.cat([tenHorizontal, tenVertical], dim=1)

    tenFlow = torch.cat(
        [
            tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
            tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0),
        ],
        dim=1,
    )

    grid = (tenGrid + tenFlow).permute(0, 2, 3, 1)
    return F.grid_sample(
        input=tenInput,
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    """Conv2d + LeakyReLU helper (matches RIFE 4.22 conv())."""
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, True),
    )


class ResConv(nn.Module):
    """Residual convolution block with learnable beta scaling (RIFE 4.22)."""

    def __init__(self, c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x) * self.beta + x)


class IFBlock(nn.Module):
    """Single-scale optical flow + mask + feature block (RIFE 4.22)."""

    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_planes, c // 2, 3, 2, 1),
            _conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4 * 13, 4, 2, 1),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        x: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.interpolate(
            x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
        )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False,
                )
                * 1.0
                / scale
            )
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(
            tmp, scale_factor=scale, mode="bilinear", align_corners=False
        )
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        feat = tmp[:, 5:]
        return flow, mask, feat


class Head(nn.Module):
    """Feature encoder producing 4-channel features at full resolution (RIFE 4.22)."""

    def __init__(self):
        super().__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        return x3


class IFNet(nn.Module):
    """4-scale IFNet optical flow network (RIFE 4.22 backbone)."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 8, c=192)
        self.block1 = IFBlock(8 + 4 + 8 + 8, c=128)
        self.block2 = IFBlock(8 + 4 + 8 + 8, c=64)
        self.block3 = IFBlock(8 + 4 + 8 + 8, c=32)
        self.encode = Head()

    def forward(
        self,
        x: torch.Tensor,
        timestep: float = 0.5,
        scale_list: Optional[list] = None,
    ) -> tuple[list, torch.Tensor, list]:
        if scale_list is None:
            scale_list = [8, 4, 2, 1]

        channel = x.shape[1] // 2
        img0 = x[:, :channel]
        img1 = x[:, channel:]

        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])

        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])

        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None

        block = [self.block0, self.block1, self.block2, self.block3]
        for i in range(4):
            if flow is None:
                flow, mask, feat = block[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1, timestep), 1),
                    None,
                    scale=scale_list[i],
                )
            else:
                wf0 = warp(f0, flow[:, :2])
                wf1 = warp(f1, flow[:, 2:4])
                fd, m0, feat = block[i](
                    torch.cat(
                        (
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            wf0,
                            wf1,
                            timestep,
                            mask,
                            feat,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )
                mask = m0
                flow = flow + fd

            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))

        mask = torch.sigmoid(mask)
        merged[3] = warped_img0 * mask + warped_img1 * (1 - mask)

        return flow_list, mask_list[3], merged


class Model:
    """Wraps IFNet, provides load_model() and inference() API."""

    def __init__(self):
        self.flownet = IFNet()
        self.device_type: str = "cpu"

    def eval(self) -> "Model":
        self.flownet.eval()
        return self

    def device(self) -> torch.device:
        return next(self.flownet.parameters()).device

    def load_model(self, path: str, strip_module_prefix: bool = True) -> None:
        """Load weights from {path}/flownet.pkl.

        Args:
            path: Directory containing ``flownet.pkl``.
            strip_module_prefix: If True, strip the ``module.`` prefix that
                ``DataParallel`` / ``DistributedDataParallel`` adds to keys.
        """
        flownet_path = os.path.join(path, "flownet.pkl")
        if not os.path.isfile(flownet_path):
            raise FileNotFoundError(
                f"RIFE weight file not found: {flownet_path}\n"
                "Expected layout: <model_path>/flownet.pkl"
            )

        def convert(param):
            if strip_module_prefix:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return {k: v for k, v in param.items() if "module." not in k}

        state = torch.load(flownet_path, map_location="cpu", weights_only=False)
        self.flownet.load_state_dict(convert(state), strict=False)
        logger.info("Loaded RIFE weights from %s", flownet_path)

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        scale: float = 1.0,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        """Interpolate a single intermediate frame between img0 and img1."""
        n, c, h, w = img0.shape

        # Pad to multiples of 32 so that RIFE's downsample/upsample round-trips
        # preserve spatial dimensions exactly.
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        pad = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, pad)
        img1 = F.pad(img1, pad)

        imgs = torch.cat((img0, img1), 1)
        scale_list = [8 / scale, 4 / scale, 2 / scale, 1 / scale]
        with torch.no_grad():
            flow_list, mask, merged = self.flownet(
                imgs,
                timestep=timestep,
                scale_list=scale_list,
            )

        # Crop back to original resolution
        return merged[3][:, :, :h, :w]


# ---------------------------------------------------------------------------
# FrameInterpolator public class
# ---------------------------------------------------------------------------


class FrameInterpolator:
    """
    Lazy-loaded RIFE 4.22.lite frame interpolator.

    Weights are loaded on first call to `.interpolate()` and cached globally
    per model_path to avoid reloading across requests.
    """

    def __init__(self, model_path: Optional[str] = None):
        self._model_path = model_path
        self._resolved_path: Optional[str] = None

    def _ensure_model_loaded(self) -> Model:
        """Load RIFE model weights.

        Accepts a local directory **or** a HuggingFace repo ID.  When *None*
        (the default) the weights are downloaded (and cached) automatically
        from ``elfgum/RIFE-4.22.lite`` via ``maybe_download_model()``.
        """
        from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
            maybe_download_model,
        )

        model_path = self._model_path or _DEFAULT_RIFE_HF_REPO

        # Resolve: local path pass-through, HF repo ID → download & cache
        model_path = maybe_download_model(model_path)

        self._resolved_path = model_path

        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]

        device = current_platform.get_local_torch_device()
        model = Model()
        model.load_model(model_path, strip_module_prefix=True)
        model.eval()
        model.flownet = model.flownet.to(device)
        _MODEL_CACHE[model_path] = model
        logger.info("RIFE model loaded on device: %s", device)
        return model

    @staticmethod
    def _frame_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
        """Convert uint8 HWC numpy frame to float32 CHW tensor on device."""
        t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return t.to(device)

    @staticmethod
    def _tensor_to_frame(t: torch.Tensor) -> np.ndarray:
        """Convert float32 CHW tensor (batch=1) to uint8 HWC numpy frame."""
        arr = t.squeeze(0).permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
        return (arr * 255.0).astype(np.uint8)

    def _make_inference(
        self, model: Model, I0: torch.Tensor, I1: torch.Tensor, n: int, scale: float
    ) -> list[torch.Tensor]:
        """
        Recursively generate n-1 intermediate frames between I0 and I1.

        Returns a list of intermediate frame tensors (not including I0 or I1).
        """
        if n == 1:
            return [model.inference(I0, I1, scale=scale)]
        mid = model.inference(I0, I1, scale=scale)
        return (
            self._make_inference(model, I0, mid, n // 2, scale)
            + [mid]
            + self._make_inference(model, mid, I1, n // 2, scale)
        )

    def interpolate(
        self,
        frames: list[np.ndarray],
        exp: int = 1,
        scale: float = 1.0,
    ) -> tuple[list[np.ndarray], int]:
        """
        Interpolate frames using RIFE.

        Args:
            frames: List of uint8 numpy arrays with shape [H, W, 3].
            exp:    Exponent for interpolation factor. 1 → 2×, 2 → 4×.
            scale:  RIFE inference scale. Use 0.5 for high-resolution inputs.

        Returns:
            (interpolated_frames, multiplier) where multiplier = 2**exp.
        """
        if len(frames) < 2:
            logger.warning(
                "Frame interpolation requires at least 2 frames; returning input unchanged."
            )
            return frames, 1

        model = self._ensure_model_loaded()
        device = model.device()

        n_intermediate = 2**exp // 2  # intermediates per adjacent pair

        result: list[np.ndarray] = []
        for i in range(len(frames) - 1):
            I0 = self._frame_to_tensor(frames[i], device)
            I1 = self._frame_to_tensor(frames[i + 1], device)

            intermediate_tensors = self._make_inference(
                model, I0, I1, n_intermediate, scale
            )

            result.append(frames[i])
            for t in intermediate_tensors:
                result.append(self._tensor_to_frame(t))

        result.append(frames[-1])
        multiplier = 2**exp
        return result, multiplier


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def interpolate_video_frames(
    frames: list[np.ndarray],
    exp: int = 1,
    scale: float = 1.0,
    model_path: Optional[str] = None,
) -> tuple[list[np.ndarray], int]:
    """
    Convenience wrapper around FrameInterpolator.

    Args:
        frames:     List of uint8 HWC numpy frames.
        exp:        Interpolation exponent (1=2×, 2=4×).
        scale:      RIFE inference scale (default 1.0; use 0.5 for high-res).
        model_path: Local directory or HuggingFace repo ID containing
                    ``flownet.pkl``.  *None* → default ``elfgum/RIFE-4.22.lite``.

    Returns:
        (interpolated_frames, multiplier)
    """
    interpolator = FrameInterpolator(model_path=model_path)
    return interpolator.interpolate(frames, exp=exp, scale=scale)
