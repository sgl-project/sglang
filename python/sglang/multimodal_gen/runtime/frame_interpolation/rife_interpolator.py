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

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DEFAULT_HF_REPO_ID = "hzwer/ECCV2022-RIFE"

# Module-level cache: model_path -> Model instance
_MODEL_CACHE: dict[str, "Model"] = {}


# ---------------------------------------------------------------------------
# Vendored RIFE 4.22.lite model code
# (IFBlock, IFNet_HDv3 backbone, Model wrapper)
# ---------------------------------------------------------------------------


def warp(tenInput: torch.Tensor, tenFlow: torch.Tensor) -> torch.Tensor:
    """Warp tenInput by tenFlow using grid_sample."""
    k = (str(tenFlow.device), str(tenFlow.size()))
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


class IFBlock(nn.Module):
    """Single-scale optical flow + alpha blending block (RIFE IFBlock)."""

    def __init__(self, in_planes: int, c: int = 64):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_planes, c // 2, 3, 2, 1),
            nn.PReLU(c // 2),
            nn.Conv2d(c // 2, c, 3, 2, 1),
            nn.PReLU(c),
        )
        self.convblock = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.PReLU(c),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.PReLU(c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)

    def forward(
        self, x: torch.Tensor, flow: torch.Tensor, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scale != 1:
            x = F.interpolate(
                x, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
            )
        if flow is not None:
            flow = (
                F.interpolate(
                    flow, scale_factor=1.0 / scale, mode="bilinear", align_corners=False
                )
                / scale
            )
            x = torch.cat((x, flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(
            tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False
        )
        flow = tmp[:, :4] * scale * 2
        mask = tmp[:, 4:5]
        return flow, mask


class IFNet(nn.Module):
    """3-scale IFNet_HDv3 optical flow network (RIFE backbone)."""

    def __init__(self):
        super().__init__()
        self.block0 = IFBlock(7 + 4, c=192)
        self.block1 = IFBlock(8 + 4 + 4, c=128)
        self.block2 = IFBlock(8 + 4 + 4, c=96)
        self.block_tea = IFBlock(10 + 4 + 4, c=64)
        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.PReLU(16),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.PReLU(16),
        )

    def forward(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        timestep: float = 0.5,
        scale_list: Optional[list] = None,
        training: bool = False,
        fastmode: bool = True,
        ensemble: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list]:
        if scale_list is None:
            scale_list = [8, 4, 2, 1]
        channel = img0.shape[1]
        img0 = img0.float()
        img1 = img1.float()
        f0 = self.encode(img0[:, :3])
        f1 = self.encode(img1[:, :3])
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None
        mask = None
        loss_distill = 0
        stu = [self.block0, self.block1, self.block2]
        for i in range(3):
            if flow is not None:
                flow_d, mask_d = stu[i](
                    torch.cat(
                        (
                            img0[:, :3],
                            img1[:, :3],
                            warped_img0[:, :3],
                            warped_img1[:, :3],
                            mask,
                        ),
                        1,
                    ),
                    flow,
                    scale=scale_list[i],
                )
                flow = flow + flow_d
                mask = mask + mask_d
            else:
                flow, mask = stu[i](
                    torch.cat((img0[:, :3], img1[:, :3], f0, f1), 1),
                    None,
                    scale=scale_list[i],
                )
            mask_list.append(torch.sigmoid(mask))
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append(
                (warped_img0 * mask_list[i] + warped_img1 * (1 - mask_list[i]))[:, :3]
            )
        if not fastmode:
            c0 = self.encode(warped_img0[:, :3])
            c1 = self.encode(warped_img1[:, :3])
            flow_d, mask_d = self.block_tea(
                torch.cat(
                    (
                        img0[:, :3],
                        img1[:, :3],
                        warped_img0[:, :3],
                        warped_img1[:, :3],
                        mask,
                        merged[2],
                    ),
                    1,
                ),
                flow,
                scale=scale_list[3],
            )
            flow = flow + flow_d
            mask = mask + mask_d
            mask = torch.sigmoid(mask)
            merged.append(
                (
                    warp(img0, flow[:, :2]) * mask
                    + warp(img1, flow[:, 2:4]) * (1 - mask)
                )[:, :3]
            )

        return flow_list, mask_list[2], merged, loss_distill


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

    def load_model(self, path: str, rank: int = 0) -> None:
        """Load weights from {path}/flownet.pkl."""
        flownet_path = os.path.join(path, "flownet.pkl")
        if not os.path.isfile(flownet_path):
            raise FileNotFoundError(
                f"RIFE weight file not found: {flownet_path}\n"
                "Expected layout: <model_path>/flownet.pkl"
            )

        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return {k: v for k, v in param.items() if "module." not in k}

        state = torch.load(flownet_path, map_location="cpu", weights_only=False)
        self.flownet.load_state_dict(convert(state))
        logger.info("Loaded RIFE weights from %s", flownet_path)

    def inference(
        self,
        img0: torch.Tensor,
        img1: torch.Tensor,
        scale: float = 1.0,
        timestep: float = 0.5,
    ) -> torch.Tensor:
        """Interpolate a single intermediate frame between img0 and img1."""
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        with torch.no_grad():
            flow_list, mask, merged, _ = self.flownet(
                imgs[:, :3],
                imgs[:, 3:6],
                timestep=timestep,
                scale_list=scale_list,
            )
        return merged[2]


# ---------------------------------------------------------------------------
# FrameInterpolator public class
# ---------------------------------------------------------------------------


class FrameInterpolator:
    """
    Lazy-loaded RIFE 4.22.lite frame interpolator.

    Weights are loaded on first call to `.interpolate()` and cached globally
    per model_path to avoid reloading across requests.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hf_repo_id: str = _DEFAULT_HF_REPO_ID,
    ):
        self._model_path = model_path
        self._hf_repo_id = hf_repo_id
        self._resolved_path: Optional[str] = None

    def _ensure_model_loaded(self) -> Model:
        """Resolve weight path (downloading from HF if needed) and load model."""
        model_path = self._model_path

        # Auto-download from HuggingFace Hub when no valid local path is given
        if model_path is None or not os.path.isdir(model_path):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for automatic RIFE weight download. "
                    "Install it with: pip install huggingface_hub"
                )
            logger.info(
                "Downloading RIFE weights from HuggingFace Hub: %s", self._hf_repo_id
            )
            flownet_path = hf_hub_download(
                repo_id=self._hf_repo_id,
                filename="train_log/flownet.pkl",
            )
            model_path = os.path.dirname(flownet_path)
            logger.info("RIFE weights cached at: %s", model_path)

        self._resolved_path = model_path

        # Check module-level cache
        if model_path in _MODEL_CACHE:
            return _MODEL_CACHE[model_path]

        # Load and cache
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Model()
        model.load_model(model_path, rank=0)
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
    hf_repo_id: str = _DEFAULT_HF_REPO_ID,
) -> tuple[list[np.ndarray], int]:
    """
    Convenience wrapper around FrameInterpolator.

    Args:
        frames:     List of uint8 HWC numpy frames.
        exp:        Interpolation exponent (1=2×, 2=4×).
        scale:      RIFE inference scale (default 1.0; use 0.5 for high-res).
        model_path: Local directory containing flownet.pkl. None → HF auto-download.
        hf_repo_id: HuggingFace repo to download from (default: hzwer/ECCV2022-RIFE).

    Returns:
        (interpolated_frames, multiplier)
    """
    interpolator = FrameInterpolator(model_path=model_path, hf_repo_id=hf_repo_id)
    return interpolator.interpolate(frames, exp=exp, scale=scale)
