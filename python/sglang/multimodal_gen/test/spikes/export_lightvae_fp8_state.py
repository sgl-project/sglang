#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export a calibrated LightVAE FP8 encoder state for OmniDreams native VAE.

Ported from FlashDreams ``integrations/omnidreams/scripts/export_lightvae_fp8_state.py``,
retargeted at SGLang's ``LightVAEEncoder`` so the hooked module paths match what
``build_lightvae_encoder_fp8_staged_state`` looks up (``encoder.downsamples.0.``
etc.).

Run on the GPU host::

    python test/spikes/export_lightvae_fp8_state.py \
        --ckpt /path/to/lightvaew2_1.pth \
        --out /path/to/lightvae_fp8_state.pt \
        --calibration-video /path/to/video.mp4 \
        --device cuda --height 720 --width 1280 --frames 13
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

# Matches the FlashDreams reference exactly.
VAE_FP8_VERSION_KEY = "__omnidreams_vae_fp8_version__"
MODEL_KIND_KEY = "__omnidreams_vae_fp8_model_kind__"
STATE_SCALE_MAX_KEY = "__omnidreams_vae_fp8_scale_max__"
MODEL_KIND_LIGHTVAE_ENCODER = 1

DEFAULT_LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
DEFAULT_LATENTS_STD = [
    2.8184, 2.7649, 2.5679, 0.9361, 0.9143, 0.8458, 0.8740, 0.9163,
    0.8754, 0.8767, 0.8802, 0.9088, 0.8728, 0.9371, 0.8904, 0.8554,
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt", type=Path, required=True,
        help="LightVAE encoder checkpoint (lightvaew2_1.pth).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .pt path.")
    parser.add_argument(
        "--calibration-video", type=Path, required=True,
        help="Video for per-channel activation amax collection.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--scale-max", type=float, default=24.0)
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# FP8 quantization (verbatim from the FlashDreams reference)                  #
# --------------------------------------------------------------------------- #


def _require_float8() -> torch.dtype:
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise RuntimeError("PyTorch float8_e4m3fn is required for FP8 state export")
    return dtype


def _scale_view_shape(tensor: torch.Tensor, channel_dim: int) -> tuple[int, ...]:
    return tuple(
        tensor.shape[i] if i == channel_dim else 1 for i in range(tensor.dim())
    )


def _quantize_fp8_per_channel(
    tensor: torch.Tensor,
    *,
    channel_dim: int = 0,
    scale_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not torch.is_floating_point(tensor):
        raise TypeError(f"expected floating tensor, got {tensor.dtype}")
    if tensor.dim() == 0:
        raise ValueError("per-channel quantization requires a non-scalar tensor")
    if scale_max <= 0:
        raise ValueError(f"scale_max must be positive, got {scale_max}")

    fp8_dtype = _require_float8()
    if channel_dim < 0:
        channel_dim += tensor.dim()
    reduce_dims = tuple(i for i in range(tensor.dim()) if i != channel_dim)
    tensor_fp32 = tensor.detach().float()
    amax = tensor_fp32.abs().amax(dim=reduce_dims) if reduce_dims else tensor_fp32.abs()
    scale = (amax / float(scale_max)).clamp(min=1.0e-6)
    scaled = tensor_fp32 / scale.reshape(_scale_view_shape(tensor, channel_dim))
    return scaled.to(fp8_dtype).contiguous().view(torch.uint8), scale.to(torch.float16)


def _channel_amax(value: torch.Tensor, channel_dim: int) -> torch.Tensor:
    reduce_dims = tuple(dim for dim in range(value.dim()) if dim != channel_dim)
    return value.detach().float().abs().amax(dim=reduce_dims).cpu()


# --------------------------------------------------------------------------- #
# Activation amax collection (retargeted at SGLang's LightVAEEncoder)         #
# --------------------------------------------------------------------------- #


def _collect_activation_amax(
    model: nn.Module,
    video_bcthw: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run one streaming encode over ``video_bcthw``, recording per-channel amax
    at every module output + every ``cache_step`` output.

    SGLang's ``LightVAEEncoder.encode()`` manages its own cache internally
    (no ``cache=`` kwarg), so we wrap ``cache_step`` on the encoder's
    `Encaps3d` submodules to collect intermediate activation tensor stats.
    """
    if video_bcthw.dim() != 5:
        raise ValueError(
            f"expected calibration video [B,C,T,H,W], got {tuple(video_bcthw.shape)}"
        )

    amax: dict[str, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    cache_step_originals: list[tuple[nn.Module, Any]] = []

    def record(name: str, value: torch.Tensor) -> None:
        if value.dim() not in (4, 5):
            return
        current = _channel_amax(value, 1)
        previous = amax.get(name)
        amax[name] = current if previous is None else torch.maximum(previous, current)

    def hook(name: str):
        def _hook(
            _module: nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: object,
        ) -> None:
            if isinstance(output, torch.Tensor):
                record(name, output)

        return _hook

    def pre_hook(name: str):
        def _pre_hook(
            _module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            if inputs and isinstance(inputs[0], torch.Tensor):
                record(name, inputs[0])

        return _pre_hook

    # Seed the input entry so the staged-state builder always finds a scale.
    record("encoder.conv1.input", video_bcthw)
    record("encoder.input", video_bcthw)
    record("input", video_bcthw)

    # Walk the SGLang LightVAEEncoder module tree (not a FlashDreams VAE wrapper).
    encoder = model.encoder
    for name, module in encoder.named_modules():
        if not name:
            continue
        full = f"encoder.{name}"
        handles.append(module.register_forward_hook(hook(full)))
        if name == "middle.1.proj":
            handles.append(
                module.register_forward_pre_hook(
                    pre_hook("encoder.middle.1.sdpa")
                )
            )
        if hasattr(module, "cache_step"):
            handles.append(module.register_forward_hook(hook(full)))

    # Also hook model.conv1 (post quant conv) via named_modules on self.
    for name, module in model.named_modules():
        if "encoder." in name:
            continue  # already covered above
        if not name:
            continue
        handles.append(module.register_forward_hook(hook(name)))

    try:
        model.encode(video_bcthw)  # single-pass; fresh cache internally
    finally:
        for h in handles:
            h.remove()

    return amax


def _activation_scales(
    amax: Mapping[str, torch.Tensor],
    *,
    scale_max: float,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, value in amax.items():
        out[f"{name}.activation_scale"] = (
            (value.float().abs() / float(scale_max))
            .clamp(min=1.0e-6)
            .to(torch.float16)
            .contiguous()
        )
    return out


# --------------------------------------------------------------------------- #
# FP8 state builder (verbatim from the FlashDreams reference)                 #
# --------------------------------------------------------------------------- #


def _build_fp8_state(
    state_dict: Mapping[str, torch.Tensor],
    activation_scales: Mapping[str, torch.Tensor],
    *,
    scale_max: float,
) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        VAE_FP8_VERSION_KEY: torch.tensor([1], dtype=torch.int32),
        MODEL_KIND_KEY: torch.tensor([MODEL_KIND_LIGHTVAE_ENCODER], dtype=torch.int32),
        STATE_SCALE_MAX_KEY: torch.tensor([float(scale_max)], dtype=torch.float32),
    }

    for name, tensor in state_dict.items():
        if (
            name.endswith(".weight")
            and torch.is_floating_point(tensor)
            and tensor.dim() >= 2
        ):
            q, scale = _quantize_fp8_per_channel(
                tensor.detach(),
                channel_dim=0,
                scale_max=scale_max,
            )
            state[name] = q.cpu()
            state[name.replace(".weight", ".weight_scale")] = scale.cpu()
        elif torch.is_floating_point(tensor):
            state[name] = (
                tensor.detach().to(dtype=torch.float16, device="cpu").contiguous()
            )
        else:
            state[name] = tensor.detach().cpu().contiguous()

    for name, scale in activation_scales.items():
        if scale.dim() != 1:
            raise ValueError(
                f"{name} must be a 1D scale tensor, got {tuple(scale.shape)}"
            )
        state[name] = scale.detach().to(dtype=torch.float16, device="cpu").contiguous()
    return state


# --------------------------------------------------------------------------- #
# Video loader                                                               #
# --------------------------------------------------------------------------- #


def _load_video_prefix_bcthw(
    path: Path,
    *,
    frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to load calibration video frames"
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {path}")
    images: list[torch.Tensor] = []
    while len(images) < frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (height, width):
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        images.append(torch.from_numpy(rgb).permute(2, 0, 1).contiguous())
    cap.release()
    if len(images) < frames:
        raise RuntimeError(f"{path} has {len(images)} readable frames; need {frames}")
    video = (
        torch.stack(images, dim=1).unsqueeze(0).to(device=device, dtype=torch.float16)
    )
    return (video / 127.5 - 1.0).contiguous()


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is false"
        )

    from sglang.multimodal_gen.runtime.models.vaes.omnidreams_light_vae import (
        LightVAEEncoder,
    )

    # Instantiate SGLang's LightVAEEncoder on the target device.
    encoder: nn.Module = LightVAEEncoder(
        checkpoint_path=str(args.ckpt),
        latents_mean=DEFAULT_LATENTS_MEAN,
        latents_std=DEFAULT_LATENTS_STD,
        dtype=torch.float16,
    ).to(device).eval()

    video_path = args.calibration_video.expanduser().resolve()
    video_bcthw = _load_video_prefix_bcthw(
        video_path,
        frames=args.frames,
        height=args.height,
        width=args.width,
        device=device,
    )
    amax = _collect_activation_amax(encoder, video_bcthw)
    state = _build_fp8_state(
        encoder.state_dict(),
        _activation_scales(amax, scale_max=args.scale_max),
        scale_max=args.scale_max,
    )

    args.out.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.out)
    print(f"Wrote {args.out}")
    print(f"Calibration video: {video_path}")
    print(
        f"Activation scales: "
        f"{len([k for k in state if k.endswith('.activation_scale')])}"
    )


if __name__ == "__main__":
    main()
