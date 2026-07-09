# SPDX-License-Identifier: Apache-2.0
"""DreamZero text/image encoder loaders.

DreamZero-DROID stores custom Wan text encoder and open-clip XLM-RoBERTa
ViT-H weights, so these loaders instantiate the SGLang-local compatible
implementations and copy checkpoint tensors into matching state_dict keys.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from sglang.multimodal_gen.runtime.loader.component_loaders.dreamzero_checkpoint_utils import (
    DreamZeroCheckpointLoadReport,
    iter_prefixed_safetensors,
    load_matching_tensors,
    raise_for_strict_report,
)

_DROID_TEXT_ENCODER_PREFIX = "action_head.text_encoder."
_DROID_IMAGE_ENCODER_PREFIX = "action_head.image_encoder."
_WAN_TEXT_ENCODER_NAME = "models_t5_umt5-xxl-enc-bf16.pth"
_WAN_IMAGE_ENCODER_NAME = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"


class DreamZeroEncoderLoadReport(DreamZeroCheckpointLoadReport):
    include_fallback_impl = True


def _resolve_named_pth(
    model_path: str | os.PathLike[str],
    file_name: str,
) -> Path | None:
    path = Path(model_path)
    if path.is_file() and path.name == file_name:
        return path
    candidate = path / file_name
    if candidate.is_file():
        return candidate
    return None


def _load_plain_pth_checkpoint(
    model: nn.Module,
    checkpoint_path: str | os.PathLike[str],
    *,
    device: torch.device,
    strict: bool,
    fallback_impl: str,
) -> DreamZeroEncoderLoadReport:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    report = load_matching_tensors(
        model,
        state_dict.items(),
        device=device,
        report_cls=DreamZeroEncoderLoadReport,
        fallback_impl=fallback_impl,
    )
    raise_for_strict_report(
        report,
        strict=strict,
        error_prefix="DreamZero encoder checkpoint load failed",
    )
    return report


def _load_prefixed_checkpoint(
    model: nn.Module,
    model_path: str | os.PathLike[str],
    *,
    prefix: str,
    device: torch.device,
    strict: bool,
    fallback_impl: str,
) -> DreamZeroEncoderLoadReport:
    report = load_matching_tensors(
        model,
        iter_prefixed_safetensors(model_path, prefix),
        device=device,
        key_mapper=lambda checkpoint_key: checkpoint_key[len(prefix) :],
        report_cls=DreamZeroEncoderLoadReport,
        fallback_impl=fallback_impl,
    )
    raise_for_strict_report(
        report,
        strict=strict,
        error_prefix="DreamZero encoder checkpoint load failed",
    )
    return report


def build_dreamzero_text_encoder(
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    from sglang.multimodal_gen.runtime.models.encoders.dreamzero_text import (
        WanTextEncoder,
    )

    with torch.device("meta"):
        model = WanTextEncoder().to(dtype=dtype)
    model.eval()
    return model


def build_dreamzero_image_encoder(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | None = None,
) -> nn.Module:
    from sglang.multimodal_gen.runtime.models.encoders.dreamzero_image import (
        WanImageEncoder,
    )

    model = WanImageEncoder().to(dtype=dtype)
    if device is not None:
        model = model.to(device=device)
    model.eval()
    return model


def load_dreamzero_text_encoder_checkpoint(
    model: nn.Module,
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    strict: bool = False,
) -> DreamZeroEncoderLoadReport:
    pth_path = _resolve_named_pth(model_path, _WAN_TEXT_ENCODER_NAME)
    if pth_path is not None:
        return _load_plain_pth_checkpoint(
            model,
            pth_path,
            device=device,
            strict=strict,
            fallback_impl="sglang.WanTextEncoder",
        )
    return _load_prefixed_checkpoint(
        model,
        model_path,
        prefix=_DROID_TEXT_ENCODER_PREFIX,
        device=device,
        strict=strict,
        fallback_impl="sglang.WanTextEncoder",
    )


def load_dreamzero_image_encoder_checkpoint(
    model: nn.Module,
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    strict: bool = False,
) -> DreamZeroEncoderLoadReport:
    pth_path = _resolve_named_pth(model_path, _WAN_IMAGE_ENCODER_NAME)
    if pth_path is not None:
        return _load_plain_pth_checkpoint(
            model.model,
            pth_path,
            device=device,
            strict=False,
            fallback_impl="sglang.WanImageEncoder",
        )
    return _load_prefixed_checkpoint(
        model,
        model_path,
        prefix=_DROID_IMAGE_ENCODER_PREFIX,
        device=device,
        strict=strict,
        fallback_impl="sglang.WanImageEncoder",
    )


def build_dreamzero_text_encoder_from_checkpoint(
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> tuple[nn.Module, DreamZeroEncoderLoadReport]:
    model = build_dreamzero_text_encoder(dtype=dtype)
    report = load_dreamzero_text_encoder_checkpoint(
        model,
        model_path,
        device=device,
        strict=strict,
    )
    return model, report


def build_dreamzero_image_encoder_from_checkpoint(
    model_path: str | os.PathLike[str],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    strict: bool = True,
) -> tuple[nn.Module, DreamZeroEncoderLoadReport]:
    model = build_dreamzero_image_encoder(dtype=dtype, device=device)
    report = load_dreamzero_image_encoder_checkpoint(
        model,
        model_path,
        device=device,
        strict=strict,
    )
    return model, report
