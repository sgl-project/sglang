# SPDX-License-Identifier: Apache-2.0
"""DreamZero text/image encoder loaders.

DreamZero-DROID stores custom Wan text encoder and open-clip XLM-RoBERTa
ViT-H weights, so these loaders instantiate the SGLang-local compatible
implementations and copy checkpoint tensors into matching state_dict keys.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open
from torch import nn

_DROID_TEXT_ENCODER_PREFIX = "action_head.text_encoder."
_DROID_IMAGE_ENCODER_PREFIX = "action_head.image_encoder."
_WAN_TEXT_ENCODER_NAME = "models_t5_umt5-xxl-enc-bf16.pth"
_WAN_IMAGE_ENCODER_NAME = "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"


@dataclass
class DreamZeroEncoderLoadReport:
    loaded_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]
    fallback_impl: str

    @property
    def loaded_count(self) -> int:
        return len(self.loaded_keys)

    @property
    def missing_count(self) -> int:
        return len(self.missing_keys)

    @property
    def unexpected_count(self) -> int:
        return len(self.unexpected_keys)

    @property
    def shape_mismatch_count(self) -> int:
        return len(self.shape_mismatches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fallback_impl": self.fallback_impl,
            "loaded_count": self.loaded_count,
            "missing_count": self.missing_count,
            "unexpected_count": self.unexpected_count,
            "shape_mismatch_count": self.shape_mismatch_count,
            "missing_keys": self.missing_keys,
            "unexpected_keys": self.unexpected_keys,
            "shape_mismatches": {
                key: {"target": target, "checkpoint": checkpoint}
                for key, (target, checkpoint) in self.shape_mismatches.items()
            },
        }


def _iter_prefixed_safetensors(
    model_path: str | os.PathLike[str],
    prefix: str,
) -> Iterator[tuple[str, torch.Tensor]]:
    model_dir = Path(model_path)
    index_path = model_dir / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    files: list[str] = []
    seen_files: set[str] = set()
    for key, file_name in weight_map.items():
        if not key.startswith(prefix) or file_name in seen_files:
            continue
        seen_files.add(file_name)
        files.append(file_name)

    for file_name in files:
        file_path = model_dir / file_name
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for checkpoint_key in handle.keys():
                if checkpoint_key.startswith(prefix):
                    yield checkpoint_key, handle.get_tensor(checkpoint_key)


def _assign_tensor(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    parent_name, _, leaf_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if leaf_name in parent._parameters:
        parent._parameters[leaf_name] = nn.Parameter(tensor, requires_grad=False)
        return
    if leaf_name in parent._buffers:
        parent._buffers[leaf_name] = tensor
        return
    raise KeyError(f"{name} is neither a parameter nor a buffer")


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
    meta_sd = model.state_dict()
    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    with torch.no_grad():
        for checkpoint_key, full_tensor in state_dict.items():
            meta_tensor = meta_sd.get(checkpoint_key)
            if meta_tensor is None:
                unexpected_keys.append(checkpoint_key)
                continue
            if tuple(full_tensor.shape) != tuple(meta_tensor.shape):
                shape_mismatches[checkpoint_key] = (
                    tuple(meta_tensor.shape),
                    tuple(full_tensor.shape),
                )
                continue
            target_tensor = full_tensor.to(device=device, dtype=meta_tensor.dtype)
            _assign_tensor(model, checkpoint_key, target_tensor)
            loaded_keys.append(checkpoint_key)

    report = DreamZeroEncoderLoadReport(
        loaded_keys=loaded_keys,
        missing_keys=sorted(set(meta_sd) - set(loaded_keys)),
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
        fallback_impl=fallback_impl,
    )
    if strict and (
        report.missing_keys or report.unexpected_keys or report.shape_mismatches
    ):
        raise RuntimeError(
            f"DreamZero encoder checkpoint load failed: {report.as_dict()}"
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
    meta_sd = model.state_dict()
    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    with torch.no_grad():
        for checkpoint_key, full_tensor in _iter_prefixed_safetensors(model_path, prefix):
            target_name = checkpoint_key[len(prefix) :]
            meta_tensor = meta_sd.get(target_name)
            if meta_tensor is None:
                unexpected_keys.append(target_name)
                continue
            if tuple(full_tensor.shape) != tuple(meta_tensor.shape):
                shape_mismatches[target_name] = (
                    tuple(meta_tensor.shape),
                    tuple(full_tensor.shape),
                )
                continue
            target_tensor = full_tensor.to(device=device, dtype=meta_tensor.dtype)
            _assign_tensor(model, target_name, target_tensor)
            loaded_keys.append(target_name)

    report = DreamZeroEncoderLoadReport(
        loaded_keys=loaded_keys,
        missing_keys=sorted(set(meta_sd) - set(loaded_keys)),
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
        fallback_impl=fallback_impl,
    )
    if strict and (
        report.missing_keys or report.unexpected_keys or report.shape_mismatches
    ):
        raise RuntimeError(
            f"DreamZero encoder checkpoint load failed: {report.as_dict()}"
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
