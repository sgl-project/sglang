# SPDX-License-Identifier: Apache-2.0
"""Shared checkpoint loading helpers for DreamZero components."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import torch
from safetensors.torch import safe_open
from torch import nn


@dataclass
class DreamZeroCheckpointLoadReport:
    loaded_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]

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


ReportT = TypeVar("ReportT", bound=DreamZeroCheckpointLoadReport)


def assign_tensor(model: nn.Module, name: str, tensor: torch.Tensor) -> None:
    parent_name, _, leaf_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if leaf_name in parent._parameters:
        parent._parameters[leaf_name] = nn.Parameter(tensor, requires_grad=False)
        return
    if leaf_name in parent._buffers:
        parent._buffers[leaf_name] = tensor
        return
    raise KeyError(f"{name} is neither a parameter nor a buffer")


def iter_indexed_safetensors(
    model_path: str | os.PathLike[str],
    *,
    index_name: str = "model.safetensors.index.json",
    prefix_filter: str | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    model_dir = Path(model_path)
    index_path = model_dir / index_name
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    files: list[str] = []
    seen_files: set[str] = set()
    for key, file_name in weight_map.items():
        if prefix_filter is not None and not key.startswith(prefix_filter):
            continue
        if file_name in seen_files:
            continue
        seen_files.add(file_name)
        files.append(file_name)

    for file_name in files:
        file_path = model_dir / file_name
        yield from iter_safetensor_file(file_path, prefix_filter=prefix_filter)


def iter_safetensor_file(
    file_path: str | os.PathLike[str],
    *,
    prefix_filter: str | None = None,
) -> Iterator[tuple[str, torch.Tensor]]:
    with safe_open(file_path, framework="pt", device="cpu") as handle:
        for checkpoint_key in handle.keys():
            if prefix_filter is None or checkpoint_key.startswith(prefix_filter):
                yield checkpoint_key, handle.get_tensor(checkpoint_key)


def iter_prefixed_safetensors(
    model_path: str | os.PathLike[str],
    prefix: str,
    *,
    index_name: str = "model.safetensors.index.json",
) -> Iterator[tuple[str, torch.Tensor]]:
    yield from iter_indexed_safetensors(
        model_path,
        index_name=index_name,
        prefix_filter=prefix,
    )


def load_matching_tensors(
    model: nn.Module,
    tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    device: torch.device,
    key_mapper: Callable[[str], str | None] | None = None,
    report_cls: type[ReportT],
) -> ReportT:
    meta_sd = model.state_dict()
    loaded_keys: list[str] = []
    unexpected_keys: list[str] = []
    shape_mismatches: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    with torch.no_grad():
        for checkpoint_key, full_tensor in tensors:
            target_name = key_mapper(checkpoint_key) if key_mapper else checkpoint_key
            if target_name is None:
                continue
            meta_tensor = meta_sd.get(target_name)
            if meta_tensor is None:
                unexpected_keys.append(target_name)
                continue
            if tuple(full_tensor.shape) != tuple(meta_tensor.shape):
                if meta_tensor.ndim == 0 and full_tensor.numel() == 1:
                    full_tensor = full_tensor.reshape(())
                else:
                    shape_mismatches[target_name] = (
                        tuple(meta_tensor.shape),
                        tuple(full_tensor.shape),
                    )
                    continue
            target_tensor = full_tensor.to(device=device, dtype=meta_tensor.dtype)
            assign_tensor(model, target_name, target_tensor)
            loaded_keys.append(target_name)

    return report_cls(
        loaded_keys=loaded_keys,
        missing_keys=sorted(set(meta_sd) - set(loaded_keys)),
        unexpected_keys=unexpected_keys,
        shape_mismatches=shape_mismatches,
    )


def raise_for_strict_report(
    report: DreamZeroCheckpointLoadReport,
    *,
    strict: bool,
    error_prefix: str,
) -> None:
    if strict and (
        report.missing_keys or report.unexpected_keys or report.shape_mismatches
    ):
        raise RuntimeError(f"{error_prefix}: {report.as_dict()}")
