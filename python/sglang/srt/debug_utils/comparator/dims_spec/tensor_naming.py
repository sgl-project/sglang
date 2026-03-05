from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.debug_utils.comparator.dims_spec.types import DimSpec


def find_dim_index(dim_specs: list[DimSpec], name: str) -> Optional[int]:
    """Find index by name. Accepts both ``*``-form and ``___``-form for fused dims."""
    for i, spec in enumerate(dim_specs):
        if spec.name == name or spec.sanitized_name == name:
            return i
    return None


def resolve_dim_by_name(tensor: torch.Tensor, name: str) -> int:
    if tensor.names[0] is None:
        raise ValueError(f"Tensor has no names, cannot resolve {name!r}")

    names: tuple[Optional[str], ...] = tensor.names
    try:
        return list(names).index(name)
    except ValueError:
        raise ValueError(f"Dim name {name!r} not in tensor names {names}")


def apply_dim_names(tensor: torch.Tensor, dim_names: list[str]) -> torch.Tensor:
    if tensor.ndim != len(dim_names):
        raise ValueError(
            f"dims metadata mismatch: tensor has {tensor.ndim} dims (shape {list(tensor.shape)}) "
            f"but dims string specifies {len(dim_names)} names {dim_names}. "
            f"Please fix the dims string in the dumper.dump() call to match the actual tensor shape."
        )
    return tensor.refine_names(*dim_names)


def strip_dim_names(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.rename(None)
