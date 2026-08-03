"""Explicit expert-tensor contracts shared by DWDP weight backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from torch import nn


@dataclass(frozen=True)
class DwdpTensorSchema:
    """Describe how an MoE quantization method stores expert-indexed tensors.

    ``partitioned`` tensors are consumed as rank-ordered partitions by the IPC
    backend. ``replicated`` tensors are gathered once during setup. Main
    weights must be listed in ``main_weights`` so the VMM backend can stage
    only the high-volume tensors while replicating smaller metadata.
    """

    main_weights: Tuple[str, ...] = ("w13_weight", "w2_weight")
    partitioned: Tuple[str, ...] = ("w13_weight", "w2_weight")
    replicated: Tuple[str, ...] = ()

    def validate(self, layer: nn.Module) -> None:
        missing = [name for name in self.partitioned if not hasattr(layer, name)]
        if missing:
            raise RuntimeError(
                f"{type(layer).__name__} is missing DWDP partitioned tensors: "
                f"{missing}"
            )
        unknown_main = set(self.main_weights) - set(self.partitioned)
        if unknown_main:
            raise ValueError(
                "DWDP main weights must also be partitioned, got "
                f"{sorted(unknown_main)}"
            )


def existing_tensor_names(layer: nn.Module, names: Iterable[str]) -> Tuple[str, ...]:
    """Return explicitly requested tensor attributes that exist on ``layer``."""

    return tuple(name for name in names if hasattr(layer, name))
