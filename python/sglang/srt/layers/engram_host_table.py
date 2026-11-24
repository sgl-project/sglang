"""Interface for optional per-rank Engram host table providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Any

import torch


@dataclass(frozen=True)
class EngramTableSpec:
    layer_id: int
    num_embeddings: int
    dim: int
    block_size: int
    tp_rank: int
    tp_size: int
    pp_size: int
    dp_size: int
    cp_size: int
    max_num_tokens: int
    model_path: str
    device: torch.device
    dtype: torch.dtype = torch.float8_e4m3fn
    head_range: tuple[int, int] | None = None


class EngramHostTableBackend(ABC):
    """Provide one immutable shard on the current CUDA stream.

    ``row_range`` must match SGLang's partition. ``lookup`` fills ``out`` with
    BF16 rows, writing zeros for IDs outside this shard. The caller runs the
    usual TP all-reduce. The provider owns its resources and is responsible
    for ordering its work, including buffer reuse across graph replays. A
    provider that shares one buffer across calls requires serialized GPU
    execution for that instance. Finish in-flight work and retire graph
    replays before releasing resources.
    """

    def __init__(self, spec: EngramTableSpec, config: dict[str, Any]):
        self.spec = spec

    @property
    @abstractmethod
    def row_range(self) -> tuple[int, int]: ...

    @abstractmethod
    def lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None: ...

    def begin_lookup(self, indices: torch.Tensor) -> None:
        """Optional. Publish a lookup early on the current CUDA stream."""

    def finish_lookup(self, indices: torch.Tensor, out: torch.Tensor) -> None:
        """Consume the same batch of indices.

        Backends implementing only ``lookup`` keep the synchronous call path.
        """
        self.lookup(indices, out)

    def close(self) -> None:
        pass


def load_backend(name: str) -> type[EngramHostTableBackend]:
    found = tuple(entry_points(group="sglang.engram_host_table_backends", name=name))
    if not found:
        raise ValueError(f"No installed Engram host table backend named {name!r}.")
    if len(found) != 1:
        raise ValueError(f"More than one Engram host table backend named {name!r}.")
    backend = found[0].load()
    if not isinstance(backend, type) or not issubclass(backend, EngramHostTableBackend):
        raise TypeError("Engram host table backend must extend EngramHostTableBackend.")
    return backend
