from __future__ import annotations

import functools
from typing import Callable, Generic, Optional, Tuple, TypeVar

import torch
from pydantic import BaseModel, ConfigDict

_T = TypeVar("_T")
_U = TypeVar("_U")


def _check_equal_lengths(**named_lists: list) -> None:
    lengths: dict[str, int] = {name: len(lst) for name, lst in named_lists.items()}
    unique: set[int] = set(lengths.values())
    if len(unique) > 1:
        details: str = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Length mismatch: {details}")


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _FrozenBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Pair(_FrozenBase, Generic[_T]):
    x: _T
    y: _T

    def map(self, fn: Callable[[_T], _U]) -> Pair[_U]:
        return Pair(x=fn(self.x), y=fn(self.y))


def argmax_coord(x: torch.Tensor) -> Tuple[int, ...]:
    flat_idx = x.argmax()
    return tuple(idx.item() for idx in torch.unravel_index(flat_idx, x.shape))


def compute_smaller_dtype(
    dtypes: Pair[torch.dtype],
) -> Optional[torch.dtype]:
    info_dict = {
        (torch.float32, torch.bfloat16): torch.bfloat16,
        # ... add more ...
    }
    return info_dict.get((dtypes.x, dtypes.y)) or info_dict.get((dtypes.y, dtypes.x))


def try_unify_shape(x: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    x_shape = x.shape
    num_dim_to_remove = len(x_shape) - len(target_shape)
    if (x_shape[num_dim_to_remove:] == target_shape) and all(
        val == 1 for val in x_shape[:num_dim_to_remove]
    ):
        return functools.reduce(lambda a, _: a.squeeze(0), range(num_dim_to_remove), x)

    return x


# Copied from DeepGEMM
def calc_rel_diff(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def calc_per_token_rel_diff(
    x: torch.Tensor, y: torch.Tensor, *, seq_dim: int
) -> torch.Tensor:
    """Cosine-distance-like metric per token position.

    Sums over all dims except seq_dim.
    """
    x, y = x.double(), y.double()
    other_dims: list[int] = [d for d in range(x.dim()) if d != seq_dim]

    if other_dims:
        denominator: torch.Tensor = (x * x + y * y).sum(dim=other_dims)
        sim: torch.Tensor = 2 * (x * y).sum(dim=other_dims) / (denominator + 1e-10)
    else:
        denominator = x * x + y * y
        sim = 2 * (x * y) / (denominator + 1e-10)

    return (1 - sim).float()
