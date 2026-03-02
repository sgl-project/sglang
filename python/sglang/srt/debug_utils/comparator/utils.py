from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generic, Optional, Tuple, TypeVar

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


def auto_descend_dir(directory: Path, label: str) -> Path:
    """If directory has no .pt files but exactly one subdirectory does, descend into it.

    Raises ValueError when the layout is ambiguous (>=2 subdirs with .pt)
    or when no .pt data is found at all.
    """
    if any(directory.glob("*.pt")):
        return directory

    candidates: list[Path] = [
        sub for sub in directory.iterdir() if sub.is_dir() and any(sub.glob("*.pt"))
    ]

    if len(candidates) >= 2:
        names: str = ", ".join(sorted(c.name for c in candidates))
        raise ValueError(
            f"{label}: directory {directory} has no .pt files at top level "
            f"and multiple subdirectories contain data ({names}). "
            f"Please specify the exact subdirectory."
        )

    if len(candidates) == 0:
        raise ValueError(
            f"{label}: no .pt files found in {directory} or any of its subdirectories."
        )

    resolved: Path = candidates[0]

    from sglang.srt.debug_utils.comparator.log_sink import log_sink
    from sglang.srt.debug_utils.comparator.output_types import InfoLog

    log_sink.add(
        InfoLog(
            category="auto_descend",
            message=f"auto-descend {label}: {directory} -> {resolved}",
        )
    )
    return resolved


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


if TYPE_CHECKING:
    from sglang.srt.debug_utils.comparator.output_types import SummaryRecord


def compute_exit_code(
    summary: SummaryRecord,
    *,
    allow_skipped_pattern: str,
    skipped_names: list[str],
    allow_failed_pattern: Optional[str],
    failed_names: list[str],
) -> int:
    if summary.passed == 0:
        return 1

    if not _is_all_match_pattern(pattern=allow_failed_pattern, strings=failed_names):
        return 1

    if not _is_all_match_pattern(pattern=allow_skipped_pattern, strings=skipped_names):
        return 1

    return 0


def _is_all_match_pattern(*, pattern: Optional[str], strings: list[str]) -> bool:
    if pattern is None:
        return len(strings) == 0
    compiled: re.Pattern[str] = re.compile(pattern)
    return all(compiled.fullmatch(s) for s in strings)
