from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, kw_only=True)
class MarconiConfig:
    enable: bool

    @classmethod
    def enabled(cls) -> "MarconiConfig":
        return cls(enable=True)


def get_marconi_branch_align_interval(
    page_size: Optional[int] = None, *, align_interval: Optional[int] = None
) -> int:
    if align_interval is None:
        raise ValueError("Marconi branch alignment requires an explicit interval.")
    if align_interval <= 0:
        raise ValueError("Marconi branch alignment must be positive.")
    if page_size is not None and page_size > 0:
        if align_interval % page_size != 0:
            raise ValueError("Marconi branch alignment must be divisible by page_size.")
    return align_interval
