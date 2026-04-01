from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class KVPageMeta:
    page_size: int
    layout: str
    is_mla_model: bool
    tp_rank: int
    tp_size: int
    model_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
