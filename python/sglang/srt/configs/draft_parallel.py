from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class DraftTPExecution(Enum):
    SINGLE_GROUP = "single_group"
    TARGET_LAYOUT = "target_layout"


@dataclass(frozen=True)
class DraftParallelSpec:
    tp_execution: DraftTPExecution


@dataclass(frozen=True)
class DraftParallelLayout:
    """Resolved in target scope and shared by draft execution and KV sizing."""

    tp_group: GroupCoordinator
    attn_tp_size: int
    owns_attention: bool
