# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator


class DraftTPExecution(Enum):
    # All projections consume the same token rows within one TP group.
    SINGLE_GROUP = "single_group"
    # The model handles the target's attention/FFN row-layout transitions.
    TARGET_LAYOUT = "target_layout"


class DraftParallelSpec(msgspec.Struct, frozen=True, kw_only=True):
    tp_execution: DraftTPExecution


class DraftParallelLayout(msgspec.Struct, frozen=True, kw_only=True):
    """Resolved before construction and valid for the worker's group lifetime."""

    tp_group: GroupCoordinator | None
    attn_tp_size: int
    disable_cp: bool = False
    collapse_dcp: bool = False

    @property
    def local_dp(self) -> bool:
        return self.tp_group is not None
