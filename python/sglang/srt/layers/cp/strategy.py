# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Context parallel strategy facade.

The strategy implementation is split across:

* ``base.py``: base ABC, base metadata dataclass, and enums.
* ``zigzag.py``: former in-seq-split strategy and zigzag metadata.
* ``interleave.py``: former round-robin-split strategy and interleave metadata.

This module remains the stable import path for the process-wide singleton and
public re-exports.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from sglang.srt.layers.cp.base import (
    BaseContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
)
from sglang.srt.layers.cp.interleave import (
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
)
from sglang.srt.layers.cp.zigzag import (
    ContextParallelMetadata,
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


_STRATEGY: Optional[ContextParallelStrategy] = None


def use_cp_v2() -> bool:
    from sglang.srt.environ import envs

    return bool(envs.SGLANG_ENABLE_CP_V2.get())


def init_cp_strategy(server_args: "ServerArgs") -> None:
    """Bind the active strategy for this process."""
    global _STRATEGY

    if not getattr(server_args, "enable_prefill_cp", False):
        _STRATEGY = None
        return

    cp_size = getattr(server_args, "attn_cp_size", 1)
    if cp_size <= 1:
        _STRATEGY = None
        return

    kind = ContextParallelStrategyKind.from_string(server_args.cp_strategy)
    if kind == ContextParallelStrategyKind.ZIGZAG:
        _STRATEGY = ZigzagCPStrategy(cp_size=cp_size)
    elif kind == ContextParallelStrategyKind.INTERLEAVE:
        _STRATEGY = InterleaveCPStrategy(cp_size=cp_size)
    else:
        raise ValueError(
            f"Unsupported cp_strategy kind {kind} for "
            f"cp_strategy={server_args.cp_strategy!r}"
        )


def _get_cp_strategy() -> Optional[ContextParallelStrategy]:
    """Return the configured strategy, initializing lazily on first call.

    Subprocesses re-import this module with ``_STRATEGY = None`` and never
    re-run ``ServerArgs.__post_init__`` because the pickled instance bypasses
    ``__init__``. Lazy init lets worker processes recover the singleton from
    global server args. This accessor intentionally ignores
    ``SGLANG_ENABLE_CP_V2`` so deprecated v1 compatibility paths can still read
    the normalized strategy configuration.
    """
    global _STRATEGY

    if _STRATEGY is None:
        from sglang.srt.server_args import get_global_server_args

        try:
            server_args = get_global_server_args()
        except ValueError:
            return None
        if server_args is not None and getattr(server_args, "enable_prefill_cp", False):
            init_cp_strategy(server_args)
    return _STRATEGY


def get_cp_strategy() -> Optional[ContextParallelStrategy]:
    """Return the active CP-v2 runtime strategy."""
    if not use_cp_v2():
        return None
    return _get_cp_strategy()


def get_cp_strategy_kind() -> ContextParallelStrategyKind:
    strategy = _get_cp_strategy()
    if strategy is None:
        return ContextParallelStrategyKind.NONE
    return strategy.kind


def is_cp_enabled() -> bool:
    return _get_cp_strategy() is not None


def is_zigzag() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.ZIGZAG


def is_interleave() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.INTERLEAVE


def cp_active(forward_batch: "ForwardBatch") -> bool:
    """True when CP is engaged for this forward pass."""
    return (
        use_cp_v2()
        and getattr(forward_batch, "attn_cp_metadata", None) is not None
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


__all__ = [
    "BaseContextParallelMetadata",
    "CPAttentionBackendKind",
    "ContextParallelMetadata",
    "ContextParallelStrategy",
    "ContextParallelStrategyKind",
    "InterleaveCPStrategy",
    "InterleaveContextParallelMetadata",
    "ZigzagCPStrategy",
    "ZigzagContextParallelMetadata",
    "cp_active",
    "get_cp_strategy",
    "get_cp_strategy_kind",
    "init_cp_strategy",
    "is_cp_enabled",
    "is_interleave",
    "is_zigzag",
    "use_cp_v2",
]
