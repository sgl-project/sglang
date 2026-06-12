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

"""Context-parallel (CP) abstractions for prefill sequence sharding.

Public API:
    * ``ContextParallelStrategy`` — abstract base. See ``strategy.py``.
    * ``ZigzagCPStrategy`` / ``InterleaveCPStrategy`` — concrete strategies.
    * ``init_cp_strategy(server_args)`` — process-wide singleton init,
      called from ``ServerArgs.__post_init__``.
    * ``get_cp_strategy()`` / ``cp_active(forward_batch)`` — runtime gates
      that consumers (FA backend, DSA indexer, model bodies, …) read.
    * ``ContextParallelMetadata`` — per-forward payload dataclass.

The ``utils`` submodule re-exports a handful of legacy free functions
(``cp_split_and_rebuild_data``, ``cp_all_gather_rerange_output``, …) that
forward to the active strategy; existing call sites can keep importing
those names during the migration window.
"""

from sglang.srt.layers.cp.strategy import (
    BaseContextParallelMetadata,
    ContextParallelMetadata,
    ContextParallelStrategy,
    ContextParallelStrategyKind,
    CPAttentionBackendKind,
    CPAttnSlice,
    InterleaveContextParallelMetadata,
    InterleaveCPStrategy,
    ZigzagContextParallelMetadata,
    ZigzagCPStrategy,
    cp_active,
    get_cp_strategy,
    get_cp_strategy_kind,
    init_cp_strategy,
    is_cp_enabled,
    is_interleave,
    is_zigzag,
    use_cp_v2,
)

__all__ = [
    "BaseContextParallelMetadata",
    "CPAttentionBackendKind",
    "CPAttnSlice",
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
