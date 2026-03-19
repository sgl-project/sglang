# Copyright 2026 SGLang Team
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
"""UVM-based expert weight offloading for MoE layers.

Expert weights are stored in CUDA Unified Memory (cudaMallocManaged).
Resident experts are kept in GPU VRAM (PREFER_GPU advice).
Offloaded experts live in CPU DRAM and are accessible to the GPU via PCIe
read-through (PREFER_CPU + ACCESSED_BY_GPU advice) -- no page fault overhead,
no quality loss, CUDA graph compatible.

All computation remains on GPU (no CPU inference).
Mutually exclusive with KTransformers (--kt-weight-path).

Usage
-----
Pass ``--expert-offload-num-resident N`` to enable.  Additional options:

  --expert-offload-resident-selection first_n How to choose resident experts.
  --expert-offload-resident-ids       None    Comma-separated IDs for manual selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

from sglang.srt.layers.moe.expert_offload.config import (
    ExpertOffloadConfig,
    create_expert_offload_config_from_server_args,
)
from sglang.srt.layers.moe.expert_offload.wrapper import ExpertOffloadWrapperMethod

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.moe.expert_offload.manager import ExpertOffloadManager

__all__ = [
    "ExpertOffloadConfig",
    "ExpertOffloadWrapperMethod",
    "chain_managers",
    "create_expert_offload_config_from_server_args",
    "register_manager",
]

# ---------------------------------------------------------------------------
# Global manager registry for cross-layer prefetch chaining
# ---------------------------------------------------------------------------

_manager_registry: Dict[int, Tuple["ExpertOffloadManager", "torch.nn.Module"]] = {}
_managers_chained: bool = False


def register_manager(
    layer_idx: int, manager: "ExpertOffloadManager", layer: "torch.nn.Module"
) -> None:
    """Called from wrapper.process_weights_after_loading()."""
    _manager_registry[layer_idx] = (manager, layer)


def chain_managers() -> None:
    """Link managers[i].next_layer_manager = managers[i+1], build prefetch caches.

    Guarded by ``_managers_chained`` flag -- no-op after first call.
    """
    global _managers_chained
    if _managers_chained or not _manager_registry:
        return
    _managers_chained = True

    sorted_idxs = sorted(_manager_registry.keys())
    for i in range(len(sorted_idxs) - 1):
        curr_mgr, _ = _manager_registry[sorted_idxs[i]]
        next_mgr, _ = _manager_registry[sorted_idxs[i + 1]]
        curr_mgr.next_layer_manager = next_mgr

    # Build prefetch caches for all managers.
    for idx in sorted_idxs:
        mgr, layer = _manager_registry[idx]
        mgr.prepare_prefetch_cache(layer)

    _manager_registry.clear()
