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

from typing import TYPE_CHECKING, Dict, List, Tuple

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
    "notify_decode_completed",
    "prepare_for_new_prefill",
    "register_manager",
    "restore_after_prefill",
]

# ---------------------------------------------------------------------------
# Global manager registry for cross-layer prefetch chaining
# ---------------------------------------------------------------------------

_manager_registry: Dict[int, Tuple["ExpertOffloadManager", "torch.nn.Module"]] = {}
_managers_chained: bool = False

# Persistent list of all (manager, layer) pairs.
# Unlike _manager_registry which is cleared after chain_managers(), this list
# survives so that prepare_for_new_prefill() can iterate over all managers.
_all_managers: List[Tuple["ExpertOffloadManager", "torch.nn.Module"]] = []

# True once at least one decode step has executed since the last page reset.
# Used to skip the expensive prefetch+sync when consecutive extends run
# (e.g. chunked prefill) with no intervening decode.
_decode_since_last_reset: bool = False

# Set True during the first forward pass after prepare_for_new_prefill.
# Used to enable per-layer tracing in wrapper.apply() so we can see which
# MoE layer the kernel hangs on.
_tracing_prefill: bool = False


def register_manager(
    layer_idx: int, manager: "ExpertOffloadManager", layer: "torch.nn.Module"
) -> None:
    """Called from wrapper.process_weights_after_loading()."""
    _manager_registry[layer_idx] = (manager, layer)
    _all_managers.append((manager, layer))


def chain_managers() -> None:
    """Link managers[i].next_layer_managers to the next D managers, build prefetch caches.

    D is determined by each manager's ``config.prefetch_depth``.
    Guarded by ``_managers_chained`` flag -- no-op after first call.
    """
    global _managers_chained
    if _managers_chained or not _manager_registry:
        return
    _managers_chained = True

    sorted_idxs = sorted(_manager_registry.keys())
    n = len(sorted_idxs)
    for i in range(n):
        curr_mgr, _ = _manager_registry[sorted_idxs[i]]
        depth = curr_mgr.config.prefetch_depth
        targets = []
        for d in range(1, depth + 1):
            j = i + d
            if j < n:
                next_mgr, _ = _manager_registry[sorted_idxs[j]]
                targets.append(next_mgr)
        curr_mgr.next_layer_managers = targets

    # Build prefetch caches for all managers.
    for idx in sorted_idxs:
        mgr, layer = _manager_registry[idx]
        mgr.prepare_prefetch_cache(layer)

    _manager_registry.clear()


def notify_decode_completed() -> None:
    """Mark that a decode step has run, so the next prefill triggers page reset."""
    global _decode_since_last_reset
    _decode_since_last_reset = True


def prepare_for_new_prefill() -> None:
    """Notify decode->prefill transition.

    The real fix for UVM page-swap deadlocks is a 2 GiB headroom
    reserved in get_available_gpu_memory() so the UVM driver always
    has free GPU pages for bi-directional page swapping.
    This function just resets bookkeeping flags and logs memory state.
    """
    import logging

    import torch

    logger = logging.getLogger(__name__)

    global _decode_since_last_reset
    if not _all_managers or not _decode_since_last_reset:
        return
    _decode_since_last_reset = False

    global _tracing_prefill
    _tracing_prefill = True

    free, total = torch.cuda.mem_get_info()
    used_gb = (total - free) / (1 << 30)
    free_gb = free / (1 << 30)
    logger.info(
        f"[ExpertOffload] prepare_for_new_prefill: "
        f"decode->prefill transition "
        f"(GPU mem: {used_gb:.2f} GiB used, {free_gb:.2f} GiB free)"
    )


def restore_after_prefill() -> None:
    """Re-enable GPU direct mapping for offloaded experts after prefill.

    Called after forward_extend completes to restore ACCESSED_BY advice
    so decode can use fast PCIe read-through without page fault overhead.
    """
    import logging
    import time

    logger = logging.getLogger(__name__)

    if not _all_managers:
        return

    t0 = time.monotonic()
    for mgr, layer in _all_managers:
        mgr.restore_accessed_by_for_offloaded(layer)
    t1 = time.monotonic()

    logger.info(
        f"[ExpertOffload] restore_after_prefill: "
        f"re-set ACCESSED_BY in {t1 - t0:.3f}s"
    )
