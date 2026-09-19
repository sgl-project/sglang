"""NPU sleep-mode device communicator cleanup.

When ``--enable-sleep-comm-cleanup`` is on, releasing memory occupation also
destroys the HCCL process groups (whose buffers can be hundreds of MB per
group) and resuming rebuilds them. The gloo cpu groups are kept throughout:
they are the control plane and hold no device memory.

Ordering rules (mirroring the worker-side hooks in weight_updater.py):
  - sleep:   pause cuda graphs first, then release comms (graphs may hold
             HCCL references; destroying comms first can hang)
  - wakeup:  restore comms first, then resume cuda graphs (re-capture runs
             forward passes whose MoE layers need communication)
"""

from __future__ import annotations

import logging
from typing import List

import torch

from sglang.srt.distributed import parallel_state

logger = logging.getLogger(__name__)


def _iter_registered_groups() -> List[parallel_state.GroupCoordinator]:
    """Collect live GroupCoordinators from the weakref registry, deduplicated
    by object id (the same group can be referenced under several names; a
    double destroy would be an error)."""
    seen = set()
    groups = []
    for group_ref in parallel_state._groups.values():
        group = group_ref()
        if group is None:
            continue
        if id(group) in seen:
            continue
        seen.add(id(group))
        groups.append(group)
    return groups


def release_device_comms() -> int:
    """Destroy all device-side comm resources. Returns how many groups were
    actually released."""
    # HCCL destroy requires the communicators to be idle; a pending
    # collective would hang the destroy.
    torch.npu.synchronize()

    released = 0
    for group in _iter_registered_groups():
        if group.release_device_comm():
            released += 1
    logger.info(
        f"[sleep-comm] Released device comm resources of {released} group(s)."
    )
    return released


def restore_device_comms() -> int:
    """Rebuild all device-side comm resources and drop stale MoE comm-buffer
    caches that hold references to the old HCCL groups. Returns how many
    groups were actually restored."""
    restored = 0
    for group in _iter_registered_groups():
        if group.restore_device_comm():
            restored += 1

    # DeepEP / FuseEP buffers cache the old device group; the next dispatch
    # recreates them lazily on top of the rebuilt group.
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer

    DeepEPBuffer.invalidate_buffer()

    logger.info(
        f"[sleep-comm] Restored device comm resources of {restored} group(s)."
    )
    return restored
