"""Flip the first byte of a radix-cached but currently-unused (orphan) slot.

Detection should come from sweep (per-forward verify won't even look at this
slot). Designed to surface bugs where cached KV is silently corrupted and
sleeps until much later when a prefix happens to match.

To make detection robust against the allocator picking the perturbed slot
right after the flip (which would overwrite the corruption before sweep
runs), this module also pins the radix node owning the picked slot via
``inc_lock_ref`` and releases it a few sweep cycles later. While pinned,
the slot cannot be evicted from the radix cache and therefore cannot be
reallocated to a new request — guaranteeing the corruption survives until
the next sweep cycle observes it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup, PoolKind
from sglang.srt.kv_canary.perturb.config import (
    PerturbConfig,
    require_target_group_kind,
)
from sglang.srt.kv_canary.perturb.utils import (
    WarmupGate,
    flip_first_byte_in_source,
    pick_target_group,
    should_run_perturbation,
)
from sglang.srt.kv_canary.radix_cache_walker import walk_radix_cache_for_canary

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import (
        BasePrefixCache,
        DecLockRefParams,
    )
    from sglang.srt.mem_cache.radix_cache import TreeNode
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Pin the perturbed slot's radix node for this many sweep cycles after the
# byte flip. One cycle is enough for the next sweep to observe the
# corruption; we keep two for safety on systems where the sweep itself
# happens before the slot would normally be evicted but the violation log
# read is delayed.
_PIN_SWEEP_CYCLES: int = 2


@dataclass
class _PinnedNode:
    """An inc_lock_ref'd radix node awaiting dec_lock_ref."""

    node: "TreeNode"
    radix_cache: "BasePrefixCache"
    # Caller-supplied params to round-trip an SWA lock back through
    # dec_lock_ref. SWARadixCache.inc_lock_ref returns swa_uuid_for_lock /
    # swa_uuid_for_host_lock / skip_lock_node_ids which dec_lock_ref needs
    # to release only what we acquired. For plain RadixCache this is just
    # an empty params object.
    dec_params: "DecLockRefParams"
    expire_step: int


@dataclass
class PerturbRealKvUnusedCacheState:
    """Cross-call state for the real_kv_unused_cache perturbation.

    Tracks radix nodes pinned by past perturbations so they can be unpinned
    once a sweep cycle (or two) has elapsed.
    """

    pending: List[_PinnedNode] = field(default_factory=list)


def run(
    *,
    maybe_inaccurate_forward_batch: Optional["ForwardBatch"],
    config: PerturbConfig,
    buffer_groups: tuple[CanaryBufferGroup, ...],
    radix_cache: Optional["BasePrefixCache"],
    swa_window_size: int,
    sweep_interval: int,
    outer_step_counter: int,
    warmup_gate: WarmupGate,
    state: PerturbRealKvUnusedCacheState,
) -> None:
    # Unpin any previously-pinned nodes whose hold window has elapsed. This
    # runs every call regardless of whether a new perturbation happens this
    # step, so abandoned pins never leak.
    _unpin_expired(state=state, outer_step_counter=outer_step_counter)

    if sweep_interval <= 0 or outer_step_counter % sweep_interval != 0:
        return

    if not should_run_perturbation(
        perturb_name="real_kv_unused_cache",
        probability=config.real_kv_unused_cache_prob,
        warmup_gate=warmup_gate,
        maybe_inaccurate_forward_batch=maybe_inaccurate_forward_batch,
        require_forward_batch=False,
    ):
        return

    group = pick_target_group(
        buffer_groups=buffer_groups,
        target_kind=require_target_group_kind(
            target_group_kind=config.target_group_kind,
            perturb_name="real_kv_unused_cache",
        ),
    )
    if group is None:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because no target group with "
            "real_kv_sources_k matched target_group_kind=%s",
            config.target_group_kind,
        )
        return
    pick = _pick_sweep_slot_for_group(
        radix_cache=radix_cache,
        group=group,
        swa_window_size=swa_window_size,
    )
    if pick is None:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because no orphan sweep slot "
            "was found for group=%s",
            group.kind.name,
        )
        return
    slot, owning_node = pick
    source_pick = int(torch.randint(0, len(group.real_kv_sources_k), (1,)).item())
    source = group.real_kv_sources_k[source_pick]
    flip_result = flip_first_byte_in_source(
        group=group,
        source=source,
        slot_idx=slot,
        slot_is_physical=True,
    )
    if flip_result is None:
        logger.info(
            "kv_canary perturb real_kv_unused_cache: skipped because slot=%d could not be mapped "
            "into group=%s source_idx=%d",
            slot,
            group.kind.name,
            source_pick,
        )
        return
    row, col, original_byte = flip_result

    # Pin the owning radix node so the picked slot cannot be evicted (and
    # therefore cannot be reallocated by the next forward) before sweep
    # observes the corruption. SWARadixCache returns swa_uuid_for_lock /
    # related fields that must be replayed back through dec_lock_ref to
    # release only what we acquired, so capture them now.
    assert radix_cache is not None  # _pick_sweep_slot_for_group needs it
    inc_result = radix_cache.inc_lock_ref(owning_node)
    state.pending.append(
        _PinnedNode(
            node=owning_node,
            radix_cache=radix_cache,
            dec_params=inc_result.to_dec_params(),
            expire_step=outer_step_counter + _PIN_SWEEP_CYCLES * sweep_interval,
        )
    )

    logger.info(
        "kv_canary perturb real_kv_unused_cache: group=%s source_idx=%d slot=%d row=%d col=%d "
        "original_byte=0x%02X new_byte=0x%02X pinned_until_step=%d",
        group.kind.name,
        source_pick,
        slot,
        row,
        col,
        original_byte,
        original_byte ^ 0xFF,
        outer_step_counter + _PIN_SWEEP_CYCLES * sweep_interval,
    )


def _unpin_expired(
    *,
    state: PerturbRealKvUnusedCacheState,
    outer_step_counter: int,
) -> None:
    """Release inc_lock_ref pins whose expire_step has passed."""
    still_pending: List[_PinnedNode] = []
    for entry in state.pending:
        if outer_step_counter >= entry.expire_step:
            try:
                entry.radix_cache.dec_lock_ref(entry.node, entry.dec_params)
            except Exception:
                # Best-effort cleanup. Don't let a stale node lifetime kill
                # the scheduler — log and drop it.
                logger.exception(
                    "kv_canary perturb real_kv_unused_cache: dec_lock_ref failed; "
                    "dropping pinned entry"
                )
        else:
            still_pending.append(entry)
    state.pending = still_pending


def _pick_sweep_slot_for_group(
    *,
    radix_cache: Optional["BasePrefixCache"],
    group: CanaryBufferGroup,
    swa_window_size: int,
) -> Optional[tuple[int, "TreeNode"]]:
    """Return one (slot, owning_node) candidate from the radix cache, or None.

    For SWA groups the slot is translated through ``group.swa_index_lut``
    into SWA-pool space. The owning node is the radix tree node that holds
    the slot in its ``value`` tensor — used by the caller to inc_lock_ref
    the node so the slot can't be evicted/reallocated before sweep verifies.

    swa_resident_only=True regardless of group: even for FULL-target on
    SWARadixCache, we cannot inc_lock_ref a swa_tombstone node (the SWA
    lock half of SWARadixCache.inc_lock_ref asserts non-tombstone). On
    plain RadixCache the flag is a no-op (no tombstones), so a single
    setting is safe for both backends.
    """
    if radix_cache is None:
        return None

    walk_result = walk_radix_cache_for_canary(
        radix_cache=radix_cache,
        unlocked_only=True,
        swa_resident_only=True,
        collect_owning_nodes=True,
    )
    raw_slots = walk_result.slot_indices.detach().to("cpu").tolist()
    nodes = walk_result.owning_nodes
    assert nodes is not None and len(nodes) == len(raw_slots)

    candidates: list[tuple[int, "TreeNode"]] = []
    for raw_slot, node in zip(raw_slots, nodes):
        full_slot = int(raw_slot)
        if full_slot < 0:
            continue
        if group.kind is PoolKind.SWA:
            translated = _translate_full_slot_to_swa(
                full_slot=full_slot,
                full_to_swa_index_mapping=group.swa_index_lut,
            )
            if translated is None:
                continue
            candidates.append((translated, node))
        else:
            candidates.append((full_slot, node))

    if not candidates:
        return None

    pick = int(torch.randint(0, len(candidates), (1,)).item())
    return candidates[pick]


def _translate_full_slot_to_swa(
    *,
    full_slot: int,
    full_to_swa_index_mapping: Optional[torch.Tensor],
) -> Optional[int]:
    """Translate one FULL-pool slot index into its SWA-pool physical slot.

    Returns None when the LUT is absent, the slot is out of range, or the
    LUT entry sentinels the slot as not-SWA-resident (negative).
    """
    if full_to_swa_index_mapping is None:
        return None
    if full_slot < 0 or full_slot >= int(full_to_swa_index_mapping.shape[0]):
        return None
    physical_slot = int(full_to_swa_index_mapping[full_slot].item())
    if physical_slot < 0:
        return None
    return physical_slot
