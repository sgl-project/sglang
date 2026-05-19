"""Plan-input builders for the three canary caller paths.

A :class:`PlanInput` bundles every tensor that :func:`canary_plan_step` consumes (minus the static
``swa_window_size`` and ``full_to_swa_index_mapping`` arguments that come from the runner / buffer group).

Three host-side builders construct one of these from sglang state:

- :func:`build_plan_input_per_forward`     — derives counts from a live ``ForwardBatch``.
- :func:`build_plan_input_running_sweep`   — sweeps every running req's full prefix.
- :func:`build_plan_input_radix_sweep`     — sweeps the radix cache's orphan slots (radix-held but not owned
  by any running req).

The radix walker :func:`walk_radix_cache_for_canary` is the underlying helper for the radix builder.

All builders return a :class:`PlanInput` whose addresses are stable across calls; ``PlanInput`` itself is
``frozen=True, slots=True, kw_only=True`` so the runner can capture the dataclass into cuda-graph state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanInput:
    """The set of input tensors the runner hands to :func:`canary_plan_step`.

    Field names mirror the kwargs of ``canary_plan_step`` so the runner can splat one of these instances
    straight into the call site without re-mapping.

    Fields:
        fb_req_pool_indices: ``ForwardBatch.req_pool_indices``-shaped int32 tensor (``[bs]``). 0 = padding
            sentinel.
        fb_prefix_lens: Per-req prefix length already written before this step, shape ``[bs]``, int32.
        fb_extend_seq_lens: Per-req tokens being written this step, shape ``[bs]``, int32. 0 for sweep.
        req_to_token: Full-pool slot index table, shape ``[max_reqs, max_seq_len]``, int32.
        extra_verify_slot_indices: Pre-walked extra verify slots, shape ``[extra_verify_capacity]``, int32.
            Caller-translated to the target index space.
        extra_verify_positions: Expected position per extra entry, shape ``[extra_verify_capacity]``, int32.
        extra_verify_prev_slot_indices: Predecessor slot per extra entry; ``-1`` for chain-seed extras.
        extra_verify_num_valid: Active extra entry count, shape ``[1]``, int32. 0 for per-forward and
            running-sweep callers.
    """

    fb_req_pool_indices: torch.Tensor
    fb_prefix_lens: torch.Tensor
    fb_extend_seq_lens: torch.Tensor
    req_to_token: torch.Tensor
    extra_verify_slot_indices: torch.Tensor
    extra_verify_positions: torch.Tensor
    extra_verify_prev_slot_indices: torch.Tensor
    extra_verify_num_valid: torch.Tensor


def build_plan_input_per_forward(
    *,
    forward_batch: "ForwardBatch",
    req_to_token_pool: "ReqToTokenPool",
    extras_capacity: int,
) -> Optional[PlanInput]:
    """Build the per-forward :class:`PlanInput` for one ``ForwardBatch``.

    Returns ``None`` when the forward batch is missing required tensors (e.g. spec-decoding draft batches
    that don't expose ``extend_*`` lengths). Caller treats ``None`` as "no canary plan this forward".

    Extras are zeros — the per-forward path never walks radix orphans.
    """
    if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
        return None

    device = forward_batch.req_pool_indices.device
    req_pool_indices = forward_batch.req_pool_indices.to(torch.int32).contiguous()

    prefix_lens = _derive_per_forward_prefix_lens(forward_batch=forward_batch)
    if prefix_lens is None:
        return None
    extend_seq_lens = _derive_per_forward_extend_seq_lens(forward_batch=forward_batch)
    if extend_seq_lens is None:
        return None

    extras = _allocate_empty_extras(capacity=extras_capacity, device=device)

    return PlanInput(
        fb_req_pool_indices=req_pool_indices,
        fb_prefix_lens=prefix_lens,
        fb_extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token_pool.req_to_token,
        extra_verify_slot_indices=extras[0],
        extra_verify_positions=extras[1],
        extra_verify_prev_slot_indices=extras[2],
        extra_verify_num_valid=extras[3],
    )


def build_plan_input_running_sweep(
    *,
    req_to_token_pool: "ReqToTokenPool",
    running_req_pool_indices: torch.Tensor,
    running_seq_lens: torch.Tensor,
    extras_capacity: int,
) -> PlanInput:
    """Build a :class:`PlanInput` that sweeps every running req's full prefix.

    ``running_req_pool_indices`` / ``running_seq_lens`` are int32 ``[bs]`` device tensors (caller assembles
    them from the scheduler's running batch). ``fb_extend_seq_lens`` is all-zero so the plan kernel sets
    ``write_num_valid_reqs = 0`` and downstream skips ``canary_write_step``.

    Extras are zero. Radix orphans live in a separate sweep builder.
    """
    device = running_req_pool_indices.device
    extend_seq_lens = torch.zeros_like(running_seq_lens, dtype=torch.int32)
    extras = _allocate_empty_extras(capacity=extras_capacity, device=device)
    return PlanInput(
        fb_req_pool_indices=running_req_pool_indices.to(torch.int32).contiguous(),
        fb_prefix_lens=running_seq_lens.to(torch.int32).contiguous(),
        fb_extend_seq_lens=extend_seq_lens,
        req_to_token=req_to_token_pool.req_to_token,
        extra_verify_slot_indices=extras[0],
        extra_verify_positions=extras[1],
        extra_verify_prev_slot_indices=extras[2],
        extra_verify_num_valid=extras[3],
    )


def build_plan_input_radix_sweep(
    *,
    req_to_token_pool: "ReqToTokenPool",
    radix_cache: "BasePrefixCache",
    extras_capacity: int,
    swa_index_lut: Optional[torch.Tensor] = None,
) -> PlanInput:
    """Build a :class:`PlanInput` that sweeps radix-orphan slots via the ``extra_verify_*`` channel.

    ``fb_req_pool_indices`` / ``fb_prefix_lens`` / ``fb_extend_seq_lens`` are minimal one-element padding
    rows (rpi == 0 → padding) so the kernel contributes zero per-req entries. All sweep work lands in the
    extras.

    ``swa_index_lut`` controls SWA translation of the orphan slots; the plan kernel does NOT translate
    extras, so caller-side translation happens here. ``None`` leaves the indices in full-pool space.
    """
    device = req_to_token_pool.req_to_token.device
    slot_indices, positions, prev_slot_indices = walk_radix_cache_for_canary(
        radix_cache=radix_cache, device=device
    )

    if swa_index_lut is not None:
        slot_indices = _swa_translate_orphan_indices(
            indices=slot_indices, lut=swa_index_lut
        )
        prev_slot_indices = _swa_translate_orphan_indices(
            indices=prev_slot_indices, lut=swa_index_lut
        )

    extras = _materialize_extras_into_capacity(
        slot_indices=slot_indices,
        positions=positions,
        prev_slot_indices=prev_slot_indices,
        capacity=extras_capacity,
        device=device,
    )

    fb_req_pool_indices = torch.zeros(1, dtype=torch.int32, device=device)
    fb_prefix_lens = torch.zeros(1, dtype=torch.int32, device=device)
    fb_extend_seq_lens = torch.zeros(1, dtype=torch.int32, device=device)

    return PlanInput(
        fb_req_pool_indices=fb_req_pool_indices,
        fb_prefix_lens=fb_prefix_lens,
        fb_extend_seq_lens=fb_extend_seq_lens,
        req_to_token=req_to_token_pool.req_to_token,
        extra_verify_slot_indices=extras[0],
        extra_verify_positions=extras[1],
        extra_verify_prev_slot_indices=extras[2],
        extra_verify_num_valid=extras[3],
    )


def walk_radix_cache_for_canary(
    *,
    radix_cache: "BasePrefixCache",
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Walk the radix tree and emit flat ``(slot_indices, positions, prev_slot_indices)`` device tensors.

    Returns three int32 device tensors of equal length, ready to feed as ``extra_verify_*`` into
    :func:`canary_plan_step`. The walk visits every node reachable from the root and emits one entry per
    slot held by the node, chained by node-internal position (and across parent → child boundaries).

    Subclasses of :class:`BasePrefixCache` not exposing a walkable ``root_node`` (e.g. chunk cache, C++
    radix) return empty tensors — sweep degrades to running-only coverage on those backends.
    """
    slot_list: List[int] = []
    position_list: List[int] = []
    prev_slot_list: List[int] = []

    root = getattr(radix_cache, "root_node", None)
    if root is None:
        return _empty_walk_result(device=device)

    _walk_node(
        node=root,
        depth=0,
        parent_last_slot=-1,
        slot_list=slot_list,
        position_list=position_list,
        prev_slot_list=prev_slot_list,
    )

    if not slot_list:
        return _empty_walk_result(device=device)

    slot_tensor = torch.tensor(slot_list, dtype=torch.int32, device=device)
    position_tensor = torch.tensor(position_list, dtype=torch.int32, device=device)
    prev_slot_tensor = torch.tensor(prev_slot_list, dtype=torch.int32, device=device)
    return slot_tensor, position_tensor, prev_slot_tensor


def _walk_node(
    *,
    node: object,
    depth: int,
    parent_last_slot: int,
    slot_list: List[int],
    position_list: List[int],
    prev_slot_list: List[int],
) -> None:
    """Recursive helper that flattens one radix subtree into the parallel slot / position / prev lists.

    Each node contributes ``len(node.value)`` slots; intra-node chain is sequential and the first slot of a
    non-root node uses ``parent_last_slot`` as its predecessor (or ``-1`` when the walk begins at root).
    """
    value = getattr(node, "value", None)
    children = getattr(node, "children", None)

    node_slots: List[int] = []
    if value is not None:
        try:
            node_slots = [int(s) for s in list(value)]
        except (TypeError, ValueError):
            node_slots = []

    last_slot_for_children = parent_last_slot
    for j, slot in enumerate(node_slots):
        prev = parent_last_slot if j == 0 else node_slots[j - 1]
        slot_list.append(slot)
        position_list.append(depth + j)
        prev_slot_list.append(prev)
        last_slot_for_children = slot

    if children is None:
        return
    try:
        iterator = children.values() if hasattr(children, "values") else iter(children)
    except TypeError:
        return
    for child in iterator:
        _walk_node(
            node=child,
            depth=depth + len(node_slots),
            parent_last_slot=last_slot_for_children,
            slot_list=slot_list,
            position_list=position_list,
            prev_slot_list=prev_slot_list,
        )


def _derive_per_forward_prefix_lens(
    *, forward_batch: "ForwardBatch"
) -> Optional[torch.Tensor]:
    """Extend → ``extend_prefix_lens``; decode → ``seq_lens - 1`` (clipped at 0)."""
    forward_mode = forward_batch.forward_mode
    if forward_mode is not None and (
        forward_mode.is_extend() or forward_mode.is_mixed()
    ):
        if forward_batch.extend_prefix_lens is None:
            return None
        return forward_batch.extend_prefix_lens.to(torch.int32).contiguous()
    if forward_batch.seq_lens is None:
        return None
    decode_prefix = (forward_batch.seq_lens - 1).clamp(min=0)
    return decode_prefix.to(torch.int32).contiguous()


def _derive_per_forward_extend_seq_lens(
    *, forward_batch: "ForwardBatch"
) -> Optional[torch.Tensor]:
    """Extend → ``extend_seq_lens``; decode → all-ones same shape as ``req_pool_indices``."""
    forward_mode = forward_batch.forward_mode
    if forward_mode is not None and (
        forward_mode.is_extend() or forward_mode.is_mixed()
    ):
        if forward_batch.extend_seq_lens is None:
            return None
        return forward_batch.extend_seq_lens.to(torch.int32).contiguous()
    if forward_batch.req_pool_indices is None:
        return None
    bs = int(forward_batch.req_pool_indices.shape[0])
    device = forward_batch.req_pool_indices.device
    return torch.ones(bs, dtype=torch.int32, device=device)


def _allocate_empty_extras(
    *, capacity: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return four cuda-graph-stable extras tensors with ``num_valid == 0``.

    Capacity must be positive — even an unused channel allocates a one-element tensor so the kernel ABI
    always has a valid pointer.
    """
    safe_capacity = max(1, int(capacity))
    slot = torch.zeros(safe_capacity, dtype=torch.int32, device=device)
    position = torch.zeros(safe_capacity, dtype=torch.int32, device=device)
    prev_slot = torch.zeros(safe_capacity, dtype=torch.int32, device=device)
    num_valid = torch.zeros(1, dtype=torch.int32, device=device)
    return slot, position, prev_slot, num_valid


def _materialize_extras_into_capacity(
    *,
    slot_indices: torch.Tensor,
    positions: torch.Tensor,
    prev_slot_indices: torch.Tensor,
    capacity: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack the three walker tensors into capacity-sized buffers and record the active count."""
    safe_capacity = max(1, int(capacity))
    n = int(slot_indices.shape[0]) if slot_indices.numel() > 0 else 0
    if n > safe_capacity:
        logger.warning(
            "kv-canary: radix sweep produced %d extras but capacity is %d; truncating",
            n,
            safe_capacity,
        )
        n = safe_capacity

    slot = torch.zeros(safe_capacity, dtype=torch.int32, device=device)
    position = torch.zeros(safe_capacity, dtype=torch.int32, device=device)
    prev_slot = torch.zeros(safe_capacity, dtype=torch.int32, device=device)
    if n > 0:
        slot[:n] = slot_indices[:n].to(torch.int32)
        position[:n] = positions[:n].to(torch.int32)
        prev_slot[:n] = prev_slot_indices[:n].to(torch.int32)
    num_valid = torch.tensor([n], dtype=torch.int32, device=device)
    return slot, position, prev_slot, num_valid


def _swa_translate_orphan_indices(
    *, indices: torch.Tensor, lut: torch.Tensor
) -> torch.Tensor:
    """Translate full-pool slot indices through the SWA LUT, leaving ``-1`` sentinels untouched."""
    if indices.numel() == 0:
        return indices
    lut_device = lut.to(indices.device).to(torch.int32)
    lut_len = int(lut_device.shape[0])
    sentinel_mask = indices < 0
    safe = torch.where(
        sentinel_mask, torch.zeros_like(indices), indices.to(torch.int64)
    )
    if lut_len > 0:
        safe = torch.where(safe >= lut_len, torch.full_like(safe, lut_len - 1), safe)
    translated = lut_device[safe.to(torch.int64)]
    return torch.where(sentinel_mask, indices.to(torch.int32), translated.to(torch.int32))


def _empty_walk_result(
    *, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    empty = torch.empty(0, dtype=torch.int32, device=device)
    return empty, empty.clone(), empty.clone()
