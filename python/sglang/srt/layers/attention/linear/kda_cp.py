"""FLA affine-state context parallel helpers for KDA prefill.

This adapts flash-linear-attention PR 691's affine recurrent-state composition
to SGLang's two-segment zigzag layout. QKV activations stay rank-local; only the
much smaller per-segment state transforms are gathered across CP ranks.

Decode never enters these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import accumulate
from typing import Any, Optional

import torch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import is_npu


@dataclass(frozen=True)
class KDAFLACPContext:
    """Operator-level CP metadata for the PR-691-style KDA path."""

    group: Any
    cp_size: int
    cp_rank: int
    batch_size: int
    split_list: tuple[int, ...]
    local_segment_lens: tuple[int, ...]
    local_cu_seqlens: torch.Tensor
    local_segment_slots: tuple[int, ...]
    rank_segment_slots: tuple[tuple[int, ...], ...]
    fixed_segment_sources: tuple[int, ...]
    max_rank_segments: int
    fixed_segment_lens: tuple[int, ...]
    track_after_slots: tuple[int, ...]
    track_state_indices: torch.Tensor
    track_request_indices: Optional[torch.Tensor] = None
    track_request_ids: tuple[int, ...] = ()
    local_segment_indices: Optional[torch.Tensor] = None
    local_segment_has_initial_state: Optional[torch.Tensor] = None
    local_segment_lens_cpu: Optional[list[int]] = None
    local_segment_ends: tuple[int, ...] = ()
    affine_steps: tuple[
        tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...
    ] = ()
    affine_owner_ranks: Optional[torch.Tensor] = None
    affine_source_segments: Optional[torch.Tensor] = None
    affine_local_indices: Optional[torch.Tensor] = None
    affine_local_steps: tuple[int, ...] = ()
    affine_local_steps_tensor: Optional[torch.Tensor] = None
    affine_local_output_indices: Optional[torch.Tensor] = None
    affine_source_flat_indices: Optional[torch.Tensor] = None
    conv_local_tail_steps: Optional[torch.Tensor] = None
    conv_local_tail_output_indices: Optional[torch.Tensor] = None
    conv_prefix_output_index: int = -1
    affine_track_step: int = -1
    scratch_buffers: dict[str, torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )

    @property
    def num_local_segments(self) -> int:
        return len(self.local_segment_lens)

    @property
    def num_fixed_segments(self) -> int:
        # Reserve two subsegments for every natural zigzag block.  Normally only
        # part 0 is populated; part 1 is used when a radix checkpoint splits a
        # block. Collectives send only rank-owned segments, padded to the maximum
        # rank segment count.
        return self.batch_size * 2 * self.cp_size * 2


def kda_use_prefill_cp(forward_batch: Any) -> bool:
    """Whether this forward needs a context-parallel KDA implementation."""
    parallel = get_parallel()
    mode = forward_batch.forward_mode
    return bool(
        is_npu()
        and parallel.enable_prefill_context_parallel
        and parallel.attn_cp_size > 1
        and forward_batch.attn_cp_metadata is not None
        and mode.is_context_parallel_extend()
        and not mode.is_mixed()
        and not mode.is_target_verify()
        and not mode.is_draft_extend_v2()
    )


def kda_use_fla_prefill_cp(forward_batch: Any) -> bool:
    """Whether PR-691-style state composition can handle this KDA batch."""
    metadata = getattr(forward_batch, "attn_cp_metadata", None)
    has_nonempty_zigzag_blocks = bool(
        metadata is not None
        and getattr(metadata, "split_list", None)
        and min(metadata.split_list) > 0
    )
    return bool(kda_use_prefill_cp(forward_batch) and has_nonempty_zigzag_blocks)


def _validate_zigzag_metadata(metadata: Any, cp_size: int) -> None:
    required = (
        "bs",
        "split_list",
        "cp_reverse_index",
        "reverse_split_len",
        "per_rank_actual_token",
        "max_rank_len",
    )
    missing = [name for name in required if getattr(metadata, name, None) is None]
    if missing:
        raise NotImplementedError(
            "KDA prefill context parallel currently requires the zigzag CP "
            f"strategy; metadata is missing {missing}."
        )
    if len(metadata.per_rank_actual_token) != cp_size:
        raise ValueError(
            "KDA CP metadata/group size mismatch: "
            f"metadata={len(metadata.per_rank_actual_token)}, cp_size={cp_size}"
        )


def _natural_block_owner_rank(block_id: int, cp_size: int) -> int:
    if block_id < cp_size:
        return block_id
    return 2 * cp_size - block_id - 1


def _build_affine_steps(
    *,
    cp_size: int,
    batch_size: int,
    local_segment_slots: tuple[int, ...],
    fixed_segment_sources: tuple[int, ...],
) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...]], ...]:
    """Precompute the natural-order affine/conv execution plan once per batch."""
    blocks_per_request = 2 * cp_size
    local_slot_to_index = {
        fixed_slot: local_index
        for local_index, fixed_slot in enumerate(local_segment_slots)
    }
    steps = []
    for block_id in range(blocks_per_request):
        owner_rank = _natural_block_owner_rank(block_id, cp_size)
        for part_id in range(2):
            fixed_slots = tuple(
                ((request_id * blocks_per_request + block_id) * 2 + part_id)
                for request_id in range(batch_size)
            )
            source_segments = tuple(
                fixed_segment_sources[fixed_slot] for fixed_slot in fixed_slots
            )
            if all(source_segment < 0 for source_segment in source_segments):
                continue
            local_indices = tuple(
                local_slot_to_index.get(fixed_slot, -1) for fixed_slot in fixed_slots
            )
            steps.append((owner_rank, fixed_slots, source_segments, local_indices))
    return tuple(steps)


def build_kda_fla_cp_context(
    forward_batch: Any,
    *,
    device: torch.device,
    group: Optional[Any] = None,
) -> KDAFLACPContext:
    """Build local segment offsets for a zigzag-sharded KDA invocation.

    A tracked radix boundary may fall inside one natural zigzag block.  Split
    only that block at the exact cache boundary, making the recurrent and
    convolution state at the boundary directly observable without gathering
    token activations.
    """
    if group is None:
        cached = getattr(forward_batch, "kda_fla_cp_context", None)
        if cached is not None:
            return cached
    parallel = get_parallel()
    cp_size = parallel.attn_cp_size
    cp_rank = parallel.attn_cp_rank
    metadata = forward_batch.attn_cp_metadata
    _validate_zigzag_metadata(metadata, cp_size)
    bs = int(metadata.bs)
    blocks_per_request = 2 * cp_size
    split_list = tuple(int(length) for length in metadata.split_list)
    if min(split_list) <= 0:
        raise ValueError("KDA FLA CP requires every zigzag block to be non-empty")

    track_mask_tensor = getattr(forward_batch, "mamba_track_mask", None)
    if track_mask_tensor is None:
        track_mask = [False] * bs
    else:
        track_mask = [
            bool(value) for value in track_mask_tensor.detach().cpu().tolist()[:bs]
        ]
        if len(track_mask) != bs:
            raise ValueError("KDA FLA CP track mask does not match batch size")

    track_offsets = [-1] * bs
    if any(track_mask):
        track_seqlens = getattr(forward_batch, "mamba_track_seqlens", None)
        track_indices = getattr(forward_batch, "mamba_track_indices", None)
        if track_seqlens is None or track_indices is None:
            raise ValueError("KDA FLA CP radix tracking metadata is incomplete")
        track_seqlens_cpu = track_seqlens.detach().cpu().tolist()[:bs]
        prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if prefix_lens_cpu is None:
            prefix_lens_cpu = (
                forward_batch.extend_prefix_lens.detach().cpu().tolist()[:bs]
            )
        cache_chunk_size = get_global_server_args().mamba_cache_chunk_size
        for request_id, should_track in enumerate(track_mask):
            if not should_track:
                continue
            reported_extend_offset = int(track_seqlens_cpu[request_id]) - int(
                prefix_lens_cpu[request_id]
            )
            # The scheduler may add one to force an intermediate-h lookup.  The
            # actual radix state is always the preceding cache-chunk boundary,
            # which is exactly the same alignment used for the conv checkpoint.
            track_offsets[request_id] = (
                reported_extend_offset // cache_chunk_size
            ) * cache_chunk_size

    fixed_segment_lens = [0] * (bs * blocks_per_request * 2)
    track_after_slots = [-1] * bs
    for request_id in range(bs):
        cursor = 0
        request_total = sum(
            split_list[
                request_id * blocks_per_request : (request_id + 1) * blocks_per_request
            ]
        )
        track_offset = track_offsets[request_id]
        if track_mask[request_id] and not 0 < track_offset <= request_total:
            raise ValueError(
                "KDA FLA CP radix checkpoint is outside the extend range: "
                f"request={request_id}, offset={track_offset}, extend={request_total}."
            )
        for block_id in range(blocks_per_request):
            block_len = split_list[request_id * blocks_per_request + block_id]
            fixed_slot = (request_id * blocks_per_request + block_id) * 2
            block_end = cursor + block_len
            if cursor < track_offset < block_end:
                fixed_segment_lens[fixed_slot] = track_offset - cursor
                fixed_segment_lens[fixed_slot + 1] = block_end - track_offset
                track_after_slots[request_id] = fixed_slot
            else:
                fixed_segment_lens[fixed_slot] = block_len
                if track_offset == block_end:
                    track_after_slots[request_id] = fixed_slot
            cursor = block_end
        if track_mask[request_id] and track_after_slots[request_id] < 0:
            raise ValueError(
                f"KDA FLA CP could not place request {request_id}'s radix checkpoint"
            )

    rank_segment_slots_list = []
    fixed_segment_sources = [-1] * len(fixed_segment_lens)
    for rank in range(cp_size):
        rank_slots = []
        for block_id in (rank, blocks_per_request - rank - 1):
            for request_id in range(bs):
                fixed_slot = (request_id * blocks_per_request + block_id) * 2
                for part_id in range(2):
                    if fixed_segment_lens[fixed_slot + part_id] > 0:
                        fixed_segment_sources[fixed_slot + part_id] = len(rank_slots)
                        rank_slots.append(fixed_slot + part_id)
        rank_segment_slots_list.append(tuple(rank_slots))

    rank_segment_slots = tuple(rank_segment_slots_list)
    local_segment_slots = rank_segment_slots[cp_rank]
    local_segment_lens = tuple(
        fixed_segment_lens[fixed_slot] for fixed_slot in local_segment_slots
    )
    local_cu_seqlens = torch.tensor(
        [0, *accumulate(local_segment_lens)],
        dtype=torch.int32,
        device=device,
    )
    track_state_indices = torch.full((bs,), -1, dtype=torch.int64, device=device)
    track_request_ids = tuple(
        request_id for request_id, should_track in enumerate(track_mask) if should_track
    )
    track_request_indices = torch.tensor(
        track_request_ids,
        dtype=torch.int64,
        device=device,
    )
    if any(track_mask):
        track_indices = forward_batch.mamba_track_indices.to(
            device=device, dtype=torch.int64
        )[:bs]
        mask_device = torch.tensor(track_mask, dtype=torch.bool, device=device)
        track_state_indices = torch.where(
            mask_device, track_indices, track_state_indices
        )

    fixed_segment_sources_tuple = tuple(fixed_segment_sources)
    affine_steps = _build_affine_steps(
        cp_size=cp_size,
        batch_size=bs,
        local_segment_slots=local_segment_slots,
        fixed_segment_sources=fixed_segment_sources_tuple,
    )
    # The fused batch-one merge consumes this compact execution plan directly
    # on device.  Building it once with the CP context avoids reconstructing
    # Python indexing tensors in every KDA layer.  Multi-request batches keep
    # using the general composition path below.
    affine_owner_ranks = None
    affine_source_segments = None
    affine_local_indices = None
    affine_local_steps = ()
    affine_local_steps_tensor = None
    affine_local_output_indices = None
    affine_source_flat_indices = None
    conv_local_tail_steps = None
    conv_local_tail_output_indices = None
    conv_prefix_output_index = -1
    affine_track_step = -1
    if bs == 1:
        affine_owner_ranks = torch.tensor(
            [step[0] for step in affine_steps], dtype=torch.int32, device=device
        )
        affine_source_segments = torch.tensor(
            [step[2][0] for step in affine_steps],
            dtype=torch.int32,
            device=device,
        )
        affine_local_indices = torch.tensor(
            [step[3][0] for step in affine_steps],
            dtype=torch.int32,
            device=device,
        )
        affine_local_steps = tuple(
            step_id for step_id, step in enumerate(affine_steps) if step[3][0] >= 0
        )
        affine_local_steps_tensor = torch.tensor(
            affine_local_steps, dtype=torch.int64, device=device
        )
        affine_local_output_indices = torch.tensor(
            [affine_steps[step_id][3][0] for step_id in affine_local_steps],
            dtype=torch.int64,
            device=device,
        )
        affine_source_flat_indices = affine_owner_ranks.to(torch.int64) * max(
            len(rank_slots) for rank_slots in rank_segment_slots
        ) + affine_source_segments.to(torch.int64)
        conv_tail_steps = []
        conv_tail_output_indices = []
        for step_id in affine_local_steps:
            output_index = affine_steps[step_id][3][0]
            if step_id == 0:
                conv_prefix_output_index = output_index
            else:
                conv_tail_steps.append(step_id - 1)
                conv_tail_output_indices.append(output_index)
        conv_local_tail_steps = torch.tensor(
            conv_tail_steps, dtype=torch.int64, device=device
        )
        conv_local_tail_output_indices = torch.tensor(
            conv_tail_output_indices, dtype=torch.int64, device=device
        )
        if track_request_ids:
            affine_track_step = next(
                (
                    step_id
                    for step_id, step in enumerate(affine_steps)
                    if step[1][0] == track_after_slots[0]
                ),
                -1,
            )
            if affine_track_step < 0:
                raise RuntimeError(
                    "KDA FLA CP could not map the radix checkpoint into the "
                    "affine execution plan"
                )

    context = KDAFLACPContext(
        group=group or parallel.attn_cp_group,
        cp_size=cp_size,
        cp_rank=cp_rank,
        batch_size=bs,
        split_list=split_list,
        local_segment_lens=local_segment_lens,
        local_cu_seqlens=local_cu_seqlens,
        local_segment_slots=local_segment_slots,
        rank_segment_slots=rank_segment_slots,
        fixed_segment_sources=fixed_segment_sources_tuple,
        max_rank_segments=max(len(rank_slots) for rank_slots in rank_segment_slots),
        fixed_segment_lens=tuple(fixed_segment_lens),
        track_after_slots=tuple(track_after_slots),
        track_state_indices=track_state_indices,
        track_request_indices=track_request_indices,
        track_request_ids=track_request_ids,
        local_segment_indices=torch.arange(
            len(local_segment_lens), dtype=torch.int32, device=device
        ),
        local_segment_has_initial_state=torch.ones(
            len(local_segment_lens), dtype=torch.bool, device=device
        ),
        local_segment_lens_cpu=list(local_segment_lens),
        local_segment_ends=tuple(accumulate(local_segment_lens)),
        affine_steps=affine_steps,
        affine_owner_ranks=affine_owner_ranks,
        affine_source_segments=affine_source_segments,
        affine_local_indices=affine_local_indices,
        affine_local_steps=affine_local_steps,
        affine_local_steps_tensor=affine_local_steps_tensor,
        affine_local_output_indices=affine_local_output_indices,
        affine_source_flat_indices=affine_source_flat_indices,
        conv_local_tail_steps=conv_local_tail_steps,
        conv_local_tail_output_indices=conv_local_tail_output_indices,
        conv_prefix_output_index=conv_prefix_output_index,
        affine_track_step=affine_track_step,
    )
    if group is None:
        forward_batch.kda_fla_cp_context = context
    return context


def _get_scratch_buffer(
    context: KDAFLACPContext,
    key: str,
    like: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    buffer = context.scratch_buffers.get(key)
    if (
        buffer is None
        or tuple(buffer.shape) != shape
        or buffer.dtype != like.dtype
        or buffer.device != like.device
    ):
        buffer = like.new_empty(shape)
        context.scratch_buffers[key] = buffer
    return buffer


def _all_gather_fixed_shape(
    local: torch.Tensor, context: KDAFLACPContext, *, scratch_key: str
) -> torch.Tensor:
    gathered_shape = (context.cp_size * local.shape[0], *local.shape[1:])
    gathered = _get_scratch_buffer(
        context, f"{scratch_key}_gathered", local, gathered_shape
    )
    context.group.all_gather_into_tensor(gathered, local.contiguous())
    return gathered.view(context.cp_size, *local.shape)


def _write_state_pool(
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Write valid recurrent states without boolean indexing for batch one.

    Kimi-K3 PCP currently serves one request per CP batch.  In that hot path,
    boolean indexing creates an ``aten::index`` synchronization point in every
    KDA layer.  A masked single-row write is equivalent even for an invalid
    index because it writes the original sentinel row back unchanged.  Keep
    the general multi-request implementation as the correctness fallback.
    """
    valid = state_indices >= 0
    safe_indices = state_indices.clamp_min(0).to(torch.int64)
    if state_indices.numel() == 1:
        old_values = state_pool.index_select(0, safe_indices)
        masked_values = torch.where(
            valid.view(1, *([1] * (values.ndim - 1))),
            values.to(state_pool.dtype),
            old_values,
        )
        state_pool.index_copy_(0, safe_indices, masked_values)
        return

    active_indices = state_indices[valid].to(torch.int64)
    if active_indices.numel() > 0:
        state_pool.index_copy_(
            0,
            active_indices,
            values[valid].to(state_pool.dtype),
        )


def compose_kda_cp_affine_states(
    local_affine: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    context: KDAFLACPContext,
    *,
    state_value_major: bool = False,
) -> torch.Tensor:
    """Compose all zigzag segment transforms and return local initial states.

    ``local_affine`` stores ``[H | M]`` for every local segment such that
    ``state_out = M @ state_in + H``. All ranks gather these compact transforms,
    walk the natural block order, and independently derive identical final
    request states. This is the inference counterpart of FLA PR 691's forward
    pre-processing and merge kernels.
    """
    bs = context.batch_size
    expected_segments = context.num_local_segments
    if local_affine.ndim != 4 or local_affine.shape[0] != expected_segments:
        raise ValueError(
            "KDA FLA CP affine tensor must be [local_segments, heads, K, V+K], "
            f"got {tuple(local_affine.shape)} for bs={bs}."
        )
    key_dim = local_affine.shape[-2]
    value_dim = local_affine.shape[-1] - key_dim
    if value_dim <= 0:
        raise ValueError("KDA FLA CP affine tensor has an invalid value dimension")
    if initial_state_indices.numel() != bs:
        raise ValueError(
            "KDA FLA CP state indices must match batch size: "
            f"indices={initial_state_indices.numel()}, bs={bs}."
        )
    expected_state_shape = (
        (value_dim, key_dim) if state_value_major else (key_dim, value_dim)
    )
    if tuple(initial_state_source.shape[-2:]) != expected_state_shape:
        raise ValueError(
            "KDA FLA CP state shape does not match affine transform: "
            f"state={tuple(initial_state_source.shape[-2:])}, "
            f"expected={expected_state_shape}."
        )

    identity_transform = None
    if expected_segments == context.max_rank_segments:
        # The common no-checkpoint path owns exactly two segments per rank and
        # needs no collective padding or identity materialisation.
        padded_affine = local_affine
    else:
        identity_transform = local_affine.new_zeros(*local_affine.shape[1:])
        identity = torch.eye(
            key_dim, dtype=local_affine.dtype, device=local_affine.device
        ).view(1, key_dim, key_dim)
        identity_transform[..., value_dim:].copy_(identity)
        padded_affine = identity_transform.expand(
            context.max_rank_segments, *identity_transform.shape
        ).clone()
        padded_affine[:expected_segments].copy_(local_affine)
    gathered = _all_gather_fixed_shape(padded_affine, context, scratch_key="affine")
    valid = initial_state_indices >= 0
    safe_indices = initial_state_indices.clamp_min(0).to(torch.int64)
    state = initial_state_source.index_select(0, safe_indices)
    if state_value_major:
        state = state.transpose(-1, -2).contiguous()
    state = state.float()
    state = state * valid.view(bs, 1, 1, 1)
    local_initial = _get_scratch_buffer(
        context,
        "affine_initial",
        state,
        (expected_segments, *state.shape[1:]),
    )
    track_request_ids = context.track_request_ids or tuple(
        request_id
        for request_id, fixed_slot in enumerate(context.track_after_slots)
        if fixed_slot >= 0
    )
    use_fused_merge = bool(
        local_affine.device.type in ("npu", "cuda")
        and bs == 1
        and key_dim <= 128
        and len(context.affine_steps) <= 2 * context.cp_size + 1
        and (not track_request_ids or track_request_ids == (0,))
        and context.affine_owner_ranks is not None
        and context.affine_source_segments is not None
        and context.affine_local_indices is not None
        and (not track_request_ids or context.affine_track_step >= 0)
    )
    if use_fused_merge:
        # Compose the precomputed natural-order plan in one device kernel.  A
        # radix checkpoint may split one zigzag block, so the plan can contain
        # one extra transform and a rank can own three local segments.  The
        # same launch writes local initial states, the final cache state, and
        # the tracked checkpoint state.
        from sgl_kernel_npu.fla.kda_chunk_delta_h import (
            merge_kda_cp_affine_states,
        )

        final_state = _get_scratch_buffer(
            context,
            "affine_final",
            state,
            tuple(state.shape),
        )
        tracked_state = (
            _get_scratch_buffer(
                context,
                "affine_tracked",
                state,
                tuple(state.shape),
            )
            if track_request_ids
            else None
        )
        merge_kda_cp_affine_states(
            gathered,
            state,
            local_initial,
            final_state,
            cp_rank=context.cp_rank,
            owner_ranks=context.affine_owner_ranks,
            source_segments=context.affine_source_segments,
            local_indices=context.affine_local_indices,
            local_steps=context.affine_local_steps,
            tracked_state=tracked_state,
            track_step=context.affine_track_step,
        )
        _write_state_pool(
            initial_state_source,
            initial_state_indices,
            (
                final_state.transpose(-1, -2).contiguous()
                if state_value_major
                else final_state
            ),
        )
        if track_request_ids:
            assert tracked_state is not None
            _write_state_pool(
                initial_state_source,
                context.track_state_indices,
                (
                    tracked_state.transpose(-1, -2).contiguous()
                    if state_value_major
                    else tracked_state
                ),
            )
        return local_initial

    tracked_state = torch.empty_like(state) if track_request_ids else None
    tracked_state_is_set = [False] * bs if track_request_ids else None
    affine_steps = context.affine_steps or _build_affine_steps(
        cp_size=context.cp_size,
        batch_size=bs,
        local_segment_slots=context.local_segment_slots,
        fixed_segment_sources=context.fixed_segment_sources,
    )

    for owner_rank, fixed_slots, source_segments, local_indices in affine_steps:
        if owner_rank == context.cp_rank:
            for request_id, local_index in enumerate(local_indices):
                if local_index >= 0:
                    local_initial[local_index].copy_(state[request_id])
        if bs == 1 and source_segments[0] >= 0:
            # Avoid stack/index materialisation for the serving hot path.
            transforms = gathered[owner_rank, source_segments[0]].unsqueeze(0)
        else:
            if identity_transform is None:
                identity_transform = local_affine.new_zeros(*local_affine.shape[1:])
                identity = torch.eye(
                    key_dim,
                    dtype=local_affine.dtype,
                    device=local_affine.device,
                ).view(1, key_dim, key_dim)
                identity_transform[..., value_dim:].copy_(identity)
            transforms = torch.stack(
                [
                    (
                        gathered[owner_rank, source_segment]
                        if source_segment >= 0
                        else identity_transform
                    )
                    for source_segment in source_segments
                ]
            )
        transforms = transforms.float()
        additive = transforms[..., :value_dim]
        transition = transforms[..., value_dim:]
        state = torch.matmul(transition, state) + additive
        for request_id in track_request_ids:
            fixed_slot = fixed_slots[request_id]
            if context.track_after_slots[request_id] == fixed_slot:
                assert tracked_state is not None
                assert tracked_state_is_set is not None
                tracked_state[request_id].copy_(state[request_id])
                tracked_state_is_set[request_id] = True

    _write_state_pool(
        initial_state_source,
        initial_state_indices,
        state.transpose(-1, -2).contiguous() if state_value_major else state,
    )

    track_request_indices = context.track_request_indices
    if track_request_indices is None:
        track_request_indices = torch.tensor(
            track_request_ids,
            dtype=torch.int64,
            device=initial_state_source.device,
        )
    if track_request_indices.numel() > 0:
        assert tracked_state is not None
        assert tracked_state_is_set is not None
        for request_id in track_request_ids:
            if not tracked_state_is_set[request_id]:
                raise RuntimeError(
                    "KDA FLA CP did not produce a requested radix checkpoint"
                )
        track_pool_indices = context.track_state_indices.index_select(
            0, track_request_indices
        )
        tracked_values = tracked_state.index_select(0, track_request_indices)
        if state_value_major:
            tracked_values = tracked_values.transpose(-1, -2).contiguous()
        initial_state_source.index_copy_(
            0,
            track_pool_indices,
            tracked_values.to(initial_state_source.dtype),
        )
    return local_initial


def prepare_kda_cp_conv_states(
    local_x: torch.Tensor,
    conv_state_source: torch.Tensor,
    cache_indices: torch.Tensor,
    context: KDAFLACPContext,
    has_initial_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build exact initial causal-conv windows for both local zigzag blocks.

    Only ``kernel_size - 1`` raw QKV tokens per segment are gathered. Prefix
    convolution state seeds block 0; gathered tails advance the rolling window
    in natural order. The final full-head state is written on every CP rank.
    """
    if local_x.ndim != 2 or conv_state_source.ndim != 3:
        raise ValueError(
            "KDA FLA CP conv expects local [tokens, channels] and state "
            f"[slots, channels, window], got {tuple(local_x.shape)} and "
            f"{tuple(conv_state_source.shape)}."
        )
    bs = context.batch_size
    window = conv_state_source.shape[-1]
    channels = local_x.shape[-1]
    if conv_state_source.shape[-2] != channels:
        raise ValueError("KDA FLA CP conv channel count does not match cache")
    if cache_indices.numel() != bs:
        raise ValueError("KDA FLA CP conv cache indices do not match batch size")
    if sum(context.local_segment_lens) != local_x.shape[0]:
        raise ValueError(
            "KDA FLA CP conv token count does not match local zigzag segments: "
            f"tokens={local_x.shape[0]}, segments={sum(context.local_segment_lens)}."
        )

    padded_tails = _get_scratch_buffer(
        context,
        "conv_local_tails",
        local_x,
        (context.max_rank_segments, window, channels),
    )
    if len(context.local_segment_ends) == context.num_local_segments and all(
        length >= window for length in context.local_segment_lens
    ):
        for segment_id, segment_end in enumerate(context.local_segment_ends):
            padded_tails[segment_id].copy_(local_x[segment_end - window : segment_end])
    else:
        padded_tails.zero_()
        segment_start = 0
        for segment_id, segment_len in enumerate(context.local_segment_lens):
            take = min(window, segment_len)
            if take:
                segment_end = segment_start + segment_len
                padded_tails[segment_id, -take:].copy_(
                    local_x[segment_end - take : segment_end]
                )
            segment_start += segment_len
    gathered_tails = _all_gather_fixed_shape(padded_tails, context, scratch_key="conv")

    valid = cache_indices >= 0
    use_prefix = valid
    if has_initial_state is not None:
        if has_initial_state.numel() != bs:
            raise ValueError(
                "KDA FLA CP conv initial-state mask does not match batch size"
            )
        use_prefix = valid & has_initial_state.to(device=valid.device, dtype=torch.bool)
    safe_indices = cache_indices.clamp_min(0).to(torch.int64)
    prefix_state = conv_state_source.index_select(0, safe_indices)
    prefix_state = prefix_state * use_prefix.view(bs, 1, 1)
    local_initial = _get_scratch_buffer(
        context,
        "conv_initial",
        prefix_state,
        (context.num_local_segments, channels, window),
    )
    final_states = []
    track_request_ids = context.track_request_ids or tuple(
        request_id
        for request_id, fixed_slot in enumerate(context.track_after_slots)
        if fixed_slot >= 0
    )
    affine_steps = context.affine_steps or _build_affine_steps(
        cp_size=context.cp_size,
        batch_size=bs,
        local_segment_slots=context.local_segment_slots,
        fixed_segment_sources=context.fixed_segment_sources,
    )
    use_direct_conv_plan = bool(
        bs == 1
        and context.affine_owner_ranks is not None
        and context.affine_source_segments is not None
        and context.affine_local_steps_tensor is not None
        and context.affine_local_output_indices is not None
        and context.affine_source_flat_indices is not None
        and context.conv_local_tail_steps is not None
        and context.conv_local_tail_output_indices is not None
        and context.affine_local_steps_tensor.numel() == context.num_local_segments
        and (not track_request_ids or track_request_ids == (0,))
        and (not track_request_ids or context.affine_track_step >= 0)
        and all(
            step[2][0] >= 0 and context.fixed_segment_lens[step[1][0]] >= window
            for step in affine_steps
        )
    )
    if use_direct_conv_plan:
        # For long-prefill segments the complete conv window after a segment
        # is exactly that segment's gathered tail. The state before step i is
        # therefore either the radix-prefix state (i=0) or tail(i-1). Build
        # every local initial state, final cache state, and optional radix
        # checkpoint with one plan gather instead of a Python cat/copy loop.
        ordered_tails = (
            gathered_tails.flatten(0, 1)
            .index_select(0, context.affine_source_flat_indices)
            .transpose(1, 2)
        )
        if context.conv_prefix_output_index >= 0:
            local_initial[context.conv_prefix_output_index].copy_(prefix_state[0])
        if context.conv_local_tail_steps.numel() > 0:
            local_initial.index_copy_(
                0,
                context.conv_local_tail_output_indices,
                ordered_tails.index_select(0, context.conv_local_tail_steps),
            )
        _write_state_pool(
            conv_state_source,
            cache_indices,
            ordered_tails[-1:],
        )
        if track_request_ids:
            _write_state_pool(
                conv_state_source,
                context.track_state_indices,
                ordered_tails[
                    context.affine_track_step : context.affine_track_step + 1
                ],
            )
        return local_initial

    tracked_states = torch.empty_like(prefix_state) if track_request_ids else None
    tracked_state_is_set = [False] * bs if track_request_ids else None
    rolling_scratch = _get_scratch_buffer(
        context,
        "conv_rolling",
        prefix_state,
        (bs, 2, channels, window),
    )

    for request_id in range(bs):
        rolling = prefix_state[request_id]
        rolling_step = 0
        for owner_rank, fixed_slots, source_segments, local_indices in affine_steps:
            fixed_slot = fixed_slots[request_id]
            local_index = local_indices[request_id]
            if local_index >= 0:
                local_initial[local_index].copy_(rolling)
            take = min(window, context.fixed_segment_lens[fixed_slot])
            if take:
                source_segment = source_segments[request_id]
                if source_segment < 0:
                    raise RuntimeError(
                        "KDA FLA CP conv segment has no owning rank source"
                    )
                tail = gathered_tails[owner_rank, source_segment, -take:].transpose(
                    0, 1
                )
                next_rolling = rolling_scratch[request_id, rolling_step % 2]
                if take >= window:
                    next_rolling.copy_(tail[..., -window:])
                else:
                    next_rolling[..., :-take].copy_(rolling[..., take:])
                    next_rolling[..., -take:].copy_(tail)
                rolling = next_rolling
                rolling_step += 1
            if context.track_after_slots[request_id] == fixed_slot:
                assert tracked_states is not None
                assert tracked_state_is_set is not None
                tracked_states[request_id].copy_(rolling)
                tracked_state_is_set[request_id] = True
        final_states.append(rolling)

    final_states = (
        final_states[0].unsqueeze(0) if bs == 1 else torch.stack(final_states)
    )
    _write_state_pool(conv_state_source, cache_indices, final_states)

    track_request_indices = context.track_request_indices
    if track_request_indices is None:
        track_request_indices = torch.tensor(
            track_request_ids,
            dtype=torch.int64,
            device=conv_state_source.device,
        )
    if track_request_indices.numel() > 0:
        assert tracked_states is not None
        assert tracked_state_is_set is not None
        for request_id in track_request_ids:
            if not tracked_state_is_set[request_id]:
                raise RuntimeError(
                    "KDA FLA CP did not produce a requested conv checkpoint"
                )
        track_pool_indices = context.track_state_indices.index_select(
            0, track_request_indices
        )
        conv_state_source.index_copy_(
            0,
            track_pool_indices,
            tracked_states.index_select(0, track_request_indices).to(
                conv_state_source.dtype
            ),
        )
    return local_initial


__all__ = [
    "KDAFLACPContext",
    "build_kda_fla_cp_context",
    "compose_kda_cp_affine_states",
    "kda_use_fla_prefill_cp",
    "kda_use_prefill_cp",
    "prepare_kda_cp_conv_states",
]
