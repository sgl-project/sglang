# SPDX-License-Identifier: Apache-2.0
"""Cube sparse attention mask machinery.

This module owns cube label layout, precomputation, condition-event
validation, and per-step top-k masks. Kernel-specific code is intentionally
kept in the backend; the padded layout gives every cube label one or more
``cube_token_size`` physical blocks for block-sparse kernels to consume
directly.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class PackedStreams:
    """The per-modality token streams of one packed batch.

    These five index tensors plus the four per-sample condition metadata lists
    always travel together and are only ever read as a set: they are the caller
    describing *what is in the packed sequence*, as distinct from ``cube_size``
    and ``position_ids``, which describe how to grid it.  Grouping them keeps
    the modality contract in one place, so adding a stream is a change to this
    class rather than to every signature along the path.

    ``text``, ``cond_image``, ``latent``, ``cond_audio`` and ``audio`` are flat
    token-index tensors into the packed sequence.  The four ``cond_*`` lists
    carry one entry per sample and default to empty via
    :func:`_normalize_cube_sample_metadata`.
    """

    text: torch.Tensor
    cond_image: torch.Tensor
    latent: torch.Tensor
    cond_audio: torch.Tensor
    audio: torch.Tensor
    cond_image_shapes: list | None = None
    cond_image_roles: list | None = None
    cond_event_orders: list | None = None
    cond_audio_stream_lens: list | None = None

    def as_index_tuple(self):
        """The five streams in packed order, for uniform per-stream handling."""
        return (self.text, self.cond_image, self.latent, self.cond_audio, self.audio)


@dataclass(frozen=True)
class CubeLayout:
    """Request-static cube layout, fully determined by ``precompute_cube_attention``.

    Every field here is an invariant of the packed sequence: it is computed once
    per request and never mutated afterwards.  Per-forward mutable state lives in
    :class:`CubeRuntime` instead, so a reader can tell at a glance which
    values exist before the first forward and which appear only during it.

    Three index spaces meet in this object, and conflating the first two is the
    main bug risk.  A tensor's *leading dimension* tells you which space it
    lives in, so that is called out per field below:

    * **semantic label** (``L = num_labels``) -- one entry per occupied
      ``(t, x, y)`` cube bin.  TopK selection happens entirely in this space.
    * **physical block** (``B = num_blocks``) -- one entry per FlexAttention
      block of ``cube_token_size`` tokens.  A semantic label owns
      ``label_block_counts[label]`` physical blocks, which exceeds one whenever
      a joint keyframe shares a target cube, so ``B >= L`` always.
    * **token** -- one entry per packed or cube-padded token.

    The naming rule: ``label_*`` fields are *indexed by* semantic label,
    ``block_labels`` maps the other way (physical block to owning label), and
    ``*_physical_layout`` dicts are FlexAttention KV descriptors in block space.
    ``num_labels`` and ``num_blocks`` are both plain counts, which is exactly
    why they read alike -- check which space a value came from before using one
    to size the other.
    """

    # ── semantic-label space (leading dim L, unless noted) ───────────────
    num_labels: int
    """``L`` -- the number of semantic cubes. Not interchangeable with ``num_blocks``."""
    topk_mask: torch.Tensor
    """``[L, L]`` bool: per-sample TopK candidate pool. Dense labels excluded."""
    base_block_mask: torch.Tensor
    """``[L, L]`` bool: always-visible edges (self-diagonal + dense row/col)."""
    sparse_label_mask: torch.Tensor
    """``[L]`` bool: which labels take the sparse path at all."""
    label_lengths: torch.Tensor
    """``[L]`` int: real token count per label, the ``segment_reduce`` lengths."""
    label_block_counts: torch.Tensor
    """``[L]`` int: physical blocks owned by each label; ``sum() == num_blocks``."""
    label_block_indices: torch.Tensor
    """``[L, max_label_block_count]`` int: label to its physical block ids, padded."""
    max_label_block_count: int
    """Row width of ``label_block_indices``; ``1`` unless a label spans blocks."""
    topk_semantic_capacity: int
    """Upper bound on selected labels per query label, for buffer sizing."""

    # ── physical-block space (leading dim B) ─────────────────────────────
    num_blocks: int
    """``B`` -- the number of FlexAttention blocks. Not interchangeable with ``num_labels``."""
    block_labels: torch.Tensor
    """``[B]`` int: physical block to owning semantic label (inverse of ``label_block_indices``)."""
    base_physical_layout: dict[str, torch.Tensor]
    """FlexAttention KV descriptor for ``base_block_mask``, in block space."""
    dense_physical_layout: dict[str, torch.Tensor]
    """All-visible KV descriptor, returned directly when ``topk_ratio == 1.0``."""

    # ── token space (packed and cube-padded) ────────────────────────────
    real_total_len: int
    """Packed token count before cube padding."""
    padded_seqlen: int
    """``num_blocks * cube_token_size`` -- token count after padding."""
    cube_token_size: int
    """``prod(cube_size)`` -- tokens per physical block; the FlexAttention BLOCK_SIZE."""
    is_real: torch.Tensor
    """``[padded_seqlen]`` int32 (0/1): real token vs. cube padding."""
    pad_indices: torch.Tensor
    """Positions of the padding slots, for zeroing padded Q/K/V rows."""
    sorted_real_indices: torch.Tensor
    """``[real_total_len]``: packed index of each token in cube-sorted order."""
    expand_indices: torch.Tensor
    """``[real_total_len]``: packed position to its cube-padded position."""
    gather_indices: torch.Tensor
    """``[padded_seqlen]``: cube-padded position to packed position (0 on pads)."""


@dataclass
class CubeRuntime:
    """Per-forward mutable state, absent until the first ``forward`` call.

    ``pad_score_mod`` is installed by the backend once the layout is known;
    the KV buffers are allocated lazily on the first sparse step and then
    reused, so their shapes double as a guard against a changing head count.
    """

    pad_score_mod: Callable[..., Any] | None = None
    kv_num_blocks_buffer: torch.Tensor | None = None
    kv_indices_buffer: torch.Tensor | None = None


@dataclass
class CubePrecomputed:
    """The cube metadata handed to the kernel: static layout + live buffers."""

    layout: CubeLayout
    runtime: CubeRuntime = field(default_factory=CubeRuntime)


def normalize_condition_event_order(
    events, visual_count, audio_count, *, allow_audio_subset=False
):
    normalized = []
    for event in events:
        event_type, index = event[0], event[1]
        event_hash = event[2] if len(event) > 2 else ""
        normalized.append((event_type, index, event_hash))
    event_types = {event_type for event_type, _, _ in normalized}
    unsupported = event_types - {"imgvid", "audio"}
    if unsupported:
        raise ValueError(f"unsupported condition event types: {sorted(unsupported)}")
    imgvid_indices = [
        index for item_type, index, _ in normalized if item_type == "imgvid"
    ]
    if imgvid_indices != list(range(visual_count)):
        raise ValueError(
            f"condition imgvid indices {imgvid_indices} do not cover "
            f"{visual_count} tensors"
        )
    audio_indices = [
        index for item_type, index, _ in normalized if item_type == "audio"
    ]
    if allow_audio_subset:
        out_of_range = [
            index for index in audio_indices if not 0 <= index < audio_count
        ]
        if out_of_range:
            raise ValueError(
                f"condition audio indices {out_of_range} out of range for "
                f"{audio_count} tensors"
            )
        if len(set(audio_indices)) != len(audio_indices):
            raise ValueError(
                f"condition audio indices {audio_indices} contain duplicates"
            )
    elif audio_indices != list(range(audio_count)):
        raise ValueError(
            f"condition audio indices {audio_indices} do not cover "
            f"{audio_count} tensors"
        )
    return normalized


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _cube_token_size(cube_size):
    cube_size = tuple(int(value) for value in cube_size)
    if len(cube_size) != 3 or any(value <= 0 for value in cube_size):
        raise ValueError(f"local_cube_size must be a positive 3D size, got {cube_size}")
    token_size = math.prod(cube_size)
    if token_size & (token_size - 1):
        raise ValueError(
            "local_cube_size product must be a power of two for FlexAttention, "
            f"got {cube_size} ({token_size} tokens)"
        )
    return token_size


def _normalize_cube_visual_shape(shape):
    shape = tuple(int(value) for value in shape)
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError(
            f"cube attention visual shapes must be positive 3D shapes, got {shape}"
        )
    return shape


_COND_IMAGE_ROLES = frozenset({"joint_cube", "independent_cube", "dense_prefix"})


def _normalize_cond_image_roles(roles, visual_count, sample_idx):
    roles = tuple(roles)
    if len(roles) != visual_count:
        raise ValueError(
            f"cube attention sample {sample_idx} has {visual_count} condition "
            f"visual streams but {len(roles)} condition roles"
        )
    unsupported = sorted(set(roles) - _COND_IMAGE_ROLES)
    if unsupported:
        raise ValueError(
            f"cube attention sample {sample_idx} has unsupported condition "
            f"visual roles: {unsupported}"
        )
    return roles


def _cube_sample_segments(indices, cu_seqlens):
    bounds = torch.searchsorted(indices, cu_seqlens).tolist()
    return [indices[start:end] for start, end in zip(bounds[:-1], bounds[1:])]


def _group_cube_visual_segment(indices, shape, cube_size, sample_idx, segment_name):
    expected = math.prod(shape)
    if indices.numel() != expected:
        raise ValueError(
            f"cube attention {segment_name} for sample {sample_idx} has "
            f"{indices.numel()} tokens, expected {expected} for shape {shape}"
        )
    cube_counts = tuple(_ceil_div(dim, extent) for dim, extent in zip(shape, cube_size))
    linear_indices = torch.arange(
        expected,
        dtype=torch.long,
        device=indices.device,
    )
    block_labels = torch.zeros_like(linear_indices)
    for coordinate, extent, count in zip(
        torch.unravel_index(linear_indices, shape),
        cube_size,
        cube_counts,
    ):
        block_labels = block_labels * count + coordinate // extent
    ordered_labels, order = torch.sort(block_labels, stable=True)
    return indices.index_select(0, order), ordered_labels, math.prod(cube_counts)


def _rank_position_axis(values):
    """Map one floating position axis to stable, zero-based unique ranks."""
    order = torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    changed = torch.empty_like(sorted_values, dtype=torch.bool)
    changed[0] = True
    changed[1:] = sorted_values[1:] != sorted_values[:-1]
    sorted_ranks = changed.to(torch.long).cumsum(0) - 1
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks


def _group_joint_cube_visual_segment(
    indices,
    position_ids,
    cube_size,
    sample_idx,
):
    """Group target + embedded keyframes on their shared position grid.

    Floating RoPE coordinates are stably ranked per axis, so a keyframe with
    the same temporal/spatial position as a target frame receives the same
    semantic cube label.  A semantic label may consequently contain more than
    one physical block.
    """
    if indices.numel() == 0:
        raise ValueError(
            f"cube attention joint visual stream for sample {sample_idx} is empty"
        )
    positions = position_ids.index_select(0, indices)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            "cube attention img_position_ids must have shape [sequence, 3], "
            f"got {tuple(position_ids.shape)}"
        )
    ranked_axes = tuple(_rank_position_axis(positions[:, axis]) for axis in range(3))
    cube_counts = tuple(
        int(axis.max().item()) // extent + 1
        for axis, extent in zip(ranked_axes, cube_size)
    )
    block_labels = torch.zeros(indices.numel(), dtype=torch.long, device=indices.device)
    for coordinate, extent, count in zip(ranked_axes, cube_size, cube_counts):
        block_labels = block_labels * count + coordinate // extent
    ordered_labels, order = torch.sort(block_labels, stable=True)
    return indices.index_select(0, order), ordered_labels, math.prod(cube_counts)


def _group_cube_1d_segment(indices, cube_token_size):
    local_labels = (
        torch.arange(
            indices.numel(),
            dtype=torch.long,
            device=indices.device,
        )
        // cube_token_size
    )
    return indices, local_labels, _ceil_div(indices.numel(), cube_token_size)


def _normalize_cube_sample_metadata(values, num_samples, name):
    if values is None:
        return [()] * num_samples
    if len(values) != num_samples:
        raise ValueError(
            f"cube attention received {len(values)} {name} entries for "
            f"{num_samples} samples"
        )
    return [() if value is None else value for value in values]


def _split_cube_streams(indices, stream_sizes, sample_idx, stream_name):
    streams = []
    offset = 0
    for stream_size in stream_sizes:
        stream_size = int(stream_size)
        if stream_size < 0:
            raise ValueError(
                f"cube attention {stream_name} stream sizes must be nonnegative"
            )
        streams.append(indices[offset : offset + stream_size])
        offset += stream_size
    if offset != indices.numel():
        raise ValueError(
            f"cube attention {stream_name} streams for sample {sample_idx} cover "
            f"{offset} tokens, but packing contains {indices.numel()} tokens"
        )
    return streams


def _pack_block_rows(block_mask):
    """Pack a head-independent ``[Q, KV]`` mask without sorting."""
    counts = block_mask.sum(dim=-1, dtype=torch.int32)
    capacity = int(counts.max().item()) if counts.numel() else 0
    indices = torch.zeros(
        block_mask.shape[0], capacity, dtype=torch.int32, device=block_mask.device
    )
    if capacity:
        positions = block_mask.to(torch.int32).cumsum(dim=-1) - 1
        row_ids, column_ids = block_mask.nonzero(as_tuple=True)
        indices[row_ids, positions[row_ids, column_ids].to(torch.long)] = column_ids.to(
            torch.int32
        )
    return counts[None, None], indices[None, None]


def _build_physical_base_layouts(
    base_block_mask,
    dense_attention_mask,
    block_labels,
):
    def pack_as_full(semantic_mask):
        physical = semantic_mask[block_labels[:, None], block_labels[None, :]]
        full_counts, full_indices = _pack_block_rows(physical)
        empty_counts = torch.zeros_like(full_counts)
        # FlexAttention's CUDA kernel still expects a valid KV-index pointer
        # when every non-full count is zero. Keep one unread placeholder slot
        # instead of passing a zero-storage tensor.
        empty_indices = torch.zeros(
            *full_indices.shape[:-1],
            1,
            dtype=torch.int32,
            device=full_indices.device,
        )
        return {
            "full_kv_num_blocks": full_counts,
            "full_kv_indices": full_indices,
            "kv_num_blocks": empty_counts,
            "kv_indices": empty_indices,
        }

    return pack_as_full(base_block_mask), pack_as_full(dense_attention_mask)


def _raise_for_unoccupied_labels(occupied_labels, sample_label_ranges):
    """Reject cube-label allocations that left grid cells without tokens.

    Joint grouping sizes each sample's label range from the full 3D grid, so
    an unoccupied label means a ``joint_cube`` condition stream introduced
    coordinates outside the target's densely tiled grid.  That is an upstream
    role-assignment error, not a supported layout: such a stream must be
    declared ``independent_cube`` (or ``dense_prefix``) instead.
    """
    occupied = set(occupied_labels.tolist())
    for sample_idx, (start, end) in enumerate(sample_label_ranges):
        missing = [label for label in range(start, end) if label not in occupied]
        if missing:
            raise ValueError(
                f"cube attention sample {sample_idx} allocated labels "
                f"[{start}, {end}) but {len(missing)} of them received no "
                f"tokens (first missing: {missing[0]}). A joint_cube "
                "condition visual stream does not share the target's "
                "position grid; declare it independent_cube or dense_prefix "
                "instead."
            )
    raise ValueError(
        "cube attention allocated labels outside every sample range; "
        f"occupied {len(occupied)} labels for ranges {sample_label_ranges}"
    )


def _build_cube_segment_layout(
    sample_shapes,
    cu_seqlens,
    real_total_len,
    cube_size,
    device,
    streams,
    position_ids,
):
    cube_token_size = math.prod(cube_size)
    num_samples = len(sample_shapes)
    if cu_seqlens.numel() != num_samples + 1:
        raise ValueError(
            f"cube attention received {cu_seqlens.numel() - 1} packed sequences "
            f"for {num_samples} target shapes"
        )
    cond_image_shapes = _normalize_cube_sample_metadata(
        streams.cond_image_shapes,
        num_samples,
        "condition-shape",
    )
    cond_image_roles = _normalize_cube_sample_metadata(
        streams.cond_image_roles,
        num_samples,
        "condition-role",
    )
    cond_event_orders = _normalize_cube_sample_metadata(
        streams.cond_event_orders,
        num_samples,
        "condition-event",
    )
    cond_audio_stream_lens = _normalize_cube_sample_metadata(
        streams.cond_audio_stream_lens,
        num_samples,
        "condition-audio",
    )
    (
        text_segments,
        cond_image_segments,
        latent_segments,
        cond_audio_segments,
        audio_segments,
    ) = [
        _cube_sample_segments(
            indices.to(device=device, dtype=torch.long),
            cu_seqlens,
        )
        for indices in streams.as_index_tuple()
    ]

    cube_labels = torch.full(
        (real_total_len,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    label_offset = 0
    sample_label_ranges = []
    sparse_labels = []
    ordered_segments = []

    def add_segment(ordered_indices, local_block_labels, num_blocks, *, sparse):
        nonlocal label_offset
        if ordered_indices.numel() == 0:
            return
        cube_labels[ordered_indices] = (local_block_labels + label_offset).to(
            torch.int32
        )
        ordered_segments.append(ordered_indices)
        sparse_labels.extend([sparse] * num_blocks)
        label_offset += num_blocks

    for sample_idx in range(num_samples):
        text = text_segments[sample_idx]
        cond_image = cond_image_segments[sample_idx]
        latent = latent_segments[sample_idx]
        cond_audio = cond_audio_segments[sample_idx]
        target_audio = audio_segments[sample_idx]
        sample_label_start = label_offset

        cond_audio_streams = _split_cube_streams(
            cond_audio,
            cond_audio_stream_lens[sample_idx],
            sample_idx,
            "condition audio",
        )
        visual_shapes = [
            _normalize_cube_visual_shape(shape)
            for shape in cond_image_shapes[sample_idx]
        ]
        raw_visual_streams = _split_cube_streams(
            cond_image,
            [math.prod(shape) for shape in visual_shapes],
            sample_idx,
            "condition visual",
        )
        visual_roles = _normalize_cond_image_roles(
            cond_image_roles[sample_idx], len(visual_shapes), sample_idx
        )

        events = normalize_condition_event_order(
            cond_event_orders[sample_idx],
            visual_count=len(raw_visual_streams),
            audio_count=len(cond_audio_streams),
            allow_audio_subset=True,
        )
        add_segment(
            *_group_cube_1d_segment(text, cube_token_size),
            sparse=False,
        )
        listed_audio = set()
        for event_type, event_idx, _ in events:
            event_idx = int(event_idx)
            if event_type == "audio":
                listed_audio.add(event_idx)
                add_segment(
                    *_group_cube_1d_segment(
                        cond_audio_streams[event_idx],
                        cube_token_size,
                    ),
                    sparse=False,
                )
            else:
                shape = visual_shapes[event_idx]
                role = visual_roles[event_idx]
                if role == "joint_cube":
                    continue
                if role == "independent_cube" and (len(shape) != 3 or shape[0] <= 1):
                    raise ValueError(
                        "independent_cube condition visual streams must have "
                        f"a genuine 3D shape, got {shape}"
                    )
                if role == "independent_cube":
                    grouped = _group_cube_visual_segment(
                        raw_visual_streams[event_idx],
                        shape,
                        cube_size,
                        sample_idx,
                        f"condition visual stream {event_idx}",
                    )
                else:
                    grouped = _group_cube_1d_segment(
                        raw_visual_streams[event_idx], cube_token_size
                    )
                add_segment(*grouped, sparse=role == "independent_cube")
        for stream_idx, stream in enumerate(cond_audio_streams):
            if stream_idx not in listed_audio:
                add_segment(
                    *_group_cube_1d_segment(stream, cube_token_size),
                    sparse=False,
                )
        add_segment(
            *_group_cube_1d_segment(target_audio, cube_token_size),
            sparse=False,
        )

        target_shape = sample_shapes[sample_idx]
        expected_target_tokens = math.prod(target_shape)
        if latent.numel() != expected_target_tokens:
            raise ValueError(
                f"cube attention target visual for sample {sample_idx} has "
                f"{latent.numel()} tokens, expected {expected_target_tokens} "
                f"for shape {target_shape}"
            )
        joint_streams = [
            stream
            for stream, role in zip(raw_visual_streams, visual_roles)
            if role == "joint_cube"
        ]
        joint_indices = torch.cat([*joint_streams, latent])
        add_segment(
            *_group_joint_cube_visual_segment(
                joint_indices,
                position_ids,
                cube_size,
                sample_idx,
            ),
            sparse=True,
        )
        sample_label_ranges.append((sample_label_start, label_offset))

    sort_idx = torch.cat(ordered_segments)
    if sort_idx.numel() != real_total_len:
        unassigned = torch.nonzero(cube_labels < 0, as_tuple=False).flatten()
        first_token = int(unassigned[0])
        first_sample = int(torch.searchsorted(cu_seqlens, first_token, right=True)) - 1
        raise ValueError(
            f"cube attention modality segments cover {sort_idx.numel()} of "
            f"{real_total_len} packed tokens; first unassigned token {first_token} "
            f"in sample {first_sample}"
        )

    return (
        cube_labels,
        sort_idx,
        sample_label_ranges,
        torch.tensor(sparse_labels, dtype=torch.bool, device=device),
    )


def precompute_cube_attention(
    sample_shapes,
    cu_seqlens,
    total_len,
    cube_size,
    device,
    streams,
    position_ids,
    max_sparse_topk_ratio,
):
    """Build the request-static cube layout for one packed batch.

    ``streams`` is a :class:`PackedStreams` describing the per-modality token
    streams; ``cube_size`` and ``position_ids`` describe the 3D grid they are
    ranked on.
    """
    sample_shapes = [_normalize_cube_visual_shape(shape) for shape in sample_shapes]
    cu_seqlens = cu_seqlens.to(device=device, dtype=torch.long)
    real_total_len = int(cu_seqlens[-1].item())
    if real_total_len > total_len:
        raise ValueError(
            f"cube attention real length {real_total_len} exceeds total length {total_len}"
        )
    cube_token_size = _cube_token_size(cube_size)
    cube_size = tuple(int(value) for value in cube_size)
    position_ids = position_ids.to(device=device)
    if position_ids.ndim != 2 or position_ids.shape != (total_len, 3):
        raise ValueError(
            "cube attention img_position_ids must have shape "
            f"[{total_len}, 3], got {tuple(position_ids.shape)}"
        )
    cube_labels, sort_idx, sample_label_ranges, sparse_label_mask = (
        _build_cube_segment_layout(
            sample_shapes,
            cu_seqlens,
            real_total_len,
            cube_size,
            device,
            streams,
            position_ids,
        )
    )
    num_labels = sum(end - start for start, end in sample_label_ranges)
    dead_label = num_labels

    topk_mask = torch.zeros(num_labels, num_labels, dtype=torch.bool, device=device)
    base_block_mask = torch.zeros(
        num_labels, num_labels, dtype=torch.bool, device=device
    )
    dense_attention_mask = torch.zeros(
        num_labels, num_labels, dtype=torch.bool, device=device
    )
    base_block_mask.fill_diagonal_(True)
    for start, end in sample_label_ranges:
        dense_attention_mask[start:end, start:end] = True
        sample_sparse = sparse_label_mask[start:end]
        topk_mask[start:end, start:end] = sample_sparse.unsqueeze(
            1
        ) & sample_sparse.unsqueeze(0)
        sample_dense = ~sample_sparse
        base_block_mask[start:end, start:end] |= sample_dense.unsqueeze(
            1
        ) | sample_dense.unsqueeze(0)

    sorted_labels = cube_labels[sort_idx]
    occupied_labels, counts_per_label = sorted_labels.unique_consecutive(
        return_counts=True
    )
    if occupied_labels.numel() != num_labels:
        _raise_for_unoccupied_labels(occupied_labels, sample_label_ranges)
    padded_counts = (
        (counts_per_label + cube_token_size - 1) // cube_token_size
    ) * cube_token_size

    padded_offsets = torch.zeros(
        len(padded_counts) + 1, dtype=torch.long, device=device
    )
    padded_offsets[1:] = padded_counts.cumsum(0)
    group_starts = torch.zeros(
        len(counts_per_label) + 1, dtype=torch.long, device=device
    )
    group_starts[1:] = counts_per_label.cumsum(0)

    sorted_positions = torch.arange(real_total_len, device=device)
    group_idx = torch.bucketize(sorted_positions, group_starts[1:], right=True)
    padded_pos_sorted = padded_offsets[group_idx] + (
        sorted_positions - group_starts[group_idx]
    )

    expand_indices = torch.empty(real_total_len, dtype=torch.long, device=device)
    expand_indices[sort_idx] = padded_pos_sorted

    padded_seqlen = int(padded_offsets[-1].item())
    label_block_counts = padded_counts // cube_token_size
    label_block_offsets = torch.zeros(num_labels + 1, dtype=torch.long, device=device)
    label_block_offsets[1:] = label_block_counts.cumsum(0)
    num_blocks = int(label_block_offsets[-1].item())
    block_labels = torch.repeat_interleave(
        torch.arange(num_labels, dtype=torch.long, device=device),
        label_block_counts,
    )
    max_label_block_count = int(label_block_counts.max().item())
    label_block_slots = torch.arange(
        max_label_block_count, dtype=torch.long, device=device
    )
    label_block_indices = label_block_offsets[:-1, None] + label_block_slots
    label_block_indices = torch.where(
        label_block_slots < label_block_counts[:, None],
        label_block_indices,
        torch.zeros_like(label_block_indices),
    )
    gather_indices = torch.zeros(padded_seqlen, dtype=torch.long, device=device)
    gather_indices[padded_pos_sorted] = sort_idx

    padded_cube_labels = torch.full(
        (padded_seqlen,), dead_label, dtype=torch.int32, device=device
    )
    padded_cube_labels[padded_pos_sorted] = sorted_labels

    is_real = (padded_cube_labels != dead_label).to(torch.int32)
    pad_indices = torch.nonzero(is_real == 0, as_tuple=False).squeeze(1)
    label_lengths = counts_per_label.to(torch.long)
    base_physical_layout, dense_physical_layout = _build_physical_base_layouts(
        base_block_mask,
        dense_attention_mask,
        block_labels,
    )
    sparse_sizes = topk_mask.sum(dim=-1)
    if max_sparse_topk_ratio > 0:
        max_selected_counts = (
            sparse_sizes.to(torch.float32) * float(max_sparse_topk_ratio)
        ).to(torch.long)
        max_selected_counts.clamp_(min=1)
        max_selected_counts = torch.minimum(max_selected_counts, sparse_sizes)
        topk_semantic_capacity = int(max_selected_counts.max().item())
    else:
        topk_semantic_capacity = 0

    return CubePrecomputed(
        layout=CubeLayout(
            num_labels=num_labels,
            topk_mask=topk_mask,
            base_block_mask=base_block_mask,
            sparse_label_mask=sparse_label_mask,
            label_lengths=label_lengths,
            label_block_counts=label_block_counts,
            label_block_indices=label_block_indices,
            max_label_block_count=max_label_block_count,
            topk_semantic_capacity=topk_semantic_capacity,
            num_blocks=num_blocks,
            block_labels=block_labels,
            base_physical_layout=base_physical_layout,
            dense_physical_layout=dense_physical_layout,
            real_total_len=real_total_len,
            padded_seqlen=padded_seqlen,
            cube_token_size=cube_token_size,
            is_real=is_real,
            pad_indices=pad_indices,
            sorted_real_indices=sort_idx,
            expand_indices=expand_indices,
            gather_indices=gather_indices,
        )
    )


def _cube_topk_selection(q_real, k_real, precomputed, topk_ratio):
    layout = precomputed.layout
    dim = q_real.shape[-1]

    qk_sorted = torch.cat((q_real, k_real), dim=-1)[layout.sorted_real_indices]
    label_lengths = layout.label_lengths
    qk_pool = torch.segment_reduce(
        qk_sorted, "sum", lengths=label_lengths, axis=0, unsafe=True
    )
    qk_pool /= label_lengths.float().view(-1, 1, 1)

    q_pool, k_pool = torch.split(qk_pool, dim, dim=-1)
    scores = torch.einsum("lhd,mhd->hlm", q_pool, k_pool) * (dim**-0.5)

    candidate_mask = layout.topk_mask
    sparse_sizes = candidate_mask.sum(dim=-1)
    sparse_labels = sparse_sizes > 0
    scores.masked_fill_(~candidate_mask.unsqueeze(0), float("-inf"))

    selected_counts = (sparse_sizes.to(torch.float32) * topk_ratio).to(torch.long)
    selected_counts.clamp_(min=1)
    selected_counts = torch.minimum(selected_counts, sparse_sizes)
    selected_counts = torch.where(
        sparse_labels, selected_counts, torch.zeros_like(selected_counts)
    )

    selected_order = torch.argsort(
        scores,
        dim=-1,
        descending=True,
        stable=True,
    )
    return selected_order, selected_counts


def cube_topk_block_indices(q_real, k_real, precomputed, topk_ratio):
    """Build physical FlexAttention KV rows directly from semantic TopK.

    Semantic TopK expands directly to physical block ids without materializing
    a per-head BxB boolean mask. Static base ids and selected sparse ids are
    merged into one ordered per-head KV prefix; padding is handled by the
    backend score modifier.  Sparse steps intentionally omit full-KV metadata
    entirely so FlexAttention selects its partial-only kernel.

    This is the only production expansion of :func:`_cube_topk_selection`.  The
    test suite expands the same selection into a semantic ``[H, L, L]`` mask
    and asserts the two agree at bool level, because a divergence confined to a
    few blocks can hide under an attention numeric tolerance.
    """
    layout = precomputed.layout
    runtime = precomputed.runtime
    if topk_ratio == 1.0:
        return layout.dense_physical_layout

    selected_order, selected_counts = _cube_topk_selection(
        q_real, k_real, precomputed, topk_ratio
    )
    semantic_capacity = layout.topk_semantic_capacity
    if semantic_capacity <= 0:
        raise ValueError(
            "cube sparse metadata has no compact TopK capacity for a sparse ratio"
        )
    selected_semantic = selected_order[..., :semantic_capacity]
    semantic_rank = torch.arange(semantic_capacity, device=selected_order.device).view(
        1, 1, -1
    )
    selected_valid = semantic_rank < selected_counts.view(1, -1, 1)

    block_labels = layout.block_labels
    selected_semantic = selected_semantic.index_select(1, block_labels)
    selected_valid = selected_valid.index_select(1, block_labels).expand_as(
        selected_semantic
    )
    q_semantic = block_labels.view(1, -1, 1)
    # Deduplicate against the static base set.  This is not just an
    # optimization: the truncation to num_blocks further down is only lossless
    # because base ids and selected ids are disjoint (this line) and
    # label->physical-blocks is a partition, so each row holds at most
    # num_blocks distinct valid ids.  Removing this intersection would make
    # the sort below silently drop real KV blocks whenever
    # base_capacity + selected blocks exceeds num_blocks.
    selected_valid = (
        selected_valid & ~layout.base_block_mask[q_semantic, selected_semantic]
    )

    label_block_indices = layout.label_block_indices
    label_block_counts = layout.label_block_counts
    max_label_blocks = layout.max_label_block_count
    selected_physical = label_block_indices[selected_semantic]
    physical_rank = torch.arange(max_label_blocks, device=selected_order.device).view(
        1, 1, 1, -1
    )
    selected_physical_valid = selected_valid.unsqueeze(-1) & (
        physical_rank < label_block_counts[selected_semantic].unsqueeze(-1)
    )
    selected_physical = selected_physical.flatten(-2)
    selected_physical_valid = selected_physical_valid.flatten(-2)

    base_layout = layout.base_physical_layout
    num_heads = q_real.shape[1]
    num_blocks = layout.num_blocks
    base_counts = base_layout["full_kv_num_blocks"].expand(1, num_heads, -1)[0]
    base_indices = base_layout["full_kv_indices"].expand(1, num_heads, -1, -1)[0]
    base_rank = torch.arange(base_indices.shape[-1], device=q_real.device).view(
        1, 1, -1
    )
    base_valid = base_rank < base_counts.unsqueeze(-1)

    # Block ids are bounded by num_blocks (< 2**31), so int32 is wide enough and
    # halves the transient footprint of the sort below, which is the largest
    # per-step allocation on the sparse path.
    candidate_indices = torch.cat(
        (base_indices.to(torch.int32), selected_physical.to(torch.int32)), dim=-1
    )
    candidate_valid = torch.cat((base_valid, selected_physical_valid), dim=-1)
    packed_kv = torch.where(
        candidate_valid,
        candidate_indices,
        torch.full_like(candidate_indices, num_blocks),
    )
    # Sorting sends the num_blocks sentinel to the tail, so the first
    # num_blocks slots hold every valid id.  The candidate width
    # (base_capacity + topk_semantic_capacity * max_label_block_count) may
    # exceed num_blocks, but the valid count per row cannot: base and selected
    # ids are disjoint and label->blocks is a partition.  That invariant is
    # what makes this truncation lossless rather than a silent block drop; it
    # is asserted in test_compact_kv_rows_never_exceed_block_count.
    packed_kv = torch.sort(packed_kv, dim=-1, stable=True).values[..., :num_blocks]
    packed_kv = torch.where(
        packed_kv < num_blocks,
        packed_kv,
        torch.zeros_like(packed_kv),
    )

    buffer = runtime.kv_indices_buffer
    if buffer is None:
        runtime.kv_num_blocks_buffer = torch.empty(
            1, num_heads, num_blocks, dtype=torch.int32, device=q_real.device
        )
        runtime.kv_indices_buffer = torch.empty(
            1,
            num_heads,
            num_blocks,
            num_blocks,
            dtype=torch.int32,
            device=q_real.device,
        )
    else:
        expected = (1, num_heads, num_blocks, num_blocks)
        if tuple(buffer.shape) != expected:
            raise ValueError(
                "cube attention KV buffer shape changed across calls: "
                f"{tuple(buffer.shape)} vs {expected}"
            )

    kv_counts = runtime.kv_num_blocks_buffer
    kv_indices = runtime.kv_indices_buffer
    selected_physical_counts = selected_physical_valid.sum(dim=-1, dtype=torch.int32)
    kv_counts[0].copy_(base_counts + selected_physical_counts)
    kv_indices[0].copy_(packed_kv.to(torch.int32))

    return {
        "kv_num_blocks": kv_counts,
        "kv_indices": kv_indices,
        "full_kv_num_blocks": None,
        "full_kv_indices": None,
    }
