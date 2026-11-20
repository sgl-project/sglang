"""Index and token bookkeeping for HY4 DFly draft blocks.

A DFly block holds ``query_width`` queries per request: the bonus token at the
anchor position ``p`` (the committed length) followed by mask tokens at
``p + 1 ..``. The draft attends over external sparse rows whose entries are in
the id space the draft attention backend reads: request-local positions, or
physical KV slots when the DSA top-k transform is fused.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.dsa.dsa_backend_mtp_precompute import (
    compute_cu_seqlens,
)
from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
)

# Never allocated to a request; KV writes routed here are dropped.
DFLY_DISCARD_SLOT = 0


def dfly_query_positions(
    *,
    anchors: torch.Tensor,
    sampled: torch.Tensor,
    query_width: int,
    max_context_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[bs, query_width]`` positions and the ``[bs]`` active mask.

    A request drafts only when it sampled a bonus token this step and its whole
    block fits in the context. Inactive requests get positions ``0..N``, which
    keep RoPE in range; their writes and reads must be suppressed separately.
    """
    offsets = torch.arange(query_width, device=anchors.device)
    anchors = anchors.to(torch.int64)
    active = sampled & (anchors + query_width <= max_context_len)
    positions = torch.where(active[:, None], anchors[:, None] + offsets, offsets)
    return positions, active


def dfly_draft_write_slots(
    *, verify_slots: torch.Tensor, active: torch.Tensor
) -> torch.Tensor:
    """Flat draft KV write slots: the verify slots for active requests, else discarded.

    ``verify_slots`` is ``[bs, query_width]``; padded batch rows must be inactive.
    """
    return torch.where(
        active[:, None], verify_slots.to(torch.int64), DFLY_DISCARD_SLOT
    ).flatten()


def dfly_position_ids(
    *,
    positions: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    physical: bool,
) -> torch.Tensor:
    """``positions`` in the draft's sparse-index space; -1 where position < 0."""
    positions = positions.to(torch.int64)
    if physical:
        rows = req_pool_indices.to(torch.int64)[:, None]
        ids = req_to_token[rows, positions.clamp(min=0)].to(torch.int64)
    else:
        ids = positions
    return torch.where(positions >= 0, ids, -1)


def dfly_window_positions(
    *, anchors: torch.Tensor, context_local_window: int
) -> torch.Tensor:
    """``[bs, window]`` positions ``[p - window, p)``; negative before the prompt."""
    offsets = torch.arange(-context_local_window, 0, device=anchors.device)
    return anchors.to(torch.int64)[:, None] + offsets


def build_dfly_sparse_indices(
    *,
    target_rows: torch.Tensor,
    window_ids: torch.Tensor,
    block_ids: torch.Tensor,
    active: torch.Tensor,
    index_topk: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return ``[bs * query_width, index_topk]`` int32 sparse KV ids, written
    into ``out`` when given.

    Each query row reads ``[history | window | causal block | -1 padding]``:
    the target indexer's top-k at the request's last committed row minus the
    window, the ids of ``[p - window, p)``, and the block's ids up to and
    including its own position. Inactive requests read nothing. Valid entries
    are spread over the whole width, so the consumer must scan it all
    (``dfly_sparse_attention_metadata``).
    """
    bs, query_width = block_ids.shape
    topk = target_rows.shape[1]
    window = window_ids.shape[1]
    if target_rows.shape[0] != bs or window_ids.shape[0] != bs:
        raise ValueError(
            f"DFly sparse inputs disagree on batch size: target rows "
            f"{tuple(target_rows.shape)}, window {tuple(window_ids.shape)}, "
            f"block {tuple(block_ids.shape)}"
        )
    if topk + window + query_width > index_topk:
        raise ValueError(
            f"DFly sparse indices need {topk + window + query_width} entries, "
            f"but the draft top-k width is {index_topk}"
        )
    history = target_rows.to(torch.int64)
    invalid = history < 0
    if window:
        # The reference drops history at positions >= p - window. The snapshot
        # holds only positions < p, each with a unique id, so membership is that test.
        sorted_window = window_ids.to(torch.int64).sort(dim=1).values
        found = torch.searchsorted(sorted_window, history.contiguous())
        found = found.clamp(max=window - 1)
        invalid |= sorted_window.gather(1, found) == history
    history = history.masked_fill(invalid, -1)

    if out is None:
        out = torch.empty(
            (bs * query_width, index_topk), dtype=torch.int32, device=block_ids.device
        )
    rows = out.view(bs, query_width, index_topk)
    rows.fill_(-1)
    rows[:, :, :topk] = history[:, None, :]
    rows[:, :, topk : topk + window] = window_ids[:, None, :]
    steps = torch.arange(query_width, device=block_ids.device)
    causal = steps[None, :] <= steps[:, None]
    rows[:, :, topk + window : topk + window + query_width] = torch.where(
        causal[None], block_ids.to(torch.int64)[:, None, :], -1
    )
    rows.masked_fill_(~active[:, None, None], -1)
    return out


def dfly_full_width_lengths(
    *, num_rows: int, index_topk: int, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[num_rows]`` int32 top-k lengths that scan a whole external row, and
    their ``[num_rows + 1]`` cumulative lengths."""
    lengths = torch.full((num_rows,), index_topk, dtype=torch.int32, device=device)
    return lengths, compute_cu_seqlens(lengths)


def dfly_sparse_attention_metadata(
    metadata, *, lengths: torch.Tensor, cu_lengths: torch.Tensor
):
    """Draft DSA metadata whose per-row top-k length covers the external rows.

    The backend derives ``dsa_cache_seqlens_int32`` from the logical sequence
    length, which bounds how far a sparse kernel scans each row. External rows
    keep valid entries past that bound, so every row scans the full width
    (``dfly_full_width_lengths``); the kernel masks the -1 holes.
    """
    return dataclasses.replace(
        metadata, dsa_cache_seqlens_int32=lengths, dsa_cu_seqlens_k=cu_lengths
    )


def split_dfly_drafts_for_verify(
    *, bonus_tokens: torch.Tensor, draft_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[bs, N + 1]`` accept candidates and the flat verify input ids.

    Inactive drafts are -1. They stay -1 in the candidates, so no target
    prediction can accept them, and become 0 in the verify input, which the
    target embedding can read.
    """
    candidates = torch.cat(
        [bonus_tokens.to(torch.int64)[:, None], draft_tokens.to(torch.int64)], dim=1
    )
    return candidates, candidates.clamp(min=0).flatten()


def dfly_source_rows(
    *, row_starts: torch.Tensor, commit_lens: torch.Tensor
) -> torch.Tensor:
    """Row of each request's last committed token in a flat target forward.

    ``row_starts`` is each request's first row; ``commit_lens`` counts its
    committed rows in this forward (extend length, or accepted drafts + 1).
    """
    return row_starts.to(torch.int64) + commit_lens.to(torch.int64) - 1


class DFlyBlock(msgspec.Struct, frozen=True, kw_only=True):
    """One step's draft block: ``[bs * query_width]`` ids and write slots,
    ``[bs, query_width]`` draft and verify positions, ``[bs]`` anchors and
    active mask."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    verify_positions: torch.Tensor
    write_slots: torch.Tensor
    anchors: torch.Tensor
    active: torch.Tensor


def dfly_verify_positions(
    *, anchors: torch.Tensor, query_width: int, max_context_len: int
) -> torch.Tensor:
    """``[bs, query_width]`` verify positions from the anchor ``p``.

    Rows past the context are rejected but still rotated, so they are clamped
    into the RoPE table.
    """
    offsets = torch.arange(query_width, device=anchors.device)
    return (anchors.to(torch.int64)[:, None] + offsets).clamp(max=max_context_len - 1)


def plan_dfly_block(
    *,
    bonus_tokens: torch.Tensor,
    anchors: torch.Tensor,
    sampled: torch.Tensor,
    verify_slots: torch.Tensor,
    mask_token_id: int,
    max_context_len: int,
) -> DFlyBlock:
    """Lay out the draft block over the verify slots ``[bs, query_width]``.

    Only active requests read their bonus token and write KV; every other row
    (no sample this step, or a block past the context) writes the discard slot.
    """
    bs, query_width = verify_slots.shape
    positions, active = dfly_query_positions(
        anchors=anchors,
        sampled=sampled,
        query_width=query_width,
        max_context_len=max_context_len,
    )
    input_ids = torch.full(
        (bs, query_width), mask_token_id, dtype=torch.int64, device=anchors.device
    )
    input_ids[:, 0] = torch.where(active, bonus_tokens.to(torch.int64), 0)
    return DFlyBlock(
        input_ids=input_ids.flatten(),
        positions=positions,
        verify_positions=dfly_verify_positions(
            anchors=anchors, query_width=query_width, max_context_len=max_context_len
        ),
        write_slots=dfly_draft_write_slots(verify_slots=verify_slots, active=active),
        anchors=anchors.to(torch.int64),
        active=active,
    )


def dfly_producer_layers(index_layers: tuple[int, ...]) -> tuple[int, ...]:
    """The distinct target index layers, in layer order: one top-k plane each."""
    return tuple(sorted(set(index_layers)))


def dfly_draft_topk_planes(index_layers: tuple[int, ...]) -> tuple[int, ...]:
    """The top-k plane each draft layer reads, for ``index_layers[i]`` of draft
    layer ``i``; draft layers sharing a target index layer share its plane."""
    producers = dfly_producer_layers(index_layers)
    return tuple(producers.index(layer) for layer in index_layers)


class DFlyProducerCapture:
    """Fixed per-producer copies of the target indexer top-k of one forward.

    Each distinct producer layer owns a ``[max_rows, topk]`` int32 plane. A
    forward begun with ``keep_rows`` (each request's last prefill row) keeps
    those rows; otherwise it keeps every row (a verify block). Each record also
    stamps its plane with the forward's number, a device tensor, so a CUDA
    graph replay, which runs no Python, still proves which producers ran.
    ``finish`` refuses the rows unless every plane holds the number of the
    forward begun last. Under a breakable prefill graph a record is an eager
    break, so each replay keeps the rows of the forward begun before it.
    """

    def __init__(
        self,
        *,
        producer_layers: tuple[int, ...],
        max_rows: int,
        topk: int,
        device: torch.device | str,
    ):
        self._planes = {layer: plane for plane, layer in enumerate(producer_layers)}
        self._rows = torch.full(
            (len(producer_layers), max_rows, topk),
            -1,
            dtype=torch.int32,
            device=device,
        )
        self._forward = torch.zeros(1, dtype=torch.int64, device=device)
        self._stamps = torch.full(
            (len(producer_layers),), -1, dtype=torch.int64, device=device
        )
        self._num_rows: Optional[int] = None
        self._keep_rows: Optional[torch.Tensor] = None

    def begin(self, *, num_rows: int, keep_rows: Optional[torch.Tensor] = None) -> None:
        """Number the coming forward and keep its rows; outside any graph."""
        self.arm(num_rows=num_rows, keep_rows=keep_rows)
        self._forward.add_(1)

    def arm(self, *, num_rows: int, keep_rows: Optional[torch.Tensor] = None) -> None:
        """Keep the coming forward's rows without numbering it. A graph capture
        arms; the ``begin`` before each replay numbers the replayed forward."""
        if num_rows > self._rows.shape[1]:
            raise RuntimeError(
                f"DFly top-k capture holds {self._rows.shape[1]} rows, "
                f"a forward needs {num_rows}"
            )
        self._num_rows = num_rows
        self._keep_rows = keep_rows

    def disarm(self) -> None:
        self._num_rows = None
        self._keep_rows = None

    @eager_on_graph
    def record(self, layer_id: int, topk_indices: Optional[torch.Tensor]) -> None:
        plane = self._planes.get(layer_id)
        # Target forwards no DFly step armed, and non-producer layers, record nothing.
        if self._num_rows is None or plane is None:
            return
        if topk_indices is None:
            raise RuntimeError(f"DFly target index layer {layer_id} produced no top-k")
        if self._keep_rows is not None:
            topk_indices = topk_indices[self._keep_rows]
        expected = (self._num_rows, self._rows.shape[2])
        if tuple(topk_indices.shape) != expected:
            raise RuntimeError(
                f"DFly target index layer {layer_id} produced top-k "
                f"{tuple(topk_indices.shape)}, expected {expected}"
            )
        self._rows[plane, : self._num_rows].copy_(topk_indices)
        self._stamps[plane : plane + 1].copy_(self._forward)

    def finish(self) -> torch.Tensor:
        """``[producers, rows, topk]`` rows of the forward begun last.

        A view of the capture planes: gather from it before the next forward.
        Reading the stamps synchronizes with the device.
        """
        num_rows = self._num_rows
        self.disarm()
        if num_rows is None:
            raise RuntimeError("DFly top-k capture finished without a begun forward")
        stale = (self._stamps != self._forward).tolist()
        skipped = [layer for layer, plane in self._planes.items() if stale[plane]]
        if skipped:
            raise RuntimeError(
                f"DFly target forward skipped index layers {skipped}; the draft "
                "needs every index layer's top-k"
            )
        return self._rows[:, :num_rows]


class VLDFlyRequestState:
    """Target state that seeds each request's next draft block, by pool row.

    Row ``r`` holds the final hidden and every producer's top-k of the target
    row at the request's last committed token, the committed length after it,
    and the request's mRoPE delta. A row seeds a draft only while the pool
    generation and state epoch it was written under, and its committed
    length, still match the request. Row 0 is the pool's padding row.
    """

    def __init__(
        self,
        *,
        num_rows: int,
        hidden_size: int,
        hidden_dtype: torch.dtype,
        num_producers: int,
        topk: int,
        device: torch.device | str,
    ):
        self.hidden = torch.zeros(
            (num_rows, hidden_size), dtype=hidden_dtype, device=device
        )
        self.topk = torch.full(
            (num_producers, num_rows, topk), -1, dtype=torch.int32, device=device
        )
        self.committed_len = torch.full(
            (num_rows,), -1, dtype=torch.int64, device=device
        )
        self.mrope_delta = torch.zeros((num_rows,), dtype=torch.int64, device=device)
        # Host-side, so the validity check needs no device sync.
        self._generation = torch.full((num_rows,), -1, dtype=torch.int64)
        self._epoch = torch.full((num_rows,), -1, dtype=torch.int64)
        self._current_epoch = 0

    def invalidate_all(self) -> None:
        self._current_epoch += 1

    def write(
        self,
        *,
        rows_cpu: torch.Tensor,
        rows: torch.Tensor,
        generations: torch.Tensor,
        hidden: torch.Tensor,
        topk: torch.Tensor,
        committed_len: torch.Tensor,
        mrope_delta: Optional[torch.Tensor] = None,
    ) -> None:
        """``rows_cpu``/``rows`` are the requests' pool rows on host and device;
        ``hidden`` is ``[n, hidden]``, ``topk`` ``[producers, n, topk]``."""
        if bool((rows_cpu <= 0).any()):
            raise RuntimeError("DFly request state never holds the padding row 0")
        if hidden.dtype != self.hidden.dtype:
            raise RuntimeError(
                f"DFly request state keeps {self.hidden.dtype} target hidden, "
                f"got {hidden.dtype}"
            )
        self.hidden[rows] = hidden
        self.topk[:, rows] = topk
        self.committed_len[rows] = committed_len.to(torch.int64)
        if mrope_delta is not None:
            self.mrope_delta[rows] = mrope_delta.to(torch.int64)
        self._generation[rows_cpu] = generations
        self._epoch[rows_cpu] = self._current_epoch

    def live(
        self,
        *,
        rows_cpu: torch.Tensor,
        rows: torch.Tensor,
        generations: torch.Tensor,
        committed_len: torch.Tensor,
    ) -> torch.Tensor:
        """``[n]`` bool on ``rows``' device: whether each row's state is current."""
        current = (self._generation[rows_cpu] == generations) & (
            self._epoch[rows_cpu] == self._current_epoch
        )
        return current.to(rows.device) & (
            self.committed_len[rows] == committed_len.to(torch.int64)
        )


def vl_dfly_buffer_bytes(
    *, spec, hidden_size: int, hidden_dtype: torch.dtype, num_request_rows: int
) -> int:
    """Device bytes of the VL DFly worker's buffers beside the target KV pool.

    ``num_request_rows`` counts the request pool's rows, padding row included.
    Request state is held per row; the top-k capture and the verify buffers
    hold one block per request, which bounds every decode graph bucket.
    """
    producers = len(dfly_producer_layers(spec.target_index_layers))
    max_requests = num_request_rows - 1
    verify_rows = max_requests * spec.query_width
    int32, int64 = 4, 8
    hidden_itemsize = torch.empty((), dtype=hidden_dtype).element_size()
    request_state = (
        num_request_rows
        * (hidden_size * hidden_itemsize + producers * spec.target_index_topk * int32)
        + num_request_rows * 2 * int64
    )
    topk_capture = (
        producers * verify_rows * spec.target_index_topk * int32
        + (producers + 1) * int64
    )
    verify_buffers = (
        max_requests
        + verify_rows * int64
        + producers * verify_rows * spec.draft_index_topk * int32
        + (2 * verify_rows + 1) * int32
    )
    planning_buffers = (
        7 * verify_rows * int64
        + max_requests * (int64 + 1 + hidden_size * hidden_itemsize)
        + max_requests * spec.context_local_window * int64
    )
    return request_state + topk_capture + verify_buffers + planning_buffers


def dfly_mrope_positions(
    *, positions: torch.Tensor, active: torch.Tensor, deltas: torch.Tensor
) -> torch.Tensor:
    """``[3, bs * query_width]`` text-aligned mRoPE positions of a draft block.

    Draft queries follow the request's last text position, so every section
    takes the position plus the request's mRoPE delta, as a verify does.
    """
    shifted = torch.where(
        active[:, None], positions + deltas.to(positions.dtype)[:, None], positions
    )
    return shifted.flatten().unsqueeze(0).repeat(3, 1)


def check_dfly_attention_backends(
    *,
    draft_decode_impl: str,
    draft_fused_topk: bool,
    target_fused_topk: bool,
    transform_methods: dict[str, TopkTransformMethod],
    index_kpool: int,
    has_hisparse: bool,
    translates_ids: bool,
) -> bool:
    """Fail at startup unless the draft can consume the target's top-k as is.

    Returns whether sparse ids are physical KV slots (fused top-k transform)
    rather than request-local positions.
    """
    unsupported = []
    if draft_decode_impl != "flashmla_sparse":
        unsupported.append(
            f"--dsa-decode-backend {draft_decode_impl!r} (use flashmla_sparse)"
        )
    if draft_fused_topk != target_fused_topk:
        unsupported.append(
            "the target and draft disagree on fused DSA top-k "
            f"({target_fused_topk} vs {draft_fused_topk})"
        )
    for role, method in transform_methods.items():
        if method != TopkTransformMethod.PAGED:
            unsupported.append(f"{role} top-k is {method.name}, not PAGED")
    if index_kpool > 1:
        unsupported.append(f"pooled DSA index (kpool {index_kpool})")
    if has_hisparse:
        unsupported.append("HiSparse")
    if translates_ids:
        unsupported.append("unified memory (translated KV ids)")
    if unsupported:
        raise ValueError(
            "DFLY reads the target's index top-k and KV directly. Unsupported: "
            + "; ".join(unsupported)
            + "."
        )
    return draft_fused_topk


@triton.jit
def _build_draft_topk_kernel(
    state_topk,
    req_pool_indices,
    verify_slots,
    out_rows,
    out_counts,
    state_layer_stride,
    state_req_stride,
    out_layer_stride,
    out_row_stride,
    WIDTH: tl.constexpr,
    TARGET_TOPK: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
):
    req_row = tl.program_id(0)
    layer = tl.program_id(1)
    req = tl.load(req_pool_indices + req_row).to(tl.int64)
    cols = tl.arange(0, TARGET_TOPK)
    history = tl.load(
        state_topk + layer * state_layer_stride + req * state_req_stride + cols
    )
    valid = history >= 0
    packed = tl.cumsum(valid.to(tl.int32), axis=0) - 1
    num_valid = tl.sum(valid.to(tl.int32), axis=0)
    block = tl.arange(0, BLOCK_WIDTH)
    slots = tl.load(
        verify_slots + req_row * WIDTH + block, mask=block < WIDTH, other=-1
    ).to(tl.int32)
    for j in tl.static_range(WIDTH):
        row = req_row * WIDTH + j
        base = out_rows + layer * out_layer_stride + row * out_row_stride
        tl.store(base + packed, history, mask=valid)
        tl.store(base + num_valid + block, slots, mask=block <= j)
        if layer == 0:
            tl.store(out_counts + row, num_valid + j + 1)


def build_draft_topk(
    *,
    state_topk: torch.Tensor,
    req_pool_indices: torch.Tensor,
    verify_slots: torch.Tensor,
    out_rows: torch.Tensor,
    out_counts: torch.Tensor,
) -> None:
    bs, width = verify_slots.shape
    layers, _, target_topk = state_topk.shape
    rows = out_rows[:, : bs * width]
    rows.fill_(-1)
    _build_draft_topk_kernel[(bs, layers)](
        state_topk,
        req_pool_indices,
        verify_slots,
        out_rows,
        out_counts,
        state_topk.stride(0),
        state_topk.stride(1),
        out_rows.stride(0),
        out_rows.stride(1),
        WIDTH=width,
        TARGET_TOPK=target_topk,
        BLOCK_WIDTH=triton.next_power_of_2(width),
    )


class DFlyRequestState:
    def __init__(
        self,
        *,
        pool_size: int,
        hidden_size: int,
        num_layers: int,
        target_topk: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.hidden = torch.zeros((pool_size, hidden_size), dtype=dtype, device=device)
        self.topk = torch.full(
            (num_layers, pool_size, target_topk), -1, dtype=torch.int32, device=device
        )

    def write(
        self,
        *,
        req_pool_indices: torch.Tensor,
        hidden: torch.Tensor,
        topk: torch.Tensor,
    ) -> None:
        self.hidden[req_pool_indices] = hidden
        self.topk[:, req_pool_indices] = topk
