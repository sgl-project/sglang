# SPDX-License-Identifier: Apache-2.0
"""Replay visibility and masked attention execution for Cosmos3-Nano-Sim-Auto.

Every current-chunk token attends one softmax over the keys the replay policy
admits from three streams: the current chunk, the KV ring of committed chunks
and the per-camera caption documents. The policy is evaluated once per chunk on
metadata runs (contiguous tokens agreeing on every predicate field), giving a
small run-level truth table; the kernel (FlexAttention Triton anywhere, or
FlashAttention-4 block-sparse on Hopper and Blackwell) sees it as a block map
whose partial tiles consult the table through per-token run ids. A dense
token-level expansion serves tests.

Semantics follow the imaginaire4 maskless replay plan the student trained with:
the role and causality rules hold for every pair, same-view pairs are always in
scope, and cross-view pairs are admitted only between non-control tokens
captured at most ``window_seconds`` apart (key no later than the query).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import msgspec
import torch
from torch.nn.attention.flex_attention import BlockMask

from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_attention import (
    TRITON_KV_BLOCK_SIZE,
    TRITON_Q_BLOCK_SIZE,
    _block_indices,
    fa4_sparse_block_sizes,
    flex_attention,
    pack_allowed_bits,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.multiview_fa4 import (
    Fa4BlockSparsity,
    fa4_block_sparse_attention,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3_simulation.sim_auto_layout import (
    CAPTION_SCOPE_ALL,
    CAPTION_SCOPE_SAME_VIEW,
    ROLE_CLEAN_TARGET,
    ROLE_CONTROL,
    ROLE_CURRENT_TARGET,
    ROLE_TARGET_CONDITION,
    ROLE_UND,
    CaptionMetadata,
    TokenMetadata,
)

# Reference tolerances: the window and caption bounds compare float32 seconds.
WINDOW_EPS = 1e-4
CAPTION_EPS = 1e-4

REPLAY_BLOCK_SIZE = (TRITON_Q_BLOCK_SIZE, TRITON_KV_BLOCK_SIZE)
# Kernels behind the replay mask; both count every admitted key exactly once.
REPLAY_BACKENDS = ("triton", "fa4")

_RUN_FIELDS = (
    "start",
    "length",
    "view_id",
    "frame_id",
    "timestamp",
    "role",
    "step",
    "caption_scope",
    "caption_start",
    "caption_end",
)


class RunTable(msgspec.Struct, eq=False):
    """Contiguous token runs of one stream that agree on every predicate field.

    ``start``/``length`` address tokens of the stream; caption runs are whole
    documents and carry time bounds, sensor runs carry ``-inf``/``inf``.
    """

    start: torch.Tensor
    length: torch.Tensor
    view_id: torch.Tensor
    frame_id: torch.Tensor
    timestamp: torch.Tensor
    role: torch.Tensor
    step: torch.Tensor
    caption_scope: torch.Tensor
    caption_start: torch.Tensor
    caption_end: torch.Tensor

    @property
    def num_runs(self) -> int:
        return int(self.start.numel())

    @property
    def num_tokens(self) -> int:
        return int(self.length.sum()) if self.num_runs else 0

    @property
    def is_und(self) -> torch.Tensor:
        return self.role == ROLE_UND


def _run_boundaries(*fields: torch.Tensor) -> torch.Tensor:
    """First-token indexes of maximal runs with identical field values."""
    length = fields[0].numel()
    if length == 0:
        return torch.empty(0, dtype=torch.int64)
    change = torch.zeros(length, dtype=torch.bool)
    change[0] = True
    for field in fields:
        change[1:] |= field[1:] != field[:-1]
    return torch.nonzero(change).flatten()


def empty_runs() -> RunTable:
    long = torch.empty(0, dtype=torch.int64)
    single = torch.empty(0, dtype=torch.float32)
    return RunTable(long, long, long, long, single, long, long, long, single, single)


def concat_runs(tables: Sequence[RunTable]) -> RunTable:
    if not tables:
        return empty_runs()
    return RunTable(
        **{name: torch.cat([getattr(t, name) for t in tables]) for name in _RUN_FIELDS}
    )


def sensor_runs(metadata: TokenMetadata, *, base: int = 0) -> RunTable:
    """Group a sensor stream by (view, frame, role, step); timestamps follow from the frame."""
    starts = _run_boundaries(
        metadata.view_id,
        metadata.frame_id,
        metadata.role,
        metadata.step,
        metadata.caption_scope,
    )
    if starts.numel() == 0:
        return empty_runs()
    ends = torch.cat(
        [starts[1:], torch.tensor([metadata.num_tokens], dtype=torch.int64)]
    )
    inf = torch.full((starts.numel(),), math.inf, dtype=torch.float32)
    return RunTable(
        start=starts + base,
        length=ends - starts,
        view_id=metadata.view_id[starts],
        frame_id=metadata.frame_id[starts],
        timestamp=metadata.timestamp[starts],
        role=metadata.role[starts],
        step=metadata.step[starts],
        caption_scope=metadata.caption_scope[starts],
        caption_start=-inf,
        caption_end=inf,
    )


def history_runs(slots: Sequence[tuple[int, TokenMetadata]]) -> RunTable:
    """Runs of the populated ring slots, each at its ``(base, metadata)`` position in the ring stream."""
    return concat_runs(
        [
            sensor_runs(metadata, base=base)
            for base, metadata in slots
            if metadata.num_tokens
        ]
    )


def caption_runs(captions: CaptionMetadata, *, base: int = 0) -> RunTable:
    """One run per caption document."""
    starts = torch.tensor(captions.doc_offsets, dtype=torch.int64)
    ends = torch.cat(
        [starts[1:], torch.tensor([captions.num_tokens], dtype=torch.int64)]
    )
    count = starts.numel()
    return RunTable(
        start=starts + base,
        length=ends - starts,
        view_id=captions.view_id[starts],
        frame_id=torch.full((count,), -1, dtype=torch.int64),
        timestamp=torch.full((count,), -1.0, dtype=torch.float32),
        role=torch.full((count,), ROLE_UND, dtype=torch.int64),
        step=torch.full((count,), -1, dtype=torch.int64),
        caption_scope=torch.zeros(count, dtype=torch.int64),
        caption_start=captions.start_seconds[starts],
        caption_end=captions.end_seconds[starts],
    )


def pair_allowed(q: RunTable, k: RunTable, *, window_seconds: float) -> torch.Tensor:
    """Replay visibility ``[Q, K]`` between current-chunk query runs and key runs of one stream.

    Control queries and control keys never take part in cross-view attention:
    controls read same-view clean history only, and targets read controls of
    their own view only. Condition K/V is reused across chunks, so a condition
    query must not encode noisy targets of its own chunk.
    """
    if not (math.isfinite(window_seconds) and window_seconds >= 0):
        raise ValueError(
            f"window_seconds must be finite and non-negative, got {window_seconds}."
        )
    if q.num_runs == 0 or k.num_runs == 0:
        return torch.zeros(q.num_runs, k.num_runs, dtype=torch.bool)
    same_view = q.view_id[:, None] == k.view_id[None, :]
    same_frame = q.frame_id[:, None] == k.frame_id[None, :]
    gap = q.timestamp[:, None] - k.timestamp[None, :]
    within_window = (gap >= -WINDOW_EPS) & (gap <= window_seconds + WINDOW_EPS)
    in_scope = same_view | within_window
    q_step, k_step = q.step[:, None], k.step[None, :]
    q_role, k_role = q.role[:, None], k.role[None, :]
    q_is_target = (q_role == ROLE_CURRENT_TARGET) | (q_role == ROLE_CLEAN_TARGET)
    q_is_current = q_role == ROLE_CURRENT_TARGET
    q_is_control = q_role == ROLE_CONTROL
    q_is_condition = q_role == ROLE_TARGET_CONDITION
    k_is_und = k_role == ROLE_UND
    k_is_control = k_role == ROLE_CONTROL
    k_is_condition = k_role == ROLE_TARGET_CONDITION
    k_is_current = k_role == ROLE_CURRENT_TARGET
    k_is_clean = k_role == ROLE_CLEAN_TARGET
    k_is_clean_sensor = k_is_condition | k_is_clean
    control_step_allowed = k_step <= q_step
    q_scope = q.caption_scope[:, None]
    q_time = q.timestamp[:, None]
    caption_reaches_query = (
        k_is_und
        & (
            (q_scope == CAPTION_SCOPE_ALL)
            | ((q_scope == CAPTION_SCOPE_SAME_VIEW) & same_view)
        )
        & (q_time >= k.caption_start[None, :] - CAPTION_EPS)
        & (q_time < k.caption_end[None, :] - CAPTION_EPS)
    )
    clean_step_allowed = (q_is_current & (k_step < q_step)) | (
        ~q_is_current & (k_step <= q_step)
    )
    return (
        ((q_is_target | q_is_control | q_is_condition) & caption_reaches_query)
        | (q_is_control & k_is_control & same_view & control_step_allowed)
        | (q_is_control & k_is_clean_sensor & (k_step < q_step) & same_view)
        | (q_is_condition & k_is_condition & same_frame & same_view)
        | (q_is_condition & k_is_control & same_view & control_step_allowed)
        | (q_is_condition & k_is_current & (k_step < q_step) & in_scope)
        | (q_is_current & k_is_current & (k_step == q_step) & in_scope)
        | (q_is_target & k_is_clean & clean_step_allowed & in_scope)
        | (q_is_target & k_is_control & same_view & control_step_allowed)
        | (q_is_target & k_is_condition & in_scope & (k_step <= q_step))
    )


def _token_run_ids(table: RunTable, num_tokens: int) -> torch.Tensor:
    """Run index of every token of a stream (``-1`` where no run covers the token)."""
    ids = torch.full((num_tokens,), -1, dtype=torch.int64)
    if table.num_runs:
        runs = torch.arange(table.num_runs, dtype=torch.int64)
        positions = torch.repeat_interleave(
            table.start - (torch.cumsum(table.length, 0) - table.length), table.length
        )
        positions = positions + torch.arange(int(table.length.sum()), dtype=torch.int64)
        ids[positions] = torch.repeat_interleave(runs, table.length)
    return ids


class ReplayPlan(msgspec.Struct, eq=False):
    """Run-level truth table of one forward plus the per-token run ids the kernel needs.

    Keys are addressed in one stream ``[current | ring | captions]``; the ring
    part spans the whole ring capacity, and tokens of empty or padded slots carry
    run id ``-1`` so no query ever reads them.
    """

    query_runs: RunTable
    key_runs: RunTable
    allowed: torch.Tensor
    num_query_tokens: int
    num_current_tokens: int
    num_history_tokens: int
    num_und_tokens: int
    query_run_ids: torch.Tensor
    key_run_ids: torch.Tensor

    @property
    def num_key_tokens(self) -> int:
        return self.num_current_tokens + self.num_history_tokens + self.num_und_tokens


def build_replay_plan(
    *,
    current: TokenMetadata,
    history: Sequence[tuple[int, TokenMetadata]],
    history_tokens: int,
    captions: CaptionMetadata,
    window_seconds: float,
) -> ReplayPlan:
    """Evaluate the replay predicate on runs of the current chunk against all three key streams."""
    if current.num_tokens == 0:
        raise ValueError("The current chunk has no tokens.")
    for base, metadata in history:
        if base < 0 or base + metadata.num_tokens > history_tokens:
            raise ValueError("A ring slot lies outside the history stream.")
    query_runs = sensor_runs(current)
    tables = (
        query_runs,
        history_runs(
            [(current.num_tokens + base, metadata) for base, metadata in history]
        ),
        caption_runs(captions, base=current.num_tokens + history_tokens),
    )
    allowed = torch.cat(
        [
            pair_allowed(query_runs, table, window_seconds=window_seconds)
            for table in tables
        ],
        1,
    )
    key_runs = concat_runs(tables)
    num_keys = current.num_tokens + history_tokens + captions.num_tokens
    query_run_ids = _token_run_ids(query_runs, current.num_tokens)
    if bool((query_run_ids < 0).any()):
        raise RuntimeError("Every current token must belong to a query run.")
    return ReplayPlan(
        query_runs=query_runs,
        key_runs=key_runs,
        allowed=allowed,
        num_query_tokens=current.num_tokens,
        num_current_tokens=current.num_tokens,
        num_history_tokens=history_tokens,
        num_und_tokens=captions.num_tokens,
        query_run_ids=query_run_ids,
        key_run_ids=_token_run_ids(key_runs, num_keys),
    )


def dense_token_mask(plan: ReplayPlan) -> torch.Tensor:
    """Expand the run-level truth table to tokens: ``[queries, current + ring + captions]`` (tests)."""
    rows = plan.query_run_ids
    columns = plan.key_run_ids
    mask = torch.zeros(rows.numel(), columns.numel(), dtype=torch.bool)
    real = columns >= 0
    mask[:, real] = plan.allowed[rows][:, columns[real]]
    return mask


class ReplayBlockMask(msgspec.Struct, eq=False):
    """FlexAttention block mask over the padded query and key streams of one plan."""

    block_mask: BlockMask
    q_len: int
    kv_len: int
    padded_q_len: int
    padded_kv_len: int


class ReplayFa4Mask(msgspec.Struct, eq=False):
    """FlashAttention-4 block map and packed run table over the padded streams of one plan."""

    sparsity: Fa4BlockSparsity
    q_len: int
    kv_len: int
    padded_q_len: int
    padded_kv_len: int


ReplayMask = ReplayBlockMask | ReplayFa4Mask


class _PaddedStreams(msgspec.Struct, eq=False):
    """Run ids of the zero-padded query and key streams and the padded truth table.

    Padding queries form an extra run that attends everything, which keeps
    their rows finite (the outputs are discarded); padding keys form an extra
    run no query may read.
    """

    q_ids: torch.Tensor
    k_ids: torch.Tensor
    allowed: torch.Tensor
    padded_q_len: int
    padded_kv_len: int


class _BlockMap(msgspec.Struct, eq=False):
    """FlexAttention KV-block layout: per query block, its partial and full key blocks."""

    partial_counts: torch.Tensor
    partial_indices: torch.Tensor
    full_counts: torch.Tensor
    full_indices: torch.Tensor


def _round_up(value: int, multiple: int) -> int:
    return -(-value // multiple) * multiple


def _presence(ids: torch.Tensor, block: int, count: int) -> torch.Tensor:
    """``[blocks, count]`` run presence of a stream whose padded length is a multiple of ``block``."""
    blocks = ids.view(-1, block)
    presence = torch.zeros((blocks.shape[0], count), dtype=torch.bool)
    presence.scatter_(1, blocks, True)
    return presence


def _padded_streams(plan: ReplayPlan, block_size: tuple[int, int]) -> _PaddedStreams:
    q_block, kv_block = block_size
    num_q_runs, num_k_runs = plan.allowed.shape
    padded_q = _round_up(plan.num_query_tokens, q_block)
    padded_kv = _round_up(plan.num_key_tokens, kv_block)
    q_ids = torch.full((padded_q,), num_q_runs, dtype=torch.int64)
    q_ids[: plan.num_query_tokens] = plan.query_run_ids
    k_ids = torch.full((padded_kv,), num_k_runs, dtype=torch.int64)
    k_ids[: plan.num_key_tokens] = torch.where(
        plan.key_run_ids >= 0, plan.key_run_ids, num_k_runs
    )
    allowed = torch.zeros((num_q_runs + 1, num_k_runs + 1), dtype=torch.bool)
    allowed[:num_q_runs, :num_k_runs] = plan.allowed
    allowed[num_q_runs, :] = True
    return _PaddedStreams(
        q_ids=q_ids,
        k_ids=k_ids,
        allowed=allowed,
        padded_q_len=padded_q,
        padded_kv_len=padded_kv,
    )


def _project_blocks(streams: _PaddedStreams, block_size: tuple[int, int]) -> _BlockMap:
    """Classify every tile as skipped, full or partial from the runs present in it."""
    q_block, kv_block = block_size
    num_q_runs, num_k_runs = streams.allowed.shape
    q_presence = _presence(streams.q_ids, q_block, num_q_runs).to(torch.float32)
    k_presence = _presence(streams.k_ids, kv_block, num_k_runs).to(torch.float32)
    allowed = streams.allowed.to(torch.float32)
    visible = (q_presence @ allowed @ k_presence.T) > 0
    forbidden = (q_presence @ (1.0 - allowed) @ k_presence.T) > 0
    full_blocks = visible & ~forbidden
    partial_counts, partial_indices = _block_indices(visible & ~full_blocks)
    full_counts, full_indices = _block_indices(full_blocks)
    return _BlockMap(
        partial_counts=partial_counts,
        partial_indices=partial_indices,
        full_counts=full_counts,
        full_indices=full_indices,
    )


def build_replay_block_mask(
    plan: ReplayPlan,
    *,
    device: torch.device,
    block_size: tuple[int, int] = REPLAY_BLOCK_SIZE,
) -> ReplayBlockMask:
    """Triton FlexAttention mask; partial tiles consult the run table per token."""
    streams = _padded_streams(plan, block_size)
    blocks = _project_blocks(streams, block_size)
    allowed_dev = streams.allowed.to(device)
    q_ids_dev = streams.q_ids.to(device)
    k_ids_dev = streams.k_ids.to(device)

    def mask_mod(
        batch: torch.Tensor,
        head: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        del batch, head
        return allowed_dev[q_ids_dev[q_idx], k_ids_dev[kv_idx]]

    block_mask = BlockMask.from_kv_blocks(
        blocks.partial_counts[None, None].to(device),
        blocks.partial_indices[None, None].to(device),
        blocks.full_counts[None, None].to(device),
        blocks.full_indices[None, None].to(device),
        BLOCK_SIZE=block_size,
        mask_mod=mask_mod,
        seq_lengths=(streams.padded_q_len, streams.padded_kv_len),
        compute_q_blocks=False,
    )
    return ReplayBlockMask(
        block_mask=block_mask,
        q_len=plan.num_query_tokens,
        kv_len=plan.num_key_tokens,
        padded_q_len=streams.padded_q_len,
        padded_kv_len=streams.padded_kv_len,
    )


def build_replay_fa4_mask(
    plan: ReplayPlan,
    *,
    device: torch.device,
    block_size: tuple[int, int] | None = None,
) -> ReplayFa4Mask:
    """FlashAttention-4 block map at the device's tile geometry plus the packed run table.

    ``block_size`` pins the geometry for CPU tests; on CUDA it follows the
    compute capability (Hopper 128x128, Blackwell 256x128).
    """
    if block_size is None:
        block_size = fa4_sparse_block_sizes(device)
    streams = _padded_streams(plan, block_size)
    blocks = _project_blocks(streams, block_size)
    allowed_words, words_per_row = pack_allowed_bits(streams.allowed)
    sparsity = Fa4BlockSparsity(
        partial_counts=blocks.partial_counts.to(device),
        partial_indices=blocks.partial_indices.to(device),
        full_counts=blocks.full_counts.to(device),
        full_indices=blocks.full_indices.to(device),
        q_word_base=(streams.q_ids * words_per_row).to(torch.int32).to(device),
        k_group_ids=streams.k_ids.to(torch.int32).to(device),
        allowed_words=allowed_words.to(device),
        q_block_size=block_size[0],
        kv_block_size=block_size[1],
    )
    return ReplayFa4Mask(
        sparsity=sparsity,
        q_len=plan.num_query_tokens,
        kv_len=plan.num_key_tokens,
        padded_q_len=streams.padded_q_len,
        padded_kv_len=streams.padded_kv_len,
    )


def build_replay_mask(
    plan: ReplayPlan, *, device: torch.device, backend: str
) -> ReplayMask:
    """The mask representation ``backend`` consumes; see ``REPLAY_BACKENDS``."""
    if backend == "triton":
        return build_replay_block_mask(plan, device=device)
    if backend == "fa4":
        return build_replay_fa4_mask(plan, device=device)
    raise ValueError(
        "Cosmos3-Nano-Sim-Auto attention backend must be one of "
        f"{list(REPLAY_BACKENDS)}, got {backend!r}."
    )


def _pad_bhsd(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Zero-pad ``[B, S, H, D]`` along ``S`` and return ``[B, H, length, D]`` contiguous."""
    pad = length - tensor.shape[1]
    if pad:
        tensor = torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad))
    return tensor.transpose(1, 2).contiguous()


def _pad_bshd(tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Zero-pad ``[B, S, H, D]`` along ``S``, contiguous."""
    pad = length - tensor.shape[1]
    if pad:
        return torch.nn.functional.pad(tensor, (0, 0, 0, 0, 0, pad))
    return tensor.contiguous()


def replay_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: ReplayMask
) -> torch.Tensor:
    """Masked GQA attention of ``[B, S_q, H, D]`` queries over ``[B, S_k, H_kv, D]`` keys."""
    if q.shape[1] != mask.q_len or k.shape[1] != mask.kv_len or k.shape != v.shape:
        raise ValueError(
            "Cosmos3-Nano-Sim-Auto attention geometry does not match its plan: "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, plan=({mask.q_len}, {mask.kv_len})."
        )
    if isinstance(mask, ReplayFa4Mask):
        out = fa4_block_sparse_attention(
            _pad_bshd(q, mask.padded_q_len),
            _pad_bshd(k, mask.padded_kv_len),
            _pad_bshd(v, mask.padded_kv_len),
            sparsity=mask.sparsity,
        )
        return out[:, : mask.q_len]
    out = flex_attention(
        _pad_bhsd(q, mask.padded_q_len),
        _pad_bhsd(k, mask.padded_kv_len),
        _pad_bhsd(v, mask.padded_kv_len),
        block_mask=mask.block_mask,
        backend="triton",
    )
    return out[:, :, : mask.q_len].transpose(1, 2)
