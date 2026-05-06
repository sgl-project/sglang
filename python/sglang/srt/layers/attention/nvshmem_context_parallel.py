"""Experimental context-parallel attention helpers for NVSHMEM peer K/V.

This module deliberately separates CP semantics from the transport/kernel
implementation.  The reference path uses regular torch ops so tests can lock
down chunk ownership, load-balanced ordering, LSE merging, and autograd
ownership before a Triton/NVSHMEM production kernel replaces the inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Iterable, Literal, Optional, Sequence

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional CUDA/Triton path
    triton = None
    tl = None

LoadBalanceMode = Literal["contiguous", "headtail"]


@dataclass(frozen=True)
class SequenceChunk:
    """A global sequence interval owned by one CP rank.

    ``local_start``/``local_end`` describe where the global interval lives inside
    the owner's local symmetric K/V tensor.  This matters for head-tail sharding,
    where one rank owns two non-contiguous global intervals concatenated locally.
    """

    global_start: int
    global_end: int
    owner_rank: int
    local_start: int
    local_end: int

    @property
    def length(self) -> int:
        return self.global_end - self.global_start


@dataclass(frozen=True)
class KVChunk:
    """K/V tensors plus the CP ownership metadata for their sequence interval."""

    meta: SequenceChunk
    k: torch.Tensor
    v: torch.Tensor


def _partition_evenly(length: int, parts: int) -> list[tuple[int, int]]:
    if parts <= 0:
        raise ValueError("parts must be positive")
    base = length // parts
    extra = length % parts
    out = []
    start = 0
    for idx in range(parts):
        size = base + (1 if idx < extra else 0)
        end = start + size
        out.append((start, end))
        start = end
    return out


def build_sequence_chunks(
    seq_len: int,
    world_size: int,
    mode: LoadBalanceMode = "contiguous",
) -> list[SequenceChunk]:
    """Build CP rank ownership for contiguous or head-tail load balancing."""

    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if world_size <= 0:
        raise ValueError("world_size must be positive")

    if mode == "contiguous":
        global_parts = _partition_evenly(seq_len, world_size)
        per_rank: list[list[tuple[int, int]]] = [[part] for part in global_parts]
    elif mode == "headtail":
        fine_parts = _partition_evenly(seq_len, 2 * world_size)
        per_rank = []
        for rank in range(world_size):
            pair = [fine_parts[rank], fine_parts[2 * world_size - 1 - rank]]
            per_rank.append(sorted(pair))
    else:
        raise ValueError(f"unsupported load-balance mode: {mode}")

    chunks: list[SequenceChunk] = []
    for owner_rank, parts in enumerate(per_rank):
        local_offset = 0
        for global_start, global_end in parts:
            length = global_end - global_start
            chunks.append(
                SequenceChunk(
                    global_start=global_start,
                    global_end=global_end,
                    owner_rank=owner_rank,
                    local_start=local_offset,
                    local_end=local_offset + length,
                )
            )
            local_offset += length

    return sorted(chunks, key=lambda c: (c.global_start, c.global_end, c.owner_rank))


def chunks_for_rank(chunks: Sequence[SequenceChunk], rank: int) -> list[SequenceChunk]:
    return [chunk for chunk in chunks if chunk.owner_rank == rank]


def query_positions_for_chunks(
    chunks: Sequence[SequenceChunk], *, device: torch.device | str | None = None
) -> torch.Tensor:
    positions = [
        torch.arange(chunk.global_start, chunk.global_end, device=device)
        for chunk in chunks
        if chunk.length > 0
    ]
    if not positions:
        return torch.empty((0,), dtype=torch.long, device=device)
    return torch.cat(positions, dim=0).to(torch.long)


def shard_sequence_tensor(
    tensor: torch.Tensor,
    chunks: Sequence[SequenceChunk],
    *,
    seq_dim: int,
) -> torch.Tensor:
    """Concatenate global sequence slices in a rank's local chunk order."""

    slices = []
    for chunk in chunks:
        index = [slice(None)] * tensor.ndim
        index[seq_dim] = slice(chunk.global_start, chunk.global_end)
        slices.append(tensor[tuple(index)])
    if not slices:
        shape = list(tensor.shape)
        shape[seq_dim] = 0
        return tensor.new_empty(shape)
    return torch.cat(slices, dim=seq_dim)


def local_kv_chunks(
    local_k_by_rank: Sequence[torch.Tensor],
    local_v_by_rank: Sequence[torch.Tensor],
    chunks: Sequence[SequenceChunk],
) -> list[KVChunk]:
    """Build KVChunk objects from per-rank local tensors.

    Per-rank tensors are expected to use shape ``[B, H, T_local, D]``.
    """

    if len(local_k_by_rank) != len(local_v_by_rank):
        raise ValueError("local_k_by_rank and local_v_by_rank length mismatch")

    kv_chunks: list[KVChunk] = []
    for chunk in chunks:
        k_owner = local_k_by_rank[chunk.owner_rank]
        v_owner = local_v_by_rank[chunk.owner_rank]
        kv_chunks.append(
            KVChunk(
                meta=chunk,
                k=k_owner[:, :, chunk.local_start : chunk.local_end, :],
                v=v_owner[:, :, chunk.local_start : chunk.local_end, :],
            )
        )
    return kv_chunks


def nvshmem_peer_kv_chunks(
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    chunks: Sequence[SequenceChunk],
    *,
    rank: int,
    get_peer_tensor: Optional[Callable[[torch.Tensor, int], torch.Tensor]] = None,
) -> list[KVChunk]:
    """Build KVChunk objects backed by NVSHMEM peer tensor views.

    ``local_k`` and ``local_v`` are this process' symmetric tensors with shape
    ``[B, H, T_local, D]``.  For remote owners, ``get_peer_tensor`` is called
    lazily so the returned tensors can be passed to a direct-load kernel.
    """

    if get_peer_tensor is None:
        from nvshmem.core.interop.torch import get_peer_tensor as _get_peer_tensor

        get_peer_tensor = _get_peer_tensor

    peer_cache: dict[tuple[int, str], torch.Tensor] = {}

    def peer_view(tensor: torch.Tensor, owner: int, name: str) -> torch.Tensor:
        if owner == rank:
            return tensor
        key = (owner, name)
        if key not in peer_cache:
            peer_cache[key] = get_peer_tensor(tensor, owner)
        return peer_cache[key]

    kv_chunks: list[KVChunk] = []
    for chunk in chunks:
        k_peer = peer_view(local_k, chunk.owner_rank, "k")
        v_peer = peer_view(local_v, chunk.owner_rank, "v")
        kv_chunks.append(
            KVChunk(
                meta=chunk,
                k=k_peer[:, :, chunk.local_start : chunk.local_end, :],
                v=v_peer[:, :, chunk.local_start : chunk.local_end, :],
            )
        )
    return kv_chunks


def _partial_attention_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    key_start: int,
    *,
    causal: bool,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float()) * sm_scale

    if causal:
        key_positions = torch.arange(
            key_start, key_start + k.shape[2], device=q.device, dtype=torch.long
        )
        causal_mask = key_positions.view(1, 1, 1, -1) <= query_positions.view(
            1, 1, -1, 1
        )
        scores = scores.masked_fill(~causal_mask, float("-inf"))

    lse = torch.logsumexp(scores, dim=-1)
    safe_scores = torch.where(
        torch.isfinite(lse).unsqueeze(-1),
        scores,
        torch.zeros_like(scores),
    )
    probs = torch.softmax(safe_scores, dim=-1)
    probs = torch.where(torch.isfinite(lse).unsqueeze(-1), probs, 0.0)
    out = torch.einsum("bhqk,bhkd->bhqd", probs, v.float())
    return out, lse


if triton is not None:

    @triton.jit
    def _partial_attention_state_kernel(
        Q,
        K,
        V,
        QueryPositions,
        Out,
        Lse,
        sm_scale: tl.constexpr,
        key_start: tl.constexpr,
        B: tl.constexpr,
        H: tl.constexpr,
        Q_LEN: tl.constexpr,
        K_LEN: tl.constexpr,
        D: tl.constexpr,
        stride_qb: tl.constexpr,
        stride_qh: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kh: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vh: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oh: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lh: tl.constexpr,
        stride_lq: tl.constexpr,
        BLOCK_Q: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        CAUSAL: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
        offs_d = tl.arange(0, BLOCK_D)
        q_mask = (offs_q < Q_LEN)[:, None] & (offs_d < D)[None, :]
        q = tl.load(
            Q
            + pid_b * stride_qb
            + pid_h * stride_qh
            + offs_q[:, None] * stride_qq
            + offs_d[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )

        m_i = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_Q,), tl.float32)
        acc = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)
        query_pos = tl.load(QueryPositions + offs_q, mask=offs_q < Q_LEN, other=-1)

        for start_k in range(0, K_LEN, BLOCK_K):
            offs_k = start_k + tl.arange(0, BLOCK_K)
            k = tl.load(
                K
                + pid_b * stride_kb
                + pid_h * stride_kh
                + offs_k[:, None] * stride_kk
                + offs_d[None, :] * stride_kd,
                mask=(offs_k < K_LEN)[:, None] & (offs_d < D)[None, :],
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * sm_scale
            score_mask = (offs_q < Q_LEN)[:, None] & (offs_k < K_LEN)[None, :]
            if CAUSAL:
                key_pos = key_start + offs_k
                score_mask = score_mask & (key_pos[None, :] <= query_pos[:, None])
            scores = tl.where(score_mask, scores, -float("inf"))

            m_ij = tl.max(scores, axis=1)
            safe_m_ij = tl.where(m_ij == -float("inf"), 0.0, m_ij)
            p = tl.exp(scores - safe_m_ij[:, None])
            p = tl.where(score_mask, p, 0.0)
            l_ij = tl.sum(p, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            beta = tl.exp(m_ij - m_new)
            l_new = l_i * alpha + l_ij * beta
            safe_l_new = tl.where(l_new > 0.0, l_new, 1.0)

            v = tl.load(
                V
                + pid_b * stride_vb
                + pid_h * stride_vh
                + offs_k[:, None] * stride_vk
                + offs_d[None, :] * stride_vd,
                mask=(offs_k < K_LEN)[:, None] & (offs_d < D)[None, :],
                other=0.0,
            )
            acc = acc * ((l_i * alpha) / safe_l_new)[:, None]
            acc += tl.dot(
                (p * (beta / safe_l_new)[:, None]).to(v.dtype),
                v,
                input_precision="ieee",
            )
            l_i = l_new
            m_i = m_new

        finite = l_i > 0.0
        acc = tl.where(finite[:, None], acc, 0.0)
        tl.store(
            Out
            + pid_b * stride_ob
            + pid_h * stride_oh
            + offs_q[:, None] * stride_oq
            + offs_d[None, :] * stride_od,
            acc,
            mask=q_mask,
        )
        lse = tl.where(finite, tl.log(l_i) + m_i, -float("inf"))
        tl.store(
            Lse + pid_b * stride_lb + pid_h * stride_lh + offs_q * stride_lq,
            lse,
            mask=offs_q < Q_LEN,
        )


def _partial_attention_state_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_positions: torch.Tensor,
    key_start: int,
    *,
    causal: bool,
    sm_scale: float,
    block_q: int,
    block_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or not q.is_cuda:
        return _partial_attention_state(
            q,
            k,
            v,
            query_positions,
            key_start,
            causal=causal,
            sm_scale=sm_scale,
        )

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, H, T, D]")
    if q.shape[:2] != k.shape[:2] or k.shape != v.shape:
        raise ValueError("q/k/v batch, head, and k/v shapes must match")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dimensions must match")
    if block_q < 16 or block_k < 16:
        raise ValueError("block_q and block_k must be >= 16 for tl.dot")

    bsz, heads, q_len, head_dim = q.shape
    k_len = k.shape[2]
    block_d = triton.next_power_of_2(head_dim)
    if block_d > 256:
        raise ValueError("Triton CP reference supports head_dim <= 256")

    out = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty((bsz, heads, q_len), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(q_len, block_q), heads, bsz)
    _partial_attention_state_kernel[grid](
        q,
        k,
        v,
        query_positions.to(device=q.device),
        out,
        lse,
        float(sm_scale),
        int(key_start),
        bsz,
        heads,
        q_len,
        k_len,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        BLOCK_Q=block_q,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        CAUSAL=causal,
    )
    return out, lse


def merge_attention_states(
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge two attention states using log-sum-exp normalization."""

    merged_lse = torch.logaddexp(prefix_lse, suffix_lse)
    finite = torch.isfinite(merged_lse)
    prefix_scale = torch.where(
        finite, torch.exp(prefix_lse - merged_lse), torch.zeros_like(merged_lse)
    )
    suffix_scale = torch.where(
        finite, torch.exp(suffix_lse - merged_lse), torch.zeros_like(merged_lse)
    )
    merged = (
        prefix_output * prefix_scale.unsqueeze(-1)
        + suffix_output * suffix_scale.unsqueeze(-1)
    )
    return merged, merged_lse


def context_parallel_attention_reference(
    q: torch.Tensor,
    kv_chunks: Sequence[KVChunk],
    *,
    query_positions: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference CP attention over local and peer K/V chunks.

    Shapes:
    - q: ``[B, H, Q, D]``
    - each K/V chunk: ``[B, H, K_chunk, D]``

    Gradients flow into each chunk tensor.  When chunks are backed by owner-rank
    tensors or NVSHMEM peer views, dK/dV ownership stays attached to the owner
    chunk instead of a locally materialized allgather buffer.
    """

    if q.ndim != 4:
        raise ValueError("q must have shape [B, H, Q, D]")
    if query_positions.shape != (q.shape[2],):
        raise ValueError("query_positions must have shape [Q]")
    if sm_scale is None:
        sm_scale = 1.0 / sqrt(q.shape[-1])

    out: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None
    for kv_chunk in kv_chunks:
        if kv_chunk.meta.length == 0:
            continue
        partial_out, partial_lse = _partial_attention_state(
            q,
            kv_chunk.k,
            kv_chunk.v,
            query_positions.to(device=q.device),
            kv_chunk.meta.global_start,
            causal=causal,
            sm_scale=float(sm_scale),
        )
        if out is None or lse is None:
            out, lse = partial_out, partial_lse
        else:
            out, lse = merge_attention_states(out, lse, partial_out, partial_lse)

    if out is None or lse is None:
        out = torch.zeros_like(q, dtype=torch.float32)
        lse = torch.full(q.shape[:3], float("-inf"), dtype=torch.float32, device=q.device)

    out = out.to(dtype=q.dtype)
    if return_lse:
        return out, lse
    return out


def context_parallel_attention_triton_forward(
    q: torch.Tensor,
    kv_chunks: Sequence[KVChunk],
    *,
    query_positions: torch.Tensor,
    causal: bool = True,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    block_q: int = 16,
    block_k: int = 64,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Triton forward path over local or NVSHMEM peer K/V chunks.

    The wrapper intentionally mirrors ``context_parallel_attention_reference``.
    It computes one partial attention state per chunk, then merges states with
    the same LSE contract.  Backward is not implemented here; production
    training should use an explicit dQ/dK/dV kernel or fall back to the
    reference path until that kernel is available.
    """

    if sm_scale is None:
        sm_scale = 1.0 / sqrt(q.shape[-1])

    out: Optional[torch.Tensor] = None
    lse: Optional[torch.Tensor] = None
    for kv_chunk in kv_chunks:
        if kv_chunk.meta.length == 0:
            continue
        partial_out, partial_lse = _partial_attention_state_triton(
            q,
            kv_chunk.k,
            kv_chunk.v,
            query_positions,
            kv_chunk.meta.global_start,
            causal=causal,
            sm_scale=float(sm_scale),
            block_q=block_q,
            block_k=block_k,
        )
        if out is None or lse is None:
            out, lse = partial_out, partial_lse
        else:
            out, lse = merge_attention_states(out, lse, partial_out, partial_lse)

    if out is None or lse is None:
        out = torch.zeros_like(q, dtype=torch.float32)
        lse = torch.full(q.shape[:3], float("-inf"), dtype=torch.float32, device=q.device)

    out = out.to(dtype=q.dtype)
    if return_lse:
        return out, lse
    return out


def owned_kv_grads(kv_chunks: Iterable[KVChunk]) -> dict[int, list[tuple[SequenceChunk, torch.Tensor | None, torch.Tensor | None]]]:
    """Return dK/dV tensors grouped by owning CP rank after backward."""

    grads: dict[int, list[tuple[SequenceChunk, torch.Tensor | None, torch.Tensor | None]]] = {}
    for kv_chunk in kv_chunks:
        grads.setdefault(kv_chunk.meta.owner_rank, []).append(
            (kv_chunk.meta, kv_chunk.k.grad, kv_chunk.v.grad)
        )
    return grads
