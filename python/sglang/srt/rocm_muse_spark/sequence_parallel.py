"""Sequence-parallel Muse Spark prefill on ROCm, with Ulysses all-to-all attention.

Plain EXTEND batches of at least ``SP_MIN_TOKENS`` tokens run the norms, residuals, router and the replicated
pooler / unpooler on 1/TP of the tokens (each rank keeps a contiguous shard, padded to a multiple of TP).
Attention and the experts still see every token:

* Ulysses (``SGLANG_ROCM_MUSE_PREFILL_ULYSSES``): each rank projects its own token shard for every head with the
  gathered QKV/gate weight, an all-to-all hands every rank all tokens of its own heads, attention and the output
  norm/gate run as under TP, a second all-to-all returns all heads of the rank's tokens and the full O projection
  finishes them without a reduction. Per 16K tokens and layer that moves ~95 MB per rank instead of the
  all-gather + reduce-scatter's ~450 MB. Without it the shard is all-gathered before attention and the
  attention output reduce-scattered. With ``SGLANG_ROCM_MUSE_ULYSSES_PIPELINE`` both all-to-alls run in row chunks
  of the shard on a side stream, each overlapped with the projection of a neighbouring chunk.
* Experts: the shard's router outputs and expert inputs are all-gathered (as MXFP4 bytes when the a4w4 experts
  run, since per-row quantization commutes with the gather) and the expert output is reduce-scattered.

Target-verify batches of at least ``VERIFY_SP_MIN_ROWS`` rows (``SGLANG_ROCM_MUSE_VERIFY_SP``) shard the stream the same
way, so the replicated boundary norms, joint router / pooler and unpooler run on 1/TP of the rows; attention gathers
its input and reduce-scatters its output (no Ulysses: decode reads each weight once), and every collective runs on
the carv2 push plane (Lamport, CUDA-graph capturable) instead of the two all-reduces per layer.
"""

from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.kernels.ops.gemm.fp8_prefill_gemm import (
    fp8_prefill_gemm,
    quantize_head_rows_fp8,
    quantize_rows_fp8,
)
from sglang.kernels.ops.gemm.fp8w_small_m_gemm import quantize_weight
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.rocm_muse_spark.mxfp4_moe import _mx_ru_quant
from sglang.srt.runtime_context import get_parallel

SP_MIN_TOKENS = 2048
# Pipelined Ulysses row chunks, measured on MI355X per layer (check_ulysses_pipeline.py): shards from
# ULYSSES_PIPELINE_MIN_ROWS (2048-row shards in two chunks run 7 % slower, 3072-row ones 9 % faster) are cut into
# 2..ULYSSES_MAX_CHUNKS near-equal chunks of about ULYSSES_CHUNK_ROWS (whole waves of the [8192, 8192] O GEMM).
# The QKV side moves ULYSSES_QKV_SKEW_ROWS of the last chunk, whose all-to-all runs exposed, into the first when
# the last keeps ULYSSES_QKV_MIN_TAIL_ROWS: 4096 / 8192-row shards -9 / -15 % (equal chunks -6 / -14 %).
ULYSSES_PIPELINE_MIN_ROWS = 3072
ULYSSES_CHUNK_ROWS = 2048
ULYSSES_MAX_CHUNKS = 4
ULYSSES_QKV_SKEW_ROWS = 512
ULYSSES_QKV_MIN_TAIL_ROWS = 1536

_PREFILL_SP = envs.SGLANG_ROCM_MUSE_PREFILL_SP.get()
_PREFILL_ULYSSES = envs.SGLANG_ROCM_MUSE_PREFILL_ULYSSES.get()
_ULYSSES_LOCAL_GATE = envs.SGLANG_ROCM_MUSE_ULYSSES_LOCAL_GATE.get()
_ULYSSES_PIPELINE = envs.SGLANG_ROCM_MUSE_ULYSSES_PIPELINE.get()
_comm_stream = None
_VERIFY_SP = os.environ.get("SGLANG_ROCM_MUSE_VERIFY_SP", "1") == "1"
# From 128-row verify widths; narrower ones keep the all-reduce path.
VERIFY_SP_MIN_ROWS = 128
# ROCm push reduce-scatter with one warp group per peer (sp_collective reduce_scatter_res_unrolled), from 2 MiB of
# input (gfx1250 TP8, graph replay): faster from 2 MiB up, slower at 1 MiB, so the per-vector kernel stays below.
_RS_UNROLLED_MIN_BYTES = 2 << 20


@dataclass(frozen=True)
class SPShape:
    """Token counts of one sequence-parallel forward: real rows and rows padded to a multiple of TP. ``verify``:
    a target-verify batch, whose collectives run on the push plane."""

    num_tokens: int
    padded: int
    verify: bool = False


def sp_shape(
    forward_batch: ForwardBatch, num_tokens: int, row_bytes: int
) -> SPShape | None:
    """``row_bytes``: the widest row a verify collective pushes (the bf16 hidden row)."""
    world = get_parallel().tp_group.world_size
    if world == 1:
        return None
    padded = -(-num_tokens // world) * world
    if (
        _VERIFY_SP
        and forward_batch.forward_mode.is_target_verify()
        and num_tokens >= VERIFY_SP_MIN_ROWS
    ):
        push = _push_plane()
        if push and padded // world * row_bytes <= push[1]:
            return SPShape(num_tokens, padded, verify=True)
        return None
    if not (
        _PREFILL_SP
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and num_tokens >= SP_MIN_TOKENS
    ):
        return None
    return SPShape(num_tokens, padded)


def sp_shard(stream: torch.Tensor, sp: SPShape) -> torch.Tensor:
    """This rank's contiguous token shard of ``stream`` (padded to ``sp.padded`` rows)."""
    group = get_parallel().tp_group
    stream = F.pad(stream, (0, 0, 0, sp.padded - sp.num_tokens))
    return stream.tensor_split(group.world_size)[group.rank_in_group].contiguous()


def sp_gather(shard: torch.Tensor, rows: int, push: bool = False) -> torch.Tensor:
    """All-gather token shards and keep the first ``rows`` rows; ``push``: bf16 shards on the push plane (verify SP)."""
    group = get_parallel().tp_group
    shard = shard.contiguous()
    if push:
        spc = _push_plane()[0]
        out = shard.new_empty((shard.shape[0] * group.world_size, *shard.shape[1:]))
        spc.all_gather_push(
            group.world_size, shard, out, tuning=spc.Tuning(*_push_tuning())
        )
        return out[:rows]
    if shard.dtype == torch.uint8 and shard.dim() == 2 and shard.shape[1] % 2 == 0:
        # MXFP4 rows / E8M0 scales: gathered as bf16 words, so the aiter custom all-gather (f32 / f16 / bf16
        # only) carries them instead of RCCL; a gather copies bytes, the result is exact.
        return sp_gather(shard.view(torch.bfloat16), rows).view(torch.uint8)
    out = shard.new_empty((shard.shape[0] * group.world_size, *shard.shape[1:]))
    group.all_gather_into_tensor(out, shard)
    return out[:rows]


_PUSH = None  # (sp_collective module, push slot bytes) when the TP group runs carv2 with a push plane, else False


@functools.cache
def _num_cus() -> int:
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _push_tuning():
    """(blocks, threads) of the push gather / reduce-scatter: two blocks per CU."""
    return (2 * _num_cus(), 512)


def _rs_unrolled_tuning():
    """(blocks, threads) of the unrolled push reduce-scatter: one block per CU."""
    return (_num_cus(), 256)


def _push_plane():
    global _PUSH
    if _PUSH is None:
        _PUSH = False
        ca = get_parallel().tp_group.ca_comm
        obj = (
            getattr(ca, "obj", None)
            if ca is not None and not getattr(ca, "disabled", True)
            else None
        )
        push = obj.get_push() if obj is not None and hasattr(obj, "get_push") else None
        if push is not None:
            import sglang.kernels.ops.communication.sp_collective as spc

            spc.register_comm(lambda: obj, world_size=obj.world_size)
            _PUSH = (spc, int(push.slot_bytes))
            logger.info(
                "Muse SP collectives on carv2 push memory (slot %d bytes)",
                push.slot_bytes,
            )
    return _PUSH


_PUSH_TARGET = None


def sp_push_target(rows: int, row_bytes: int):
    """Verify SP: (slot base per owner rank [world] int64, phase stride bytes, phase counter, shard rows) for a
    producer that writes its partial rows [rows, row_bytes] straight into the owners' push slots, as
    ``sp_collective.reduce_scatter_pushed`` expects them; None when the push plane cannot hold a shard."""
    global _PUSH_TARGET
    group = get_parallel().tp_group
    push = _push_plane()
    if (
        not push
        or rows % group.world_size
        or rows // group.world_size * row_bytes > push[1]
    ):
        return None
    if _PUSH_TARGET is None:
        # the carv2 symmetric workspace: per rank [2 * world, slot_bytes] push slots and the phase word
        lease = group.ca_comm._lease
        workspaces, counter = lease.data.push_slots, lease.control.epoch
        rank, world = group.rank_in_group, group.world_size
        # slot (dst = owner, src = this rank) of phase 0; phase 1 sits world slots further
        bases = [workspaces[owner][rank].data_ptr() for owner in range(world)]
        _PUSH_TARGET = (torch.tensor(bases, dtype=torch.int64, device=counter.device),
                        world * workspaces[0].shape[1], counter)  # fmt: skip
    return (*_PUSH_TARGET, rows // group.world_size)


def sp_reduce_scatter_pushed(
    rows: int, width: int, dtype: torch.dtype, device
) -> torch.Tensor:
    """This rank's shard of rows a producer pushed with ``sp_push_target(rows, ...)``, summed over ranks."""
    group = get_parallel().tp_group
    spc = _push_plane()[0]
    out = torch.empty((rows // group.world_size, width), dtype=dtype, device=device)
    spc.reduce_scatter_pushed(group.world_size, out, tuning=spc.Tuning(*_push_tuning()))
    return out


def sp_reduce_scatter(
    full: torch.Tensor, padded: int, push: bool = False
) -> torch.Tensor:
    """Sum ``full`` over ranks and return this rank's token shard; ``push``: on the push plane (verify SP)."""
    group = get_parallel().tp_group
    if full.shape[0] != padded:
        full = F.pad(full, (0, 0, 0, padded - full.shape[0]))
    out = full.new_empty((padded // group.world_size, *full.shape[1:]))
    if push:
        spc = _push_plane()[0]
        if (
            torch.version.hip
            and full.numel() * full.element_size() >= _RS_UNROLLED_MIN_BYTES
        ):
            spc.reduce_scatter_res_unrolled(
                group.world_size,
                full.contiguous(),
                out,
                tuning=spc.Tuning(*_rs_unrolled_tuning()),
            )
        else:
            spc.reduce_scatter_res(
                group.world_size,
                full.contiguous(),
                out,
                tuning=spc.Tuning(*_push_tuning()),
            )
        return out
    group.reduce_scatter_tensor(out, full.contiguous())
    return out


def sp_gather_routed(
    x: torch.Tensor, topk_ids: torch.Tensor, topk_scores: torch.Tensor, rows: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Verify SP: push-plane gather of the bf16 expert input rows with their routing in one collective. The ids /
    scores travel one byte per bf16 word as 0x01XX, so no pushed 32-bit word is zero (the Lamport empty marker)."""
    n, width = x.shape
    side = torch.cat(
        [
            topk_ids.view(torch.uint8).view(n, -1),
            topk_scores.view(torch.uint8).view(n, -1),
        ],
        dim=1,
    )
    words = (side.to(torch.int16) | 256).view(torch.bfloat16)
    gathered = sp_gather(torch.cat([x, words], dim=1), rows, push=True)
    side = (gathered[:, width:].view(torch.int16) & 255).to(torch.uint8)
    ids_bytes = topk_ids[0].numel() * topk_ids.element_size()
    ids = (
        side[:, :ids_bytes]
        .contiguous()
        .view(topk_ids.dtype)
        .view(rows, *topk_ids.shape[1:])
    )
    scores = (
        side[:, ids_bytes:]
        .contiguous()
        .view(topk_scores.dtype)
        .view(rows, *topk_scores.shape[1:])
    )
    return gathered[:, :width], ids, scores


@triton.jit
def _side_words(w):
    # 32-bit words -> two words of 16-bit 0x01XX halves, one byte each: no pushed word is zero (the Lamport marker)
    b0, b1, b2, b3 = w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF
    return (
        tl.reshape(tl.join(b0 | (b1 << 16), b2 | (b3 << 16)), (2 * w.shape[0],))
        | 0x01000100
    )


@triton.jit
def _side_unwords(v):
    lo, hi = tl.split(tl.reshape(v, (v.shape[0] // 2, 2)))
    return (
        (lo & 0xFF)
        | (((lo >> 16) & 0xFF) << 8)
        | ((hi & 0xFF) << 16)
        | (((hi >> 16) & 0xFF) << 24)
    )


@triton.jit
def _routed_mx_pack_kernel(
    X,
    IDS,
    SC,
    OUT,
    W,
    eps,
    K: tl.constexpr,
    IDW: tl.constexpr,
    SCW: tl.constexpr,
    NORM: tl.constexpr,
):
    # One row: MXFP8 payload (aiter's MoE input-quant rule) as int32 words, e8m0 scales two per word as 0x01XX
    # halves, then the routing ids / scores words split the same way. NORM: the row is first RMS-normalized with
    # weight W in BF16, as MuseSpark1xRMSNorm returns it (same math as F.rms_norm, its own summation order).
    row = tl.program_id(0).to(tl.int64)
    width = K // 4 + K // 64 + 2 * IDW + 2 * SCW
    out = OUT + row * width
    x = tl.load(X + row * K + tl.arange(0, K)).to(tl.float32)
    if NORM:
        rstd = tl.rsqrt(tl.sum(x * x, axis=0) / K + eps)
        x = (
            (tl.load(W + tl.arange(0, K)).to(tl.float32) * (rstd * x))
            .to(tl.bfloat16)
            .to(tl.float32)
        )
    q, e = _mx_ru_quant(tl.reshape(x, (K // 32, 32)))
    qb = tl.reshape(q.to(tl.uint8, bitcast=True).to(tl.int32), (K // 4, 4))
    tl.store(
        out + tl.arange(0, K // 4), tl.sum(qb << (tl.arange(0, 4) * 8)[None, :], axis=1)
    )
    e0, e1 = tl.split(tl.reshape(e, (K // 64, 2)))
    tl.store(out + K // 4 + tl.arange(0, K // 64), (e0 | (e1 << 16)) | 0x01000100)
    tl.store(
        out + K // 4 + K // 64 + tl.arange(0, 2 * IDW),
        _side_words(tl.load(IDS + row * IDW + tl.arange(0, IDW))),
    )
    tl.store(
        out + K // 4 + K // 64 + 2 * IDW + tl.arange(0, 2 * SCW),
        _side_words(tl.load(SC + row * SCW + tl.arange(0, SCW))),
    )


@triton.jit
def _routed_mx_unpack_kernel(
    PACKED,
    X,
    IDS,
    SC,
    COUNT,
    K: tl.constexpr,
    IDW: tl.constexpr,
    SCW: tl.constexpr,
    HAS_COUNT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    width = K // 4 + K // 64 + 2 * IDW + 2 * SCW
    src = PACKED + row * width
    w = tl.load(src + tl.arange(0, K // 4))
    qb = ((w[:, None] >> (tl.arange(0, 4) * 8)[None, :]) & 0xFF).to(tl.uint8)
    q = tl.reshape(qb.to(tl.float8e4nv, bitcast=True).to(tl.float32), (K // 32, 32))
    v = tl.load(src + K // 4 + tl.arange(0, K // 64))
    e = tl.reshape(tl.join(v & 0xFF, (v >> 16) & 0xFF), (K // 32,)).to(tl.float32)
    # e4m3 x 2^(e8m0 - 127) is exact in BF16 (3 mantissa bits, fp32 exponent range)
    tl.store(
        X + row * K + tl.arange(0, K),
        tl.reshape(q * tl.exp2(e - 127.0)[:, None], (K,)).to(tl.bfloat16),
    )
    ids = _side_unwords(tl.load(src + K // 4 + K // 64 + tl.arange(0, 2 * IDW)))
    tl.store(IDS + row * IDW + tl.arange(0, IDW), ids)
    if HAS_COUNT:
        # routed rows per expert (int32 ids) for expert_placement
        tl.atomic_add(COUNT + ids, 1, sem="relaxed")
    tl.store(
        SC + row * SCW + tl.arange(0, SCW),
        _side_unwords(
            tl.load(src + K // 4 + K // 64 + 2 * IDW + tl.arange(0, 2 * SCW))
        ),
    )


@triton.jit
def _routed_side_words(SRC, J: tl.constexpr):
    # 32-bit words from their 0x01XX halves (_side_words) at SRC [rows, 2 * J], read as even / odd strided loads
    lo = tl.load(SRC + 2 * tl.arange(0, J)[None, :])
    hi = tl.load(SRC + 2 * tl.arange(0, J)[None, :] + 1)
    return (
        (lo & 0xFF)
        | (((lo >> 16) & 0xFF) << 8)
        | ((hi & 0xFF) << 16)
        | (((hi >> 16) & 0xFF) << 24)
    )


@triton.jit
def _routed_side_unpack_kernel(
    PACKED,
    IDS,
    SC,
    COUNT,
    LOCAL,
    rows,
    local_start,
    num_local,
    K: tl.constexpr,
    IDW: tl.constexpr,
    SCW: tl.constexpr,
    HAS_COUNT: tl.constexpr,
    HAS_LOCAL: tl.constexpr,
    BR: tl.constexpr,
):
    # The routing words of BR gathered rows; their MXFP8 payload stays packed for scatter_routed_mx. HAS_LOCAL (int32
    # ids): also the rank-local ids, -1 for experts of other ranks (mxfp4_moe._local_expert_ids).
    r = tl.minimum(tl.program_id(0) * BR + tl.arange(0, BR), rows - 1).to(tl.int64)
    width = K // 4 + K // 64 + 2 * IDW + 2 * SCW
    ids = _routed_side_words(PACKED + r[:, None] * width + K // 4 + K // 64, IDW)
    tl.store(IDS + r[:, None] * IDW + tl.arange(0, IDW)[None, :], ids)
    if HAS_LOCAL:
        local = ids - local_start
        tl.store(
            LOCAL + r[:, None] * IDW + tl.arange(0, IDW)[None, :],
            tl.where((local >= 0) & (local < num_local), local, -1),
        )
    if HAS_COUNT:
        # rows past the end repeat the last row: count each route once
        live = (tl.program_id(0) * BR + tl.arange(0, BR)) < rows
        tl.atomic_add(COUNT + ids, 1, mask=live[:, None], sem="relaxed")
    sc = _routed_side_words(
        PACKED + r[:, None] * width + K // 4 + K // 64 + 2 * IDW, SCW
    )
    tl.store(SC + r[:, None] * SCW + tl.arange(0, SCW)[None, :], sc)


@triton.jit
def _routed_mx_scatter_kernel(
    PACKED,
    T2R,
    NVR,
    OUTP,
    OUTS,
    tokens,
    K: tl.constexpr,
    IDW: tl.constexpr,
    SCW: tl.constexpr,
    TOPK: tl.constexpr,
    WMMA_REP: tl.constexpr,
    BT: tl.constexpr,
    HAS_NVR: tl.constexpr,
):
    # BT gathered rows per program, each read once and written to every grouped row its local routes map to
    # (T2R[token * TOPK + slot] >= 0, route < NVR): the payload as is, the e8m0 bytes as dwords in the GEMM's
    # preshuffled order (aiter moe_fused_route_quant_scatter: tile = row // (WMMA_REP * 16), dword g of the row at
    # tile * (K / 128) * WMMA_REP * 16 + g * WMMA_REP * 16 + (row % (WMMA_REP * 16)) // 16 * 16 + row % 16).
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    t64 = t.to(tl.int64)
    nvr = tl.load(NVR) if HAS_NVR else tokens * TOPK
    width = K // 4 + K // 64 + 2 * IDW + 2 * SCW
    src = PACKED + t64[:, None] * width
    live = tl.zeros((BT,), dtype=tl.int1)
    for s in tl.static_range(TOPK):
        route = t * TOPK + s
        live = live | (
            tl.load(T2R + route, mask=(t < tokens) & (route < nvr), other=-1) >= 0
        )
    c = tl.arange(0, K // 4)
    payload = tl.load(src + c[None, :], mask=live[:, None])
    g = tl.arange(0, K // 128)
    lo = tl.load(src + K // 4 + 2 * g[None, :], mask=live[:, None])
    hi = tl.load(src + K // 4 + 2 * g[None, :] + 1, mask=live[:, None])
    dw = (
        (lo & 0xFF)
        | (((lo >> 16) & 0xFF) << 8)
        | ((hi & 0xFF) << 16)
        | (((hi >> 16) & 0xFF) << 24)
    )
    rows_per_tile = WMMA_REP * 16
    for s in tl.static_range(TOPK):
        route = t * TOPK + s
        d = tl.load(T2R + route, mask=(t < tokens) & (route < nvr), other=-1).to(
            tl.int64
        )
        ok = d >= 0
        tl.store(OUTP + d[:, None] * (K // 4) + c[None, :], payload, mask=ok[:, None])
        base = (
            (d // rows_per_tile) * ((K // 128) * rows_per_tile)
            + (d % rows_per_tile) // 16 * 16
            + d % 16
        )
        tl.store(
            OUTS + base[:, None] + g[None, :] * rows_per_tile, dw, mask=ok[:, None]
        )


def scatter_routed_mx(
    packed: torch.Tensor,
    topids_to_rows: torch.Tensor,
    num_valid_routes: torch.Tensor | None,
    max_m: int,
    wmma_rep: int,
    k: int,
    idw: int,
    scw: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """aiter's grouped MXFP8 GEMM input (payload uint8 [1, max_m, k], preshuffled e8m0 [1, max_m / wmma_rep,
    k / 32 * wmma_rep]) from ``sp_gather_routed_mx(mx=True)`` rows, which its prequantized route scatter would build
    from the unpacked rows: ``topids_to_rows`` [rows * topk] maps each route to its grouped row (-1: not here)."""
    tokens = packed.shape[0]
    payload = torch.empty((1, max_m, k), dtype=torch.uint8, device=packed.device)
    scale = torch.empty(
        (1, max_m // wmma_rep, k // 32 * wmma_rep),
        dtype=torch.uint8,
        device=packed.device,
    )
    if tokens:
        # rows per program / warps: 4 / 4 at prefill widths, 1 / 1 at verify widths (faster at 128-512 rows)
        bt, warps = (1, 1) if tokens <= 1024 else (4, 4)
        _routed_mx_scatter_kernel[(triton.cdiv(tokens, bt),)](
            packed, topids_to_rows, topids_to_rows if num_valid_routes is None else num_valid_routes,
            payload.view(torch.int32), scale.view(torch.int32), tokens, k, idw, scw, topk, wmma_rep, bt,
            num_valid_routes is not None, num_warps=warps,
        )  # fmt: skip
    return payload, scale


def sp_gather_routed_mx(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_scores: torch.Tensor,
    rows: int,
    count: torch.Tensor | None = None,
    push: bool = True,
    mx: bool = False,
    norm: tuple[torch.Tensor, float] | None = None,
    local_experts: tuple[int, int] | None = None,
) -> tuple[torch.Tensor | tuple, torch.Tensor, torch.Tensor]:
    """``sp_gather_routed`` with the expert input rows sent as MXFP8 under aiter's own MoE input-quant rule and
    widened back to BF16 on arrival (exact): the experts quantize them to the same values, at half the BF16 bytes,
    with the routing in the same rows. ``count`` (int32 per expert, int32 ids only): adds the gathered rows' routes.
    ``push``: on the push plane (verify SP), else the RCCL gather (prefill sizes). ``mx``: the rows stay as they
    travelled and only the routing is unpacked; the first return is then (rows int32 [rows, width], id words,
    score words per row, rank-local ids or None), for ``scatter_routed_mx``; ``local_experts`` = (first, count) of
    this rank's experts asks for the local ids (int32 ids only). ``norm`` =
    (BF16 weight, eps): ``x`` is the pre-norm row and the pack applies that RMSNorm (no separate norm pass)."""
    n, k = x.shape
    idw = topk_ids[0].numel() * topk_ids.element_size() // 4
    scw = topk_scores[0].numel() * topk_scores.element_size() // 4
    width = k // 4 + k // 64 + 2 * idw + 2 * scw
    packed = torch.empty((n, width), dtype=torch.int32, device=x.device)
    if n:
        weight, eps = norm if norm is not None else (x, 0.0)
        _routed_mx_pack_kernel[(n,)](
            x.contiguous(), topk_ids.contiguous().view(torch.int32), topk_scores.contiguous().view(torch.int32),
            packed, weight, eps, k, idw, scw, norm is not None, num_warps=4,
        )  # fmt: skip
    gathered = sp_gather(packed.view(torch.bfloat16), rows, push=push).view(torch.int32)
    ids = torch.empty(
        (rows, *topk_ids.shape[1:]), dtype=topk_ids.dtype, device=x.device
    )
    scores = torch.empty(
        (rows, *topk_scores.shape[1:]), dtype=topk_scores.dtype, device=x.device
    )
    has_count = count is not None and topk_ids.dtype == torch.int32
    if mx:
        local = (
            torch.empty_like(ids)
            if local_experts is not None and topk_ids.dtype == torch.int32
            else None
        )
        first, num = local_experts if local is not None else (0, 0)
        br = 16
        _routed_side_unpack_kernel[(triton.cdiv(rows, br),)](
            gathered, ids.view(torch.int32), scores.view(torch.int32), count if has_count else ids,
            ids if local is None else local, rows, first, num, k, idw, scw, has_count, local is not None, br,
            num_warps=4,
        )  # fmt: skip
        return (gathered, idw, scw, local), ids, scores
    out = torch.empty((rows, k), dtype=torch.bfloat16, device=x.device)
    _routed_mx_unpack_kernel[(rows,)](
        gathered, out, ids.view(torch.int32), scores.view(torch.int32), count if has_count else out,
        k, idw, scw, has_count, num_warps=4,
    )  # fmt: skip
    return out, ids, scores


@triton.jit
def _copy_rows_kernel(
    SRC,
    DST,
    rows,
    src_s0,
    src_s1,
    dst_s0,
    dst_s1,
    C: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    w = tl.program_id(1).to(tl.int64)
    rmask = r < rows
    for c0 in tl.static_range(0, C, BC):
        c = c0 + tl.arange(0, BC)
        mask = rmask[:, None] & (c < C)[None, :]
        x = tl.load(
            SRC + w * src_s0 + r.to(tl.int64)[:, None] * src_s1 + c[None, :], mask=mask
        )
        tl.store(
            DST + w * dst_s0 + r.to(tl.int64)[:, None] * dst_s1 + c[None, :],
            x,
            mask=mask,
        )


def _copy_rows(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.copy_(src)`` for [world, rows, C] views whose last dim is contiguous (the Ulysses head-major regroupings
    of the QKV projection output): one pass at memory speed where torch's strided copy is much slower."""
    world, rows, c = src.shape
    assert dst.shape == src.shape and src.stride(2) == 1 and dst.stride(2) == 1
    br = 16
    _copy_rows_kernel[(triton.cdiv(rows, br), world)](
        src,
        dst,
        rows,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        c,
        br,
        256,
        num_warps=4,
    )


def _pad_rows(x: torch.Tensor, rows: int) -> torch.Tensor:
    """``x`` with zero rows appended up to ``rows``, contiguous (no copy when it already is and has them)."""
    if x.shape[0] != rows:
        x = F.pad(x, (0, 0, 0, rows - x.shape[0]))
    return x.contiguous()


def _sp_comm_stream() -> torch.cuda.Stream:
    global _comm_stream
    if _comm_stream is None:
        # Default priority: RCCL collectives issued from a high-priority stream through torch's process group
        # deadlock on ROCm (measured 10-23; the same sequence completes at priority 0).
        _comm_stream = torch.cuda.Stream()
    return _comm_stream


def ulysses_enabled() -> bool:
    return _PREFILL_SP and _PREFILL_ULYSSES and get_parallel().tp_group.world_size > 1


@torch.no_grad()
def prepare_ulysses(attention, fp8: bool) -> None:
    """Gather the TP shards of the QKV/gate and O projections once after loading (rank-major head order); with
    ``fp8`` keep only their FP8 copies + channel scales (``fp8_prefill_gemm``), half the bytes of the BF16 gather."""
    group = get_parallel().tp_group
    qkv = group.all_gather(attention.qkv_proj.weight.data.contiguous(), dim=0)
    o = group.all_gather(attention.o_proj.weight.data.contiguous(), dim=1).contiguous()
    attention.ulysses_qkv_weight = None if fp8 else qkv
    attention.ulysses_o_weight = None if fp8 else o
    attention.ulysses_qkv_fp8 = quantize_weight(qkv) if fp8 else None
    attention.ulysses_o_fp8 = quantize_weight(o) if fp8 else None


def ulysses_prepared(attention) -> bool:
    return (
        attention.ulysses_qkv_weight is not None
        or attention.ulysses_qkv_fp8 is not None
    )


def _all_to_all(x: torch.Tensor, group) -> torch.Tensor:
    out = torch.empty_like(x)
    dist.all_to_all_single(out, x, group=group.device_group)
    return out


def ulysses_attention(
    attention,
    x: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    sp: SPShape,
    x_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Attention block for this rank's token shard ``x`` [padded / world, hidden]; returns its output rows.
    ``x_fp8`` = (per-row FP8 copy of ``x``, scales) from the boundary kernel, for the FP8 QKV projection."""
    group = get_parallel().tp_group
    world, rows = group.world_size, x.shape[0]
    plan = _ulysses_chunk_plan(attention, rows)
    if plan is not None:
        return _ulysses_attention_pipelined(
            attention, x, x_fp8, positions, forward_batch, sp, group, plan
        )
    qkv = _qkv_projection(attention, x, x_fp8).view(rows, world, -1)
    if _ULYSSES_LOCAL_GATE:
        return _ulysses_attention_local_gate(
            attention, qkv, positions, forward_batch, sp, group
        )
    # [rows, hidden] shard -> all-to-all -> [world * rows, q|k|v|gate of this rank's heads]
    qkv = _all_to_all(qkv.transpose(0, 1).contiguous(), group).view(world * rows, -1)
    q, k, v, o_gate = qkv[: sp.num_tokens].split(
        [attention.q_size, attention.kv_size, attention.kv_size, attention.q_size],
        dim=-1,
    )
    q, k = attention.qk_norm_rope(positions, q, k)
    out = attention.attn(q, k, v, forward_batch)
    out = attention._fused_o_norm_gate.forward(out, o_gate)
    # [tokens, this rank's heads] -> all-to-all -> full O projection of this rank's token shard
    out = _pad_rows(out, sp.padded)
    shard_rows = sp.padded // world
    heads = _all_to_all(out, group).view(world, shard_rows, -1)
    if attention.ulysses_o_fp8 is not None:
        # The row quantization reads the head blocks in place, no transposed copy.
        xq, xs = quantize_head_rows_fp8(heads)
        return fp8_prefill_gemm(xq, xs, attention.ulysses_o_fp8)
    heads = heads.transpose(0, 1).reshape(shard_rows, -1)
    return F.linear(heads, attention.ulysses_o_weight)


def _qkv_projection(attention, x, x_fp8) -> torch.Tensor:
    if attention.ulysses_qkv_fp8 is not None:
        xq, xs = x_fp8 if x_fp8 is not None else quantize_rows_fp8(x)
        return fp8_prefill_gemm(xq, xs, attention.ulysses_qkv_fp8)
    return F.linear(x, attention.ulysses_qkv_weight)


def _gated_heads(attention, heads: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Norm-gate of ``heads`` [world, rows, this rank's heads], the output all-to-all result for ``rows``
    token-shard rows, with their kept ``gate`` [world, rows, gate]."""
    world, rows = heads.shape[0], heads.shape[1]
    return attention._fused_o_norm_gate.forward(
        heads.view(world * rows, -1), gate.view(world * rows, -1)
    ).view(world, rows, -1)


def _o_projection(attention, heads: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    heads = _gated_heads(attention, heads, gate)
    if attention.ulysses_o_fp8 is not None:
        xq, xs = quantize_head_rows_fp8(heads)
        return fp8_prefill_gemm(xq, xs, attention.ulysses_o_fp8)
    rows = heads.shape[1]
    return F.linear(heads.transpose(0, 1).reshape(rows, -1), attention.ulysses_o_weight)


def _fp8_prefill_gemm_into(out, xq, xs, fp8_weight) -> None:
    """``fp8_prefill_gemm`` written into ``out``, a row block of a preallocated output: the same aiter CK call
    (no tuned config for these shapes, so no split-K)."""
    from aiter import gemm_a8w8_ck

    wq, ws = fp8_weight
    gemm_a8w8_ck(xq, wq, xs.view(-1, 1), ws.view(-1, 1), out, None, 0)


def _ulysses_attention_local_gate(
    attention, qkv, positions, forward_batch, sp: SPShape, group
) -> torch.Tensor:
    """``ulysses_attention`` with the output gate kept on the token shard: the QKV all-to-all carries q|k|v only
    and the per-(token, head) norm-gate runs on the head-major output all-to-all result, so every gated value is
    computed by the same kernel from the same operands as on the head shard."""
    rows, world = qkv.shape[0], qkv.shape[1]
    qkv_width = attention.q_size + 2 * attention.kv_size
    # [rows, world, q|k|v|gate] -> [world, rows, q|k|v] to send (one copy); the gate stays in place for the BF16 O
    # projection, whose norm-gate reads it token-major, and is copied to [world, rows, gate] for the FP8 one.
    send = qkv.new_empty((world, rows, qkv_width))
    _copy_rows(send, qkv[:, :, :qkv_width].transpose(0, 1))
    gate = qkv[:, :, qkv_width:]
    if attention.ulysses_o_fp8 is not None:
        gate_rows = qkv.new_empty((world, rows, gate.shape[2]))
        _copy_rows(gate_rows, gate.transpose(0, 1))
        gate = gate_rows
    recv = _all_to_all(send, group).view(world * rows, qkv_width)
    q, k, v = recv[: sp.num_tokens].split(
        [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
    )
    q, k = attention.qk_norm_rope(positions, q, k)
    out = attention.attn(q, k, v, forward_batch)
    # Padded rows enter the norm-gate as zeros and leave as zeros, as when they were padded after it.
    out = _pad_rows(out, sp.padded)
    heads = _all_to_all(out, group).view(world, rows, -1)
    if attention.ulysses_o_fp8 is None:
        return F.linear(
            attention._fused_o_norm_gate.forward_token_major(heads, gate),
            attention.ulysses_o_weight,
        )
    return _o_projection(attention, heads, gate)


def _ulysses_chunk_plan(
    attention, rows: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """(QKV-side, output-side) row ranges of the pipelined Ulysses shard, or None when it runs unchunked. FP8
    projections only: aiter's CK FP8 GEMM runs one kernel config for every M >= 1024 (no tuned rows for these
    shapes) and the gfx1250 MXFP8 GEMM one per weight shape, so chunk rows are bit-identical to whole-shard rows,
    while hipBLASLt picks its BF16 algorithm per M.
    Every chunk keeps at least 1024 rows."""
    if not (
        _ULYSSES_PIPELINE
        and _ULYSSES_LOCAL_GATE
        and attention.ulysses_qkv_fp8 is not None
        and attention.ulysses_o_fp8 is not None
    ):
        return None
    if rows < ULYSSES_PIPELINE_MIN_ROWS:
        return None
    count = min(
        ULYSSES_MAX_CHUNKS, (rows + ULYSSES_CHUNK_ROWS // 2) // ULYSSES_CHUNK_ROWS
    )
    bounds = [rows * i // count for i in range(count + 1)]
    o_chunks = _row_ranges(bounds)
    if rows - bounds[-2] - ULYSSES_QKV_SKEW_ROWS >= ULYSSES_QKV_MIN_TAIL_ROWS:
        bounds = [0] + [b + ULYSSES_QKV_SKEW_ROWS for b in bounds[1:-1]] + [rows]
    return _row_ranges(bounds), o_chunks


def _row_ranges(bounds: list[int]) -> list[tuple[int, int]]:
    return list(zip(bounds[:-1], bounds[1:]))


def _ulysses_attention_pipelined(
    attention, x, x_fp8, positions, forward_batch, sp: SPShape, group, plan
) -> torch.Tensor:
    """``_ulysses_attention_local_gate`` with both all-to-alls cut into row chunks of the shard (``plan``): every
    chunk's projections are row-local, so the result is bit-identical while the side stream moves one chunk as the
    main stream projects another."""
    qkv_chunks, o_chunks = plan
    recv, gates = _pipelined_qkv_all_to_all(
        attention, x, x_fp8, group, qkv_chunks=qkv_chunks, o_chunks=o_chunks
    )
    q, k, v = recv.view(-1, recv.shape[-1])[: sp.num_tokens].split(
        [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
    )
    q, k = attention.qk_norm_rope(positions, q, k)
    out = attention.attn(q, k, v, forward_batch)
    out = _pad_rows(out, sp.padded)
    return _pipelined_o_projection(
        attention, out, gates, group, o_chunks=o_chunks, hidden=x.shape[1]
    )


def _pipelined_qkv_all_to_all(attention, x, x_fp8, group, qkv_chunks, o_chunks):
    """Per QKV chunk: projection on the main stream, its q|k|v all-to-all on the side stream straight into
    ``recv`` [world, rows, q|k|v] (the unchunked all-to-all's layout); the gate is kept per output chunk
    as [world, chunk rows, gate]."""
    world, rows = group.world_size, x.shape[0]
    qkv_width = attention.q_size + 2 * attention.kv_size
    xq, xs = x_fp8 if x_fp8 is not None else quantize_rows_fp8(x)
    main, comm = torch.cuda.current_stream(), _sp_comm_stream()
    recv = x.new_empty((world, rows, qkv_width))
    recv.record_stream(comm)
    gates = [x.new_empty((world, hi - lo, attention.q_size)) for lo, hi in o_chunks]
    for lo, hi in qkv_chunks:
        qkv = fp8_prefill_gemm(xq[lo:hi], xs[lo:hi], attention.ulysses_qkv_fp8)
        qkv = qkv.view(hi - lo, world, -1)
        send = qkv.new_empty((world, hi - lo, qkv_width))
        _copy_rows(send, qkv[:, :, :qkv_width].transpose(0, 1))
        comm.wait_stream(main)
        with torch.cuda.stream(comm):
            send.record_stream(comm)
            # Sender s's chunk rows land at recv[s, lo:hi], each a contiguous block.
            dist.all_to_all(
                list(recv[:, lo:hi].unbind(0)),
                list(send.unbind(0)),
                group=group.device_group,
            )
        for (a, b), gate in zip(o_chunks, gates):
            if a < hi and lo < b:
                first, last = max(a, lo), min(b, hi)
                _copy_rows(
                    gate[:, first - a : last - a],
                    qkv[first - lo : last - lo, :, qkv_width:].transpose(0, 1),
                )
    main.wait_stream(comm)
    return recv, gates


def _pipelined_o_projection(attention, out, gates, group, o_chunks, hidden):
    """Output all-to-all of ``out`` [world * rows, this rank's heads] per row chunk on the side stream; each
    chunk's norm-gate + O projection runs on the main stream as soon as it lands."""
    world = group.world_size
    out = out.view(world, -1, out.shape[-1])
    main, comm = torch.cuda.current_stream(), _sp_comm_stream()
    result = out.new_empty((out.shape[1], hidden))
    comm.wait_stream(main)
    received = []
    with torch.cuda.stream(comm):
        out.record_stream(comm)
        for lo, hi in o_chunks:
            heads = out.new_empty((world, hi - lo, out.shape[-1]))
            dist.all_to_all(
                list(heads.unbind(0)),
                list(out[:, lo:hi].unbind(0)),
                group=group.device_group,
            )
            received.append((heads, comm.record_event()))
    for (lo, hi), (heads, ready), gate in zip(o_chunks, received, gates):
        main.wait_event(ready)
        heads.record_stream(main)
        xq, xs = quantize_head_rows_fp8(_gated_heads(attention, heads, gate))
        # Written in place: a per-chunk copy runs at a sixth of its speed next to the RCCL kernels.
        _fp8_prefill_gemm_into(result[lo:hi], xq, xs, attention.ulysses_o_fp8)
    return result


logger = logging.getLogger(__name__)
