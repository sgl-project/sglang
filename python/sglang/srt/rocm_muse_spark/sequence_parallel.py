"""Sequence-parallel Muse Spark prefill on ROCm, with Ulysses all-to-all attention.

Plain EXTEND batches of at least ``SP_MIN_TOKENS`` tokens run the norms, residuals, router and the replicated
pooler / unpooler on 1/TP of the tokens (each rank keeps a contiguous shard, padded to a multiple of TP).
Attention and the experts still see every token:

* Ulysses (``SGLANG_MUSE_PREFILL_ULYSSES``): each rank projects its own token shard for every head with the
  gathered QKV/gate weight, an all-to-all hands every rank all tokens of its own heads, attention and the output
  norm/gate run as under TP, a second all-to-all returns all heads of the rank's tokens and the full O projection
  finishes them without a reduction. Per 16K tokens and layer that moves ~95 MB per rank instead of the
  all-gather + reduce-scatter's ~450 MB. Without it the shard is all-gathered before attention and the
  attention output reduce-scattered. With ``SGLANG_MUSE_ULYSSES_PIPELINE`` both all-to-alls run in row chunks
  of the shard on a side stream, each overlapped with the projection of a neighbouring chunk.
* Experts: the shard's router outputs and expert inputs are all-gathered (as MXFP4 bytes when the a4w4 experts
  run, since per-row quantization commutes with the gather) and the expert output is reduce-scattered.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.kernels.ops.gemm.fp8_prefill_gemm import (
    fp8_prefill_gemm,
    quantize_head_rows_fp8,
    quantize_rows_fp8,
)
from sglang.kernels.ops.gemm.fp8w_small_m_gemm import quantize_weight
from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

SP_MIN_TOKENS = 2048
# Overlapped SP experts: the second of the two expert passes takes 1 - MOE_OVERLAP_FIRST_SHARE of the shard
# rows (at least MOE_OVERLAP_MIN_PASS_ROWS). Its reduce-scatter is the exposed tail, and the first pass's must fit
# under the second pass's experts. Measured 32K extend over a 100K prefix (4096-row shards), TP0 GPU window:
# share 0.5 / 0.6 / 0.7 / 0.75 -> 1167.7 / 1162.3 / 1156.3 / 1149.8 ms, serialized experts 1184.0 ms.
MOE_OVERLAP_FIRST_SHARE = 0.75
# aiter's CK FP8 GEMM runs one kernel config for every M >= 1024, so the per-pass unpooler matches the
# whole-shard one bit for bit.
MOE_OVERLAP_MIN_PASS_ROWS = 1024
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

_PREFILL_SP = envs.SGLANG_MUSE_PREFILL_SP.get()
_PREFILL_ULYSSES = envs.SGLANG_MUSE_PREFILL_ULYSSES.get()
_ULYSSES_LOCAL_GATE = envs.SGLANG_MUSE_ULYSSES_LOCAL_GATE.get()
_ULYSSES_PIPELINE = envs.SGLANG_MUSE_ULYSSES_PIPELINE.get()
_MOE_OVERLAP = envs.SGLANG_MUSE_PREFILL_MOE_OVERLAP.get()
_comm_stream = None


@dataclass(frozen=True)
class SPShape:
    """Token counts of one sequence-parallel prefill: real rows and rows padded to a multiple of TP."""

    num_tokens: int
    padded: int


def sp_shape(forward_batch: ForwardBatch, num_tokens: int) -> SPShape | None:
    if not (
        _PREFILL_SP
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and num_tokens >= SP_MIN_TOKENS
    ):
        return None
    world = get_tp_group().world_size
    if world == 1:
        return None
    return SPShape(num_tokens, -(-num_tokens // world) * world)


def sp_shard(stream: torch.Tensor, sp: SPShape) -> torch.Tensor:
    """This rank's contiguous token shard of ``stream`` (padded to ``sp.padded`` rows)."""
    group = get_tp_group()
    stream = F.pad(stream, (0, 0, 0, sp.padded - sp.num_tokens))
    return stream.tensor_split(group.world_size)[group.rank_in_group].contiguous()


def sp_gather(shard: torch.Tensor, rows: int) -> torch.Tensor:
    """All-gather token shards and keep the first ``rows`` rows."""
    group = get_tp_group()
    shard = shard.contiguous()
    if shard.dtype == torch.uint8 and shard.dim() == 2 and shard.shape[1] % 2 == 0:
        # MXFP4 rows / E8M0 scales: gathered as bf16 words, so the aiter custom all-gather (f32 / f16 / bf16
        # only) carries them instead of RCCL; a gather copies bytes, the result is exact.
        return sp_gather(shard.view(torch.bfloat16), rows).view(torch.uint8)
    out = shard.new_empty((shard.shape[0] * group.world_size, *shard.shape[1:]))
    group.all_gather_into_tensor(out, shard)
    return out[:rows]


def sp_reduce_scatter(full: torch.Tensor, padded: int) -> torch.Tensor:
    """Sum ``full`` over ranks and return this rank's token shard."""
    group = get_tp_group()
    if full.shape[0] != padded:
        full = F.pad(full, (0, 0, 0, padded - full.shape[0]))
    out = full.new_empty((padded // group.world_size, *full.shape[1:]))
    group.reduce_scatter_tensor(out, full.contiguous())
    return out


def _moe_first_pass_rows(shard_rows: int) -> int:
    second = round(shard_rows * (1 - MOE_OVERLAP_FIRST_SHARE))
    return shard_rows - max(MOE_OVERLAP_MIN_PASS_ROWS, second)


def moe_overlap_applies(shard_rows: int) -> bool:
    return (
        _MOE_OVERLAP and _moe_first_pass_rows(shard_rows) >= MOE_OVERLAP_MIN_PASS_ROWS
    )


def _sp_comm_stream() -> torch.cuda.Stream:
    global _comm_stream
    if _comm_stream is None:
        # Default priority: RCCL collectives issued from a high-priority stream through torch's process group
        # deadlock on ROCm (measured 10-23; the same sequence completes at priority 0).
        _comm_stream = torch.cuda.Stream()
    return _comm_stream


def _rccl_gather(shard: torch.Tensor, rows: int) -> torch.Tensor:
    group = get_tp_group()
    src = shard.contiguous()
    # FP8 / E8M0 / FP4-packed rows travel as bytes (RCCL has no fp8 types here); a gather copies bytes exactly.
    if src.element_size() == 1:
        src = src.view(torch.uint8)
    out = src.new_empty((src.shape[0] * group.world_size, *src.shape[1:]))
    dist.all_gather_into_tensor(out, src, group=group.device_group)
    return out[:rows].view(shard.dtype)


def _rccl_reduce_scatter(full: torch.Tensor) -> torch.Tensor:
    group = get_tp_group()
    out = full.new_empty((full.shape[0] // group.world_size, *full.shape[1:]))
    dist.reduce_scatter_tensor(out, full.contiguous(), group=group.device_group)
    return out


def _group_gather(shard: torch.Tensor, rows: int) -> torch.Tensor:
    """``sp_gather`` of a row tensor of any dtype; 1-byte rows travel as uint8, which it moves as bf16 words."""
    if shard.element_size() == 1 and shard.dtype != torch.uint8:
        return sp_gather(shard.view(torch.uint8), rows).view(shard.dtype)
    return sp_gather(shard, rows)


def _side_reduce_scatter(full: torch.Tensor, main, comm):
    """RCCL reduce-scatter of ``full`` on the side stream once the main stream has produced it; returns the shard
    and its ready event."""
    done = main.record_event()
    with torch.cuda.stream(comm):
        comm.wait_event(done)
        full.record_stream(comm)
        return _rccl_reduce_scatter(full), comm.record_event()


def overlapped_expert_shard(inputs, run_experts, finish=None) -> torch.Tensor:
    """This rank's token shard of the expert output summed over ranks, mapped by ``finish`` (row-local, e.g. the
    unpooler; None keeps it), with the shard cut into a larger first and a smaller second expert pass.

    The first pass's ``inputs`` (row-aligned [shard, ...] tensors) are all-gathered up front on the main stream,
    where nothing could hide them and aiter's custom all-gather is the fastest route; a side stream gathers the
    second pass's under the first pass's experts, reduce-scatters the first pass's ``run_experts`` output under
    the second pass's experts, then the second pass's under ``finish`` of the first pass's rows. Experts and
    ``finish`` are row-local, so the result equals the two passes run back to back; against one whole-shard
    gather / experts / reduce-scatter it differs only by RCCL rounding the four ranks' bf16 partial sums in a
    message-size-dependent order (max |d| 0.016 on the reduced rows, check_moe_overlap.py).

    The side-stream collectives call RCCL directly: pass-sized messages fit aiter's custom all-gather /
    reduce-scatter, whose CU-resident kernels slowed the concurrent experts by more than they hid (32K extend
    1615 vs 1566 ms serialized); RCCL concurrent with a GEMM hides 70-85% of its time."""
    world = get_tp_group().world_size
    main, comm = torch.cuda.current_stream(), _sp_comm_stream()
    shard = inputs[0].shape[0]
    split = _moe_first_pass_rows(shard)
    first = [_group_gather(t[:split], split * world) for t in inputs]
    comm.wait_stream(main)
    with torch.cuda.stream(comm):
        for t in inputs:
            t.record_stream(comm)
        second = [_rccl_gather(t[split:], (shard - split) * world) for t in inputs]
        second_ready = comm.record_event()
    out1, ready1 = _side_reduce_scatter(run_experts(*first), main, comm)
    main.wait_event(second_ready)
    for t in second:
        t.record_stream(main)
    out2, ready2 = _side_reduce_scatter(run_experts(*second), main, comm)
    main.wait_event(ready1)
    out1.record_stream(main)
    out1 = out1 if finish is None else finish(out1)
    result = out1.new_empty((shard, out1.shape[1]))
    result[:split].copy_(out1)
    main.wait_event(ready2)
    out2.record_stream(main)
    result[split:].copy_(out2 if finish is None else finish(out2))
    return result


def ulysses_enabled() -> bool:
    return _PREFILL_SP and _PREFILL_ULYSSES and get_tp_group().world_size > 1


@torch.no_grad()
def prepare_ulysses(attention, fp8: bool) -> None:
    """Gather the TP shards of the QKV/gate and O projections once after loading (rank-major head order); with
    ``fp8`` keep only their FP8 copies + channel scales (``fp8_prefill_gemm``), half the bytes of the BF16 gather."""
    group = get_tp_group()
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
    group = get_tp_group()
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
    out = F.pad(out, (0, 0, 0, sp.padded - out.shape[0])).contiguous()
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
    # [rows, world, q|k|v|gate] -> [world, rows, q|k|v] to send, [world, rows, gate] kept (one copy each).
    send = qkv[:, :, :qkv_width].transpose(0, 1).contiguous()
    gate = qkv[:, :, qkv_width:].transpose(0, 1).contiguous()
    recv = _all_to_all(send, group).view(world * rows, qkv_width)
    q, k, v = recv[: sp.num_tokens].split(
        [attention.q_size, attention.kv_size, attention.kv_size], dim=-1
    )
    q, k = attention.qk_norm_rope(positions, q, k)
    out = attention.attn(q, k, v, forward_batch)
    # Padded rows enter the norm-gate as zeros and leave as zeros, as when they were padded after it.
    out = F.pad(out, (0, 0, 0, sp.padded - out.shape[0])).contiguous()
    heads = _all_to_all(out, group).view(world, rows, -1)
    return _o_projection(attention, heads, gate)


def _ulysses_chunk_plan(
    attention, rows: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """(QKV-side, output-side) row ranges of the pipelined Ulysses shard, or None when it runs unchunked. FP8
    projections only: aiter's CK FP8 GEMM runs one kernel config for every M >= 1024 (no tuned rows for these
    shapes), so chunk rows are bit-identical to whole-shard rows, while hipBLASLt picks its BF16 algorithm per M.
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
    out = F.pad(out, (0, 0, 0, sp.padded - out.shape[0])).contiguous()
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
        send = qkv[:, :, :qkv_width].transpose(0, 1).contiguous()
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
                gate[:, first - a : last - a].copy_(
                    qkv[first - lo : last - lo, :, qkv_width:].transpose(0, 1)
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
