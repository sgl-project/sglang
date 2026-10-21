"""Sequence-parallel Muse Spark prefill on ROCm, with Ulysses all-to-all attention.

Plain EXTEND batches of at least ``SP_MIN_TOKENS`` tokens run the norms, residuals, router and the replicated
pooler / unpooler on 1/TP of the tokens (each rank keeps a contiguous shard, padded to a multiple of TP).
Attention and the experts still see every token:

* Ulysses (``SGLANG_MUSE_PREFILL_ULYSSES``): each rank projects its own token shard for every head with the
  gathered QKV/gate weight, an all-to-all hands every rank all tokens of its own heads, attention and the output
  norm/gate run as under TP, a second all-to-all returns all heads of the rank's tokens and the full O projection
  finishes them without a reduction. Per 16K tokens and layer that moves ~95 MB per rank instead of the
  all-gather + reduce-scatter's ~450 MB. Without it the shard is all-gathered before attention and the
  attention output reduce-scattered.
* Experts: the shard's router outputs and expert inputs are all-gathered (as MXFP4 bytes when the a4w4 experts
  run, since per-row quantization commutes with the gather) and the expert output is reduce-scattered.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F

from sglang.srt.distributed import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

SP_MIN_TOKENS = 2048

_PREFILL_SP = envs.SGLANG_MUSE_PREFILL_SP.get()
_PREFILL_ULYSSES = envs.SGLANG_MUSE_PREFILL_ULYSSES.get()


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


def ulysses_enabled() -> bool:
    return _PREFILL_SP and _PREFILL_ULYSSES and get_tp_group().world_size > 1


@torch.no_grad()
def prepare_ulysses(attention) -> None:
    """Gather the TP shards of the QKV/gate and O projections once after loading (rank-major head order)."""
    group = get_tp_group()
    attention.ulysses_qkv_weight = group.all_gather(
        attention.qkv_proj.weight.data.contiguous(), dim=0
    )
    attention.ulysses_o_weight = group.all_gather(
        attention.o_proj.weight.data.contiguous(), dim=1
    ).contiguous()


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
) -> torch.Tensor:
    """Attention block for this rank's token shard ``x`` [padded / world, hidden]; returns its output rows."""
    group = get_tp_group()
    world, rows = group.world_size, x.shape[0]
    # [rows, hidden] shard -> all-to-all -> [world * rows, q|k|v|gate of this rank's heads]
    qkv = F.linear(x, attention.ulysses_qkv_weight).view(rows, world, -1)
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
    heads = _all_to_all(out, group).view(world, shard_rows, -1).transpose(0, 1)
    return F.linear(heads.reshape(shard_rows, -1), attention.ulysses_o_weight)
