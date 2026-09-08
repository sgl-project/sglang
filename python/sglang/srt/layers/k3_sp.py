"""Row-parallel LatentMoE tail for Kimi-K3 (SGLANG_K3_TAIL_SHARD).

The LatentMoE tail after the MoE all-reduce is REPLICATED: every rank runs the
latent RMSNorm, the `routed_expert_up_proj` GEMM and the add3 over all T tokens
and keeps only the same full result. The work is token-local -- no cross-token
reduction anywhere in it -- so each rank can compute its own T/w rows and the
ranks can exchange what they uniquely computed:

    AR(buf) -> norm(full T) -> up_proj(full T) -> add3(full T)          today
    AR(buf) -> norm(T/w)    -> up_proj(T/w)    -> add3(T/w) -> AG       here

That drops 7/8 of a replicated GEMM and 7/8 of the norm and the add3, and pays
one bf16 all-gather for it. At T=19456 (latent 3584, hidden 7168), per layer:

    replicated up_proj GEMM   1.404 ms      sharded   0.165 ms
    bf16 all-gather of [T, H]               0.865 ms

so the GEMM alone more than pays for the all-gather (-0.374 ms/layer) before
the norm and the add3 also drop to 1/8.

ACCURACY: exact. The all-reduce is untouched and the all-gather is unquantized
bf16, so no value passes through a codec the replicated path does not already
use, and the sharded math is the same math on a row subset.

The restructure is layer-LOCAL -- every layer still enters and leaves on the
full batch -- so the bank, the attention-side aggregation, dspark capture and
the PP wire are unaffected.

Gated on token count because below the crossover the all-gather's fixed cost
exceeds the saving (the tail measures 0.525x at T=32, 1.574x at T=16384), so
chunked prefill takes this path and decode keeps the replicated one.

Needs no split collectives and so no aiter quickreduce rebuild: the only
collective it adds is torch.distributed's all_gather_into_tensor on the TP
group.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def tail_shard_enabled() -> bool:
    return envs.SGLANG_K3_TAIL_SHARD.get()


_tail_shard_armed: Optional[bool] = None


def tail_shard_eligible(num_tokens: int) -> bool:
    global _tail_shard_armed
    if not tail_shard_enabled():
        return False
    if num_tokens < envs.SGLANG_K3_TAIL_SHARD_MIN_TOKENS.get():
        return False
    from sglang.srt.distributed import get_tp_group

    ok = num_tokens % get_tp_group().world_size == 0
    #  warning, not info: sglang's default level swallows info, and a beacon
    #  nobody can grep makes an arm that silently did nothing look like a
    #  measured null. Emitted from inside the gate, so it proves the path was
    #  TAKEN rather than that the env var was set.
    if ok and _tail_shard_armed is not True:
        _tail_shard_armed = True
        logger.warning(
            "K3 tail shard armed: tokens=%d world=%d",
            num_tokens,
            get_tp_group().world_size,
        )
    return ok


def tail_shard_rows(num_tokens: int) -> slice:
    """This rank's contiguous token slice of a [T, H] activation."""
    from sglang.srt.distributed import get_tp_group

    g = get_tp_group()
    rows = num_tokens // g.world_size
    return slice(g.rank_in_group * rows, (g.rank_in_group + 1) * rows)


def tail_shard_all_gather(shard: torch.Tensor) -> torch.Tensor:
    """[T/w, H] -> [T, H] over the TP group, bf16 in and bf16 out.

    all_gather_into_tensor concatenates along dim 0 in rank order, which for a
    row-major [T, H] is exactly the token order the shards were cut in, so no
    permutation is needed on either side.
    """
    import torch.distributed as dist

    from sglang.srt.distributed import get_tp_group

    g = get_tp_group()
    out = torch.empty(
        shard.shape[0] * g.world_size,
        shard.shape[1],
        dtype=shard.dtype,
        device=shard.device,
    )
    dist.all_gather_into_tensor(out, shard.contiguous(), group=g.device_group)
    return out
