"""gfx1250 EP exchange for the Muse sequence-parallel prefill MoE through Mori dispatch/combine.

Replaces the SP path's all-gather of every rank's MoE input + local experts over all tokens + reduce-scatter with:
dispatch (each token goes only to the ranks hosting its top-k experts) -> this rank's A8W4 experts with the Muse
per-expert norm + weighted sum (``run_gfx1250_a8w4``) -> combine (sum of the ranks' partials back to the source
token). Off by default (``SGLANG_ROCM_MUSE_GFX1250_MORI_EP=1``). Mori's topology probe needs the server started with
``LD_PRELOAD=<rocm sdk>/libamd_smi.so.27`` (importing torch first leaves amd-smi with 0 sockets -> Mori exits).
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SGLANG_ROCM_MUSE_GFX1250_MORI_EP", "0") == "1"
MAX_TOKENS = int(os.environ.get("SGLANG_ROCM_MUSE_MORI_EP_MAX_TOKENS", "2048"))
_op = None


def ready() -> bool:
    return _op is not None


def init(group, hidden: int, experts_per_rank: int, topk: int) -> None:
    """Collective over the TP group; call once, outside graph capture."""
    global _op
    if _op is not None:
        return
    import mori

    name = "muse_mori_ep"
    torch._C._distributed_c10d._register_process_group(name, group.device_group)
    mori.shmem.shmem_torch_process_group_init(name)
    cfg = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=group.rank_in_group,
        world_size=group.world_size,
        hidden_dim=hidden,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=2,
        max_num_inp_token_per_rank=MAX_TOKENS,
        num_experts_per_rank=experts_per_rank,
        num_experts_per_token=topk,
    )
    _op = mori.ops.EpDispatchCombineOp(cfg)
    logger.info(
        "Muse gfx1250 Mori EP prefill exchange ready (rank %d/%d, max %d tokens/rank)",
        group.rank_in_group,
        group.world_size,
        MAX_TOKENS,
    )


def dispatch(x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor):
    """-> (recv tokens [cap, H] bf16, recv weights [cap, topk] f32, recv ids [cap, topk] i32, recv count [1] i32)."""
    out, w, _scale, rids, n = _op.dispatch(x, weights, None, ids)
    return out, w, rids, n


def combine(
    y: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, rows: int
) -> torch.Tensor:
    out = _op.combine(y, weights, ids)
    if isinstance(out, (tuple, list)):
        out = out[0]  # (combined tokens, combined weights)
    return out[:rows]
