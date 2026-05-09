from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, kw_only=True)
class ParallelState:
    __slots__ = (
        "tp_rank",
        "tp_size",
        "pp_rank",
        "pp_size",
        "dp_rank",
        "dp_size",
        "attn_tp_rank",
        "attn_tp_size",
        "attn_cp_rank",
        "attn_cp_size",
        "attn_dp_rank",
        "attn_dp_size",
        "moe_ep_rank",
        "moe_ep_size",
        "moe_dp_rank",
        "moe_dp_size",
        "gpu_id",
    )

    tp_rank: int
    tp_size: int
    pp_rank: int
    pp_size: int
    dp_rank: Optional[int]
    dp_size: int
    attn_tp_rank: int
    attn_tp_size: int
    attn_cp_rank: int
    attn_cp_size: int
    attn_dp_rank: int
    attn_dp_size: int
    moe_ep_rank: int
    moe_ep_size: int
    moe_dp_rank: Optional[int]
    moe_dp_size: int
    gpu_id: int
