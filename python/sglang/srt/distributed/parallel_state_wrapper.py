from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True, kw_only=True)
class ParallelState:
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
    dcp_size: int
    gpu_id: int

    @staticmethod
    def trivial(**overrides: Optional[int]) -> "ParallelState":
        kwargs: dict[str, Optional[int]] = dict(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            dp_rank=0,
            dp_size=1,
            attn_tp_rank=0,
            attn_tp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            attn_dp_rank=0,
            attn_dp_size=1,
            moe_ep_rank=0,
            moe_ep_size=1,
            moe_dp_rank=0,
            moe_dp_size=1,
            dcp_size=1,
            gpu_id=0,
        )
        kwargs.update(overrides)
        return ParallelState(**kwargs)
