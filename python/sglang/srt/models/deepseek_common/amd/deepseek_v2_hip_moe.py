"""ROCm glue of DeepseekV2MoE: the fused all-reduce + mHC post of the DeepSeek-V4 decode
batches."""

from __future__ import annotations

import torch

from sglang.kernels.ops.communication.all_reduce_mhc_hip import all_reduce_mhc_post
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.layers.moe.mhc_post_fusion import current_mhc_post_fusion
from sglang.srt.layers.moe.utils import post_experts_all_reduce
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    ALL_REDUCE_MHC_MAX_ROWS,
)


def all_reduce_output(moe, hidden_states: torch.Tensor) -> torch.Tensor:
    """Post-experts all-reduce of moe (a DeepseekV2MoE); the eagerly built 1-8 row mHC
    states take the fused all-reduce + post kernel."""
    mhc = current_mhc_post_fusion()
    if (
        mhc is not None
        and not moe._shared_expert_tp1
        and not mhc.overlap_only
        and mhc.post is not None
        and 1 <= hidden_states.shape[0] <= ALL_REDUCE_MHC_MAX_ROWS
    ):
        mhc.output = all_reduce_mhc_post(
            hidden_states, mhc.residual, mhc.post, mhc.comb, get_tp_group().ca_comm
        )
        # the decoder reads mhc.output; the return keeps the DP wrapper's tensor contract
        return hidden_states
    if mhc is not None:
        mhc.start_stats_before_all_reduce()
    return post_experts_all_reduce(hidden_states)
