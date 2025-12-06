from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.base import (
    MoeRunnerConfig,
    MoeRunnerCore,
    register_fused_func,
)
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerInput,
    TritonRunnerOutput,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import is_cuda

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size  # noqa: F401

_is_cuda = is_cuda()


class AlphaMoeRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
    ) -> TritonRunnerOutput:
        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        sorted_token_ids = runner_input.sorted_token_ids
        expert_ids = runner_input.expert_ids
        num_tokens_post_padded = runner_input.num_tokens_post_padded

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        b13 = quant_info.b13
        b2 = quant_info.b2
        a13_scale = quant_info.a13_scale
        a2_scale = quant_info.a2_scale
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        w13_zp = quant_info.w13_zp
        w2_zp = quant_info.w2_zp
        block_shape = quant_info.block_shape
        per_channel_quant = quant_info.per_channel_quant
        use_fp8_w8a8 = quant_info.use_fp8_w8a8
        use_int8_w8a8 = quant_info.use_int8_w8a8
        use_int8_w8a16 = quant_info.use_int8_w8a16
        use_int4_w4a16 = quant_info.use_int4_w4a16

        activation = self.config.activation
        no_combine = self.config.no_combine
        inplace = self.config.inplace
        gemm1_alpha = self.config.gemm1_alpha
        gemm1_limit = self.config.gemm1_clamp_limit
        routed_scaling_factor = self.config.routed_scaling_factor
        apply_router_weight_on_input = self.config.apply_router_weight_on_input

        M = hidden_states.shape[0]
        E, N, _ = w13.shape
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )

        assert block_shape[0] == 128
        assert block_shape[1] == 128
        assert use_fp8_w8a8
        assert hidden_states.shape[1] % 128 == 0
        assert activation == "silu"

        out_hidden_states = alpha_moe_fused_moe(
            hidden_states,
            w13,
            w2,
            StandardTopKOutput(topk_weights, topk_ids, None),
            self.config,
            w13_scale,
            w2_scale,
        )

        return TritonRunnerOutput(
            hidden_states=out_hidden_states,
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.ALPHA_MOE


def alpha_moe_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output: StandardTopKOutput,
    moe_runner_config: MoeRunnerConfig,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    inplace=False,
):
    from alpha_moe_python.utils import get_best_config

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import moe_align_block_size
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )

    topk_weights, topk_ids, _ = topk_output
    E, N, _ = w1.shape
    M = hidden_states.shape[0]
    topk = topk_ids.shape[1]

    local_conf = get_best_config(os.getenv("ALPHA_MOE_CONFIG"), M)
    block_m = local_conf["block_m"]
    bn = local_conf["block_n"]
    wn = local_conf["warp_n"]
    stages = local_conf["stages"]
    A, A_scale = sglang_per_token_group_quant_fp8(hidden_states, 128)

    out = hidden_states.zero_() if inplace else torch.zeros_like(hidden_states)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, E
    )
    routed_scaling_factor = moe_runner_config.routed_scaling_factor or 1.0
    from alpha_moe_python.alpha_moe_ops import fused_moe_w8a8_up_down

    fused_moe_w8a8_up_down(
        A,
        A_scale,
        w1,
        w1_scale,
        w2,
        w2_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        out,
        topk,
        block_m,
        bn,
        wn,
        stages,
        routed_scaling_factor,
    )
    return out


@register_fused_func("none", "alpha_moe")
def fused_experts_none_to_alpha_moe(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    assert quant_info.block_shape[0] == 128
    assert quant_info.block_shape[1] == 128
    assert quant_info.use_fp8_w8a8
    assert dispatch_output.hidden_states.shape[1] % 128 == 0
    assert runner_config.activation == "silu"

    output = alpha_moe_fused_moe(
        hidden_states=dispatch_output.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=runner_config,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
    )

    return StandardCombineInput(
        hidden_states=output,
    )
