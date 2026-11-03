"""MXFP4 expert backend running AITER's a16w4 (bf16 activation x MXFP4 weight) MoE.

``Fp8Config`` selects this for FP4-expert checkpoints under
``--moe-runner-backend aiter``. The checkpoint's separated ``[gate; up]`` w13
rows are preshuffled and consumed with ``GateMode.SEPARATED``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import Module

from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput
    from sglang.srt.layers.moe.utils import RoutingMethodType

from .flydsl_moe_tp_w4a8_gfx942.a8w4_moe_tuned import tuned_moe
from .flydsl_moe_tp_w4a8_gfx942.test_a8w4_gemm1 import a8w4_gemm1, prep_w1_fast
from .flydsl_moe_tp_w4a8_gfx942.test_a8w4_gemm2 import a8w4_gemm2, prep_w2_fast
from aiter.fused_moe import moe_sorting
from aiter import ActivationType, QuantType, dtypes

# Fast fp4 decode of gemm1 / gemm2 (bit-exact vs 0). 1: W nibbles reordered
# and -0 -> +0 (drops the interleave / sign perms). 2: additionally stores the
# per-group scale delta vs the channel max and zeroes flushed groups (one
# v_perm LUT); W2's scale buffer then carries its channel max codes. W / scales
# are rewritten in place, so the aiter-runner path in ``apply`` needs 0.
A8W4_FAST_DEC = 2

# block_m -> (gemm1 kwargs, gemm2 kwargs). Both stages share the sorting block_m.
#  32: gemm1 pipelined (prefetch) + XCD swizzle; gemm2 skips all-padding sub-blocks.
#  64: same block_m as the tuned CSV; gemm1 dyn_m (32/64 rows per block at runtime),
#      gemm2 dyn_rows.
A8W4_E2E_CFGS = {
    32: (
        dict(tile_n=128, b_cache_mod=1, xcd_swizzle=1),
        dict(tile_n=128, b_cache_mod=2, xcd_swizzle=0, dyn_rows=True),
    ),
    64: (
        dict(tile_n=128, b_cache_mod=1, xcd_swizzle=1, dyn_m=True),
        dict(tile_n=128, b_cache_mod=2, xcd_swizzle=0, dyn_rows=True),
    ),
}

def a8w4_fused_moe(
    hidden_states, weights, topk_weights, topk_ids, *, block_m=32, fuse_quant=False
):
    """SiLU SEPARATED A8W4 MoE: sort -> fp8 row-quant(x) -> gemm1 (gate+up+SiLU,
    bf16 by sorted row) -> fp8 row-quant(inter) -> gemm2 (down + weighted atomic
    scatter into the zeroed moe_buf). Returns bf16 [tokens, model_dim].

    ``fuse_quant``: gemm1 quantizes in its epilogue (``q8s``: fp8 by sorted row,
    one f32 scale per (row, 128 cols)) and gemm2 takes the per-128-K-group A
    scales; drops the second row-quant kernel and the bf16 round trip."""
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import quantize_rows

    g1_cfg, g2_cfg = A8W4_E2E_CFGS[block_m]
    g1_cfg = dict(g1_cfg, fast_dec=weights.get("w1_fast_dec", 0))
    g2_cfg = dict(g2_cfg, fast_dec=weights.get("w2_fast_dec", 0))
    E, model_dim, _ = weights["w2"].shape
    inter_dim = weights["w1"].shape[1] // 2

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = (
        moe_sorting(topk_ids, topk_weights, E, model_dim, dtypes.bf16, block_m)[:5]
    )
    x_fp8, x_scale = quantize_rows(hidden_states)
    sorted_size = sorted_expert_ids.numel() * block_m
    if fuse_quant:
        assert g1_cfg["tile_n"] == 128, "q8s scale groups = gemm1 tile_n = 128"
        groups = inter_dim // 128
        buf = torch.empty(
            sorted_size * (inter_dim + groups * 4),
            dtype=torch.uint8,
            device=hidden_states.device,
        )
        a8w4_gemm1(
            x_fp8,
            x_scale,
            weights["w1"],
            weights["w1_scale"],
            weights["w1_exp"],
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            buf,
            num_experts=E,
            tile_m=block_m,
            epilog="q8s",
            inter_dim=inter_dim,
            topk=topk_ids.shape[1],
            **g1_cfg,
        )
        nd = sorted_size * inter_dim
        a2_fp8 = buf[:nd].view(torch.float8_e4m3fnuz).view(sorted_size, inter_dim)
        a2_scale = buf[nd:].view(torch.float32).view(sorted_size, groups)
        a8w4_gemm2(
            a2_fp8,
            a2_scale,
            weights["w2"],
            weights["w2_scale"],
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            num_experts=E,
            tile_m=block_m,
            a_group=128,
            **g2_cfg,
        )
        return moe_buf
    inter = torch.empty(
        (sorted_size, inter_dim),
        dtype=dtypes.bf16,
        device=hidden_states.device,
    )
    a8w4_gemm1(
        x_fp8,
        x_scale,
        weights["w1"],
        weights["w1_scale"],
        weights["w1_exp"],
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        inter,
        num_experts=E,
        tile_m=block_m,
        **g1_cfg,
    )
    a2_fp8, a2_scale = quantize_rows(inter, num_valid_ids)
    a8w4_gemm2(
        a2_fp8,
        a2_scale,
        weights["w2"],
        weights["w2_scale"],
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_experts=E,
        tile_m=block_m,
        **g2_cfg,
    )
    return moe_buf

class Mxfp4AiterA16W4MoEMethod:
    """MXFP4 (E8M0 scales) MoE via aiter fused_moe with bf16 activations."""

    def __init__(self, fp8_method, prefix: str):
        self._fp8 = fp8_method
        self.prefix = prefix

    def create_weights(
        self,
        layer: Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        with_bias: bool = False,
        **extra_weight_attrs,
    ):
        if with_bias:
            raise NotImplementedError(
                "Mxfp4AiterA16W4MoEMethod does not support MoE GEMM biases."
            )
        # shuffle_scale_a16w4 passes fp32 scales through untouched, so keep E8M0.
        self._fp8.create_weights(
            layer,
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            params_dtype,
            fp4_scale_dtype=torch.float8_e8m0fnu,
            **extra_weight_attrs,
        )

    def create_moe_runner(
        self, layer: Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        # The native A16W4/A8W4 SiLU epilogue does not apply an activation
        # clamp. A positive swiglu_limit can also override our separated
        # weight layout in the shared runner. Reject either config spelling
        # before runner setup rather than silently changing the model.
        if any(
            limit is not None and limit != 0
            for limit in (
                moe_runner_config.swiglu_limit,
                moe_runner_config.gemm1_clamp_limit,
            )
        ):
            raise ValueError(
                "Native MXFP4 A16W4/A8W4 does not support activation clamps; "
                "swiglu_limit and gemm1_clamp_limit must be unset or 0."
            )
        self.moe_runner_config = moe_runner_config
        self.runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)

    def get_moe_runner_backend(self, layer):
        return self.runner.runner_backend

    def get_moe_routing_method_type(
        self, layer: Module, requested_method: RoutingMethodType | None
    ) -> RoutingMethodType | None:
        return requested_method

    def process_weights_after_loading(self, layer: Module) -> None:
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        num_experts = layer.w13_weight.shape[0]
        fp4_dtype = torch.float4_e2m1fn_x2

        # gate_up=False keeps the [gate; up] row order that GateMode.SEPARATED reads.
        layer.w13_weight.data = shuffle_weight_a16w4(
            layer.w13_weight.data.view(fp4_dtype), 16, False
        )
        layer.w2_weight.data = shuffle_weight_a16w4(
            layer.w2_weight.data.view(fp4_dtype), 16, False
        )
        w13_scale = layer.w13_weight_scale_inv.data
        w2_scale = layer.w2_weight_scale_inv.data
        layer.w13_weight_scale_inv = torch.nn.Parameter(
            shuffle_scale_a16w4(
                w13_scale.view(-1, w13_scale.shape[-1]), num_experts, False
            ),
            requires_grad=False,
        )
        layer.w2_weight_scale_inv = torch.nn.Parameter(
            shuffle_scale_a16w4(
                w2_scale.view(-1, w2_scale.shape[-1]), num_experts, False
            ),
            requires_grad=False,
        )

        # aiter fused_moe reads these off w1/w2 to pick the preshuffled
        # SEPARATED a16w4 kernels; the .data swap above drops them.
        for weight in (layer.w13_weight, layer.w2_weight):
            weight.is_shuffled = True
            weight.gate_up_interleave = False

        from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import channel_exponents
        w1_u8 = layer.w13_weight.data.view(torch.uint8)
        w2_u8 = layer.w2_weight.data.view(torch.uint8)
        w1_scale_u8 = layer.w13_weight_scale_inv.data.view(torch.uint8).contiguous().view(-1)
        w2_scale_u8 = layer.w2_weight_scale_inv.data.view(torch.uint8).contiguous().view(-1)
        w1_exp = channel_exponents(w1_u8, w1_scale_u8)  # from the original scales

        fast_dec = A8W4_FAST_DEC
        if fast_dec:
            # Swap the layer storage to the rewritten bytes so the originals are freed.
            w1_u8, w1_scale_u8 = prep_w1_fast(w1_u8, w1_scale_u8, w1_exp, fast_dec)
            w2_u8, w2_scale_u8 = prep_w2_fast(w2_u8, w2_scale_u8, fast_dec)
            layer.w13_weight.data = w1_u8.view(fp4_dtype)
            layer.w2_weight.data = w2_u8.view(fp4_dtype)
            layer.w13_weight_scale_inv = torch.nn.Parameter(w1_scale_u8, requires_grad=False)
            layer.w2_weight_scale_inv = torch.nn.Parameter(w2_scale_u8, requires_grad=False)

        layer.all_weights = dict(
            w1=w1_u8,
            w2=w2_u8,
            w1_scale=w1_scale_u8,
            w2_scale=w2_scale_u8,
            w1_exp=w1_exp,
            w1_fast_dec=fast_dec,
            w2_fast_dec=fast_dec,
        )

        layer.dispatcher.set_quant_config(
            {"weight_dtype": layer.w13_weight.dtype, "use_mxfp8": False}
        )

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        if 1:
            from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
            from sglang.srt.layers.moe.topk import TopKOutputChecker            
            topk_output = dispatch_output.topk_output
            hidden_states = dispatch_output.hidden_states
            if layer.all_weights["w1_fast_dec"] == 2:
                # per-batch-size best of 2-kernel / fused single-kernel variants
                # (flydsl_moe_tp_w4a8_gfx942/configs/a8w4_moe_tuned.txt; untuned
                # shapes fall back to the 2-kernel bm64 config)
                output = tuned_moe(hidden_states, layer.all_weights,
                                   topk_output.topk_weights, topk_output.topk_ids)
            else:
                output = a8w4_fused_moe(hidden_states, layer.all_weights,
                                        topk_output.topk_weights, topk_output.topk_ids,
                                        block_m=64, fuse_quant=True)
            return StandardCombineInput(hidden_states=output)

        from aiter.ops.flydsl.moe_common import GateMode

        from sglang.srt.layers.moe.moe_runner.aiter import (
            AiterMoeQuantInfo,
            AiterQuantType,
        )

        quant_info = AiterMoeQuantInfo(
            w13_weight=layer.w13_weight,
            w2_weight=layer.w2_weight,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            expert_mask=layer.dispatcher.expert_mask_gpu,
            swiglu_limit=self.moe_runner_config.swiglu_limit or 0.0,
            fused_moe_kwargs={"gate_mode": GateMode.SEPARATED.value},
        )
        return self.runner.run(dispatch_output, quant_info)
