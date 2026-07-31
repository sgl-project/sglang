"""Regression tests for UE8M0 activation scales in Triton FP8 MoE."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    fused_experts_none_to_triton,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    ("scale_fmt", "use_scale_ue8m0"),
    [(None, False), ("e4m3", False), ("ue8m0", True)],
)
def test_fp8_config_reads_checkpoint_scale_format(scale_fmt, use_scale_ue8m0):
    config = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "scale_fmt": scale_fmt,
    }

    quant_config = Fp8Config.from_config(config)

    assert quant_config.use_scale_ue8m0 is use_scale_ue8m0


@pytest.mark.parametrize("use_scale_ue8m0", [False, True])
def test_fp8_moe_quant_info_carries_checkpoint_scale_format(use_scale_ue8m0):
    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        weight_block_size=[128, 128],
        use_scale_ue8m0=use_scale_ue8m0,
    )
    method = Fp8MoEMethod(quant_config)
    layer = SimpleNamespace(
        w13_weight=torch.empty(1),
        w2_weight=torch.empty(1),
        w13_weight_scale_inv=torch.empty(1),
        w2_weight_scale_inv=torch.empty(1),
        w13_input_scale=None,
        w2_input_scale=None,
    )

    quant_info = method.get_triton_quant_info(layer)

    assert quant_info.use_scale_ue8m0 is use_scale_ue8m0


def test_ue8m0_is_ineligible_for_continuous_scale_quant_once():
    import sglang.srt.models.deepseek_v2 as deepseek_v2
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
    from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
    from sglang.srt.layers.quantization.fp8_utils import (
        cutlass_w8a8_block_fp8_linear_with_fallback,
    )

    gate_up_quant_method = object.__new__(Fp8LinearMethod)
    gate_up_quant_method.w8a8_block_fp8_linear = (
        cutlass_w8a8_block_fp8_linear_with_fallback
    )
    gate_up = SimpleNamespace(
        quant_method=gate_up_quant_method,
        weight=torch.empty((128, 128), dtype=torch.float8_e4m3fn),
    )
    experts = object.__new__(FusedMoE)
    torch.nn.Module.__init__(experts)
    experts.quant_method = Fp8MoEMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
            use_scale_ue8m0=True,
        )
    )
    moe = SimpleNamespace(
        _enable_a2a_moe=False,
        _fuse_shared_experts_inside_sbo=False,
        num_fused_shared_experts=0,
        shared_experts=SimpleNamespace(gate_up_proj=gate_up),
        shared_experts_is_fp8=True,
        shared_experts_weight_block_size=[128, 128],
        experts=experts,
    )

    with (
        patch.object(
            deepseek_v2.envs.SGLANG_OPT_MOE_QUANT_ONCE,
            "get",
            return_value=True,
        ),
        patch.object(deepseek_v2, "_is_cuda", True),
    ):
        eligible, reason = deepseek_v2.DeepseekV2MoE._compute_moe_quant_once_enabled(
            moe
        )

    assert not eligible
    assert reason == "UE8M0 activation scales unsupported by quant-once"


@pytest.mark.parametrize("use_scale_ue8m0", [False, True])
def test_ue8m0_does_not_reuse_continuous_prequant_input(use_scale_ue8m0):
    prequant_q = torch.empty(1, dtype=torch.float8_e4m3fn)
    prequant_scale = torch.empty(1, dtype=torch.float32)
    dispatch_output = SimpleNamespace(
        hidden_states=torch.empty(1),
        hidden_states_pre_quant=(prequant_q, prequant_scale),
        topk_output=(torch.empty(1), torch.empty(1), None),
    )
    quant_info = TritonMoeQuantInfo(
        w13_weight=torch.empty(1),
        w2_weight=torch.empty(1),
        use_scale_ue8m0=use_scale_ue8m0,
    )

    with patch.object(
        fused_moe, "fused_experts", return_value=torch.empty(1)
    ) as experts:
        fused_experts_none_to_triton(dispatch_output, quant_info, SimpleNamespace())

    if use_scale_ue8m0:
        assert experts.call_args.kwargs["a1_q"] is None
        assert experts.call_args.kwargs["a1_scale"] is None
    else:
        assert experts.call_args.kwargs["a1_q"] is prequant_q
        assert experts.call_args.kwargs["a1_scale"] is prequant_scale


@pytest.mark.parametrize("use_scale_ue8m0", [False, True])
def test_triton_moe_applies_scale_format_to_both_gemms(use_scale_ue8m0):
    hidden_states = torch.zeros((2, 4), dtype=torch.bfloat16)
    w1 = torch.zeros((1, 8, 4), dtype=torch.bfloat16)
    w2 = torch.zeros((1, 4, 4), dtype=torch.bfloat16)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    topk_ids = torch.zeros((2, 1), dtype=torch.int32)

    def _zero_kernel_output(*args, **kwargs):
        args[3].zero_()

    with (
        patch.object(
            fused_moe,
            "invoke_fused_moe_kernel",
            side_effect=_zero_kernel_output,
        ) as invoke,
        patch.object(
            fused_moe,
            "get_server_args",
            return_value=SimpleNamespace(enable_fused_moe_sum_all_reduce=False),
        ),
        patch.multiple(
            fused_moe,
            _is_cuda=False,
            _is_hip=False,
            _is_xpu=False,
            _is_musa=False,
            _has_vllm_ops=False,
        ),
    ):
        fused_moe._fused_moe_kernel_sequence(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            sorted_token_ids=torch.arange(2, dtype=torch.int32),
            expert_ids=torch.zeros(2, dtype=torch.int32),
            num_tokens_post_padded=torch.tensor([2], dtype=torch.int32),
            config={"BLOCK_SIZE_M": 1},
            down_config=None,
            down_moe_use_tma=False,
            b1=None,
            b2=None,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            w1_scale=torch.ones(1),
            w2_scale=torch.ones(1),
            w1_zp=None,
            w2_zp=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=[128, 128],
            use_scale_ue8m0=use_scale_ue8m0,
            activation="silu",
            is_gated=True,
            no_combine=True,
            inplace=False,
            apply_router_weight_on_input=False,
            routed_scaling_factor=None,
            gemm1_alpha=None,
            gemm1_limit=None,
            filter_expert=False,
        )

    assert invoke.call_count == 2
    assert [call.kwargs["scale_ue8m0"] for call in invoke.call_args_list] == [
        use_scale_ue8m0,
        use_scale_ue8m0,
    ]
