"""CPU checks for PPU Kimi-K3 MoE compatibility paths."""

import sys
from types import SimpleNamespace
from unittest.mock import ANY, patch, sentinel

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.layers.moe.moe_runner import acext as acext_module
from sglang.srt.layers.moe.moe_runner import deep_gemm as deep_gemm_module
from sglang.srt.layers.moe.moe_runner import ppu_deepgemm_moe as ppu_moe_module
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.deep_gemm import (
    DeepGemmMoeQuantInfo,
    DeepGemmRunnerCore,
    DeepGemmRunnerInput,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput


def _w4a16_runner():
    runner = object.__new__(DeepGemmRunnerCore)
    runner.config = MoeRunnerConfig(
        activation="situ",
        gemm1_alpha=4.0,
        gemm1_clamp_limit=25.0,
        top_k=8,
    )
    runner.swiglu_limit = None
    return runner


def _mxfp4_w4a16_quant_info():
    return DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 128, 32), dtype=torch.uint8),
        w2_weight=torch.empty((2, 64, 32), dtype=torch.uint8),
        w13_scale=torch.empty((2, 2, 128), dtype=torch.uint8),
        w2_scale=torch.empty((2, 1, 128), dtype=torch.uint8),
        use_mxfp4_w4a16=True,
    )


def _legacy_w4a16_quant_info(variant):
    scale_dtype = torch.uint8 if variant == "mxfp4_valu" else torch.bfloat16
    return DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 4, 256), dtype=torch.int32),
        w2_weight=torch.empty((2, 4, 128), dtype=torch.int32),
        w13_scale=torch.empty((2, 2, 128), dtype=scale_dtype),
        w2_scale=torch.empty((2, 2, 64), dtype=scale_dtype),
        use_mxfp4_w4a16=variant == "mxfp4_valu",
        use_int4_w4a16=variant == "int4",
    )


def test_acext_situ_falls_back_before_importing_acext():
    sentinel = object()
    runner_config = SimpleNamespace(activation="situ")
    with (
        patch.object(acext_module, "is_ppu", return_value=True),
        patch.object(acext_module.logger, "info_once", create=True),
        patch.object(
            acext_module,
            "fused_experts_none_to_triton",
            return_value=sentinel,
        ) as fallback,
    ):
        result = acext_module.fused_experts_none_to_acext(
            "dispatch", "quant", runner_config
        )

    assert result is sentinel
    fallback.assert_called_once_with("dispatch", "quant", runner_config)


@pytest.mark.parametrize("variant", ["mxfp4_mma", "mxfp4_valu", "int4"])
def test_w4a16_reuses_deepep_w4a16_runner_paths(variant):
    runner = _w4a16_runner()
    quant_info = (
        _mxfp4_w4a16_quant_info()
        if variant == "mxfp4_mma"
        else _legacy_w4a16_quant_info(variant)
    )
    contiguous_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((1, 64), dtype=torch.bfloat16),
        hidden_states_scale=None,
        use_masked_gemm=False,
    )
    masked_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((2, 1, 64), dtype=torch.bfloat16),
        hidden_states_scale=None,
        use_masked_gemm=True,
    )

    with (
        patch.object(deep_gemm_module, "SGLANG_PROFILE_NVTX", False),
        patch.object(
            runner,
            "_run_int4_contiguous_gemm",
            return_value=sentinel.contiguous_output,
        ) as contiguous,
        patch.object(
            runner,
            "_run_masked_int4_gemm",
            return_value=sentinel.masked_output,
        ) as masked,
    ):
        contiguous_output = runner.run(contiguous_input, quant_info, {})
        masked_output = runner.run(masked_input, quant_info, {})

    assert contiguous_output.hidden_states is sentinel.contiguous_output
    assert masked_output.hidden_states is sentinel.masked_output
    contiguous.assert_called_once_with(contiguous_input, quant_info, {})
    masked.assert_called_once_with(masked_input, quant_info, {})


def _run_w4a16_tp_path(use_tp_fused, variant):
    hidden_states = torch.empty((2, 64), dtype=torch.bfloat16)
    if variant == "mxfp4_mma":
        w1 = torch.empty((2, 128, 32), dtype=torch.uint8)
        w2 = torch.empty((2, 64, 32), dtype=torch.uint8)
        w1_scale = torch.empty((2, 2, 128), dtype=torch.uint8)
        w2_scale = torch.empty((2, 1, 128), dtype=torch.uint8)
    else:
        w1 = torch.empty((2, 4, 256), dtype=torch.int32)
        w2 = torch.empty((2, 4, 128), dtype=torch.int32)
        scale_dtype = torch.bfloat16 if variant == "int4" else torch.uint8
        w1_scale = torch.empty((2, 2, 128), dtype=scale_dtype)
        w2_scale = torch.empty((2, 2, 64), dtype=scale_dtype)
    topk_ids = torch.tensor([[0], [1]], dtype=torch.int32)
    topk_weights = torch.ones((2, 1), dtype=torch.float32)
    m_rows = torch.tensor([1, 1], dtype=torch.int32)
    inv_perm = torch.tensor([0, 1], dtype=torch.int32)
    expert_ids = torch.tensor([0, 1], dtype=torch.int32)

    with (
        patch.object(
            ppu_moe_module.envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED,
            "get",
            return_value=use_tp_fused,
        ),
        patch.object(
            ppu_moe_module,
            "grouped_gemm_nt_bf16i4bf16_fused",
            return_value=(m_rows, inv_perm, expert_ids),
        ) as fused_gemm,
        patch.object(
            ppu_moe_module,
            "grouped_gemm_nt_bf16i4bf16_nopad",
        ) as nopad_gemm,
        patch.object(
            ppu_moe_module,
            "deepgemm_moe_permute",
            return_value=(hidden_states, None, expert_ids, inv_perm, m_rows),
        ) as permute,
        patch.object(
            ppu_moe_module,
            "situ_and_mul",
            return_value=torch.empty((2, 64), dtype=torch.bfloat16),
        ),
        patch.object(ppu_moe_module, "ep_gather"),
        patch.object(ppu_moe_module, "SGLANG_PROFILE_NVTX", False),
    ):
        ppu_moe_module.deep_moe_impl_fused(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int4_w4a16=variant == "int4",
            use_mxfp4_w4a16=variant != "int4",
            activation="situ",
            out_hidden_states=torch.empty_like(hidden_states),
        )

    return fused_gemm, nopad_gemm, permute, w2


def test_mxfp4_w4a16_mma_tp_fused_uses_fused_gemm1():
    fused_gemm, nopad_gemm, permute, w2 = _run_w4a16_tp_path(True, "mxfp4_mma")

    fused_gemm.assert_called_once()
    permute.assert_not_called()
    assert nopad_gemm.call_count == 1
    assert nopad_gemm.call_args.args[1] is w2


def test_mxfp4_w4a16_valu_tp_fused_uses_fused_gemm1():
    fused_gemm, nopad_gemm, permute, w2 = _run_w4a16_tp_path(True, "mxfp4_valu")

    fused_gemm.assert_called_once()
    permute.assert_not_called()
    assert nopad_gemm.call_count == 1
    assert nopad_gemm.call_args.args[1] is w2


def test_int4_w4a16_tp_fused_uses_fused_gemm1():
    fused_gemm, nopad_gemm, permute, w2 = _run_w4a16_tp_path(True, "int4")

    fused_gemm.assert_called_once()
    permute.assert_not_called()
    assert nopad_gemm.call_count == 1
    assert nopad_gemm.call_args.args[1] is w2


def test_mxfp4_w4a16_mma_requires_tp_fused_for_fused_gemm1():
    fused_gemm, nopad_gemm, permute, w2 = _run_w4a16_tp_path(False, "mxfp4_mma")

    fused_gemm.assert_not_called()
    permute.assert_called_once()
    assert nopad_gemm.call_count == 2
    assert nopad_gemm.call_args_list[0].args[1] is not w2
    assert nopad_gemm.call_args_list[1].args[1] is w2


@pytest.mark.parametrize("use_tp_fused", [False, True])
@pytest.mark.parametrize("activation", ["situ", "silu"])
def test_w4a4_tp_quantization_ignores_import_time_sm(
    monkeypatch, use_tp_fused, activation
):
    monkeypatch.delattr(ppu_moe_module, "downcast_to_mxfp4", raising=False)
    monkeypatch.delattr(ppu_moe_module, "silu_and_mul_post_quant_mxfp4", raising=False)
    hidden_states = torch.empty((2, 64), dtype=torch.bfloat16)
    input_quant = torch.empty((2, 32), dtype=torch.uint8)
    input_scale = torch.empty((2, 1), dtype=torch.uint16)
    down_quant = torch.empty_like(input_quant)
    down_scale = torch.empty_like(input_scale)
    down_input = torch.empty_like(hidden_states)
    output = torch.empty_like(hidden_states)
    w1 = torch.empty((2, 128, 32), dtype=torch.uint8)
    w2 = torch.empty((2, 64, 32), dtype=torch.uint8)
    m_rows = torch.tensor([1, 1], dtype=torch.int32)
    inv_perm = torch.tensor([0, 1], dtype=torch.int32)
    expert_ids = torch.tensor([0, 1], dtype=torch.int32)

    with (
        patch.object(
            ppu_moe_module.envs.SGLANG_SAIL_DEEPGEMM_MOE_TP_FUSED,
            "get",
            return_value=use_tp_fused,
        ),
        patch(
            "sglang.srt.layers.quantization.ppu_mxfp4_utils.downcast_to_mxfp4",
            side_effect=[(input_quant, input_scale), (down_quant, down_scale)],
        ) as downcast,
        patch(
            "sglang.kernels.ops.elementwise.silu_mul_quant.silu_and_mul_post_quant_mxfp4",
            return_value=(down_quant, down_scale),
        ) as silu_quant,
        patch.object(
            ppu_moe_module,
            "grouped_gemm_nt_f4f4bf16_fused",
            return_value=(m_rows, inv_perm, expert_ids),
        ) as fused_gemm,
        patch.object(ppu_moe_module, "grouped_gemm_nt_f4f4bf16_nopad") as nopad_gemm,
        patch.object(
            ppu_moe_module,
            "deepgemm_moe_permute",
            return_value=(input_quant, input_scale, expert_ids, inv_perm, m_rows),
        ) as permute,
        patch.object(ppu_moe_module, "situ_and_mul", return_value=down_input) as situ,
        patch.object(ppu_moe_module, "ep_gather") as gather,
        patch.object(ppu_moe_module, "SGLANG_PROFILE_NVTX", False),
    ):
        ppu_moe_module.deep_moe_impl_fused(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=torch.empty((2, 2, 128), dtype=torch.uint8),
            w2_scale=torch.empty((2, 1, 128), dtype=torch.uint8),
            topk_weights=torch.ones((2, 1), dtype=torch.float32),
            topk_ids=expert_ids.view(2, 1),
            use_mxfp4=True,
            block_shape=[0, 16],
            activation=activation,
            gemm1_alpha=4.0 if activation == "situ" else None,
            gemm1_limit=25.0 if activation == "situ" else None,
            out_hidden_states=output,
        )

    gather.assert_called_once()
    assert gather.call_args.kwargs["output_tensor"] is output
    assert gather.call_args.kwargs["input_tensor"].shape == output.shape
    assert downcast.call_args_list[0].args[0] is hidden_states
    assert downcast.call_args_list[0].kwargs == {
        "axis": 1,
        "enable_fused_moe": use_tp_fused,
    }
    if activation == "situ":
        assert downcast.call_count == 2
        assert downcast.call_args.args[0] is down_input
        assert downcast.call_args.kwargs == {"axis": 1}
        situ.assert_called_once_with(ANY, None, 4.0, 25.0)
        silu_quant.assert_not_called()
    else:
        downcast.assert_called_once()
        silu_quant.assert_called_once_with(ANY, swiglu_limit=None)
        situ.assert_not_called()
    if use_tp_fused:
        fused_gemm.assert_called_once()
        assert fused_gemm.call_args.args[0] is input_quant
        assert fused_gemm.call_args.args[1] is input_scale
        permute.assert_not_called()
        assert nopad_gemm.call_count == 1
    else:
        fused_gemm.assert_not_called()
        permute.assert_called_once()
        assert permute.call_args.kwargs["aq"] is input_quant
        assert permute.call_args.kwargs["aq_scale"] is input_scale
        assert nopad_gemm.call_count == 2
    assert nopad_gemm.call_args.args[0] is down_quant
    assert nopad_gemm.call_args.args[1] is down_scale
    assert nopad_gemm.call_args.args[2] is w2


@pytest.mark.parametrize("variant", ["mxfp4_mma", "mxfp4_valu", "int4"])
def test_w4a16_deepep_ll_uses_masked_layout(variant):
    masked_m = torch.tensor([1, 2], dtype=torch.int32)
    dispatch_output = DeepEPLLDispatchOutput(
        hidden_states=torch.empty((2, 2, 64), dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_ids=torch.tensor([[0], [1], [1]], dtype=torch.int64),
        topk_weights=torch.ones((3, 1), dtype=torch.float32),
        masked_m=masked_m,
        expected_m=2,
    )
    running_state = {}
    quant_info = (
        _mxfp4_w4a16_quant_info()
        if variant == "mxfp4_mma"
        else _legacy_w4a16_quant_info(variant)
    )

    runner_input = deep_gemm_module.pre_permute_deepep_ll_to_deep_gemm(
        dispatch_output,
        quant_info,
        SimpleNamespace(),
        running_state,
    )

    assert runner_input.use_masked_gemm
    assert runner_input.hidden_states.shape == (2, 2, 64)
    assert runner_input.masked_m is masked_m
    assert runner_input.expected_m == 2

    runner = _w4a16_runner()
    masked_output = torch.empty((2, 2, 64), dtype=torch.bfloat16)
    with (
        patch.object(deep_gemm_module, "SGLANG_PROFILE_NVTX", False),
        patch.object(
            runner, "_run_masked_int4_gemm", return_value=masked_output
        ) as masked,
        patch.object(runner, "_run_int4_contiguous_gemm") as contiguous,
    ):
        runner.run(runner_input, quant_info, running_state)

    masked.assert_called_once_with(runner_input, quant_info, running_state)
    contiguous.assert_not_called()


def test_w4a16_deepep_uses_k3_situ():
    runner = _w4a16_runner()
    with patch("sglang.kernels.ops.activation.situ_and_mul") as situ_and_mul:
        runner._apply_situ_and_mul(
            sentinel.gateup_output,
            sentinel.down_input,
        )

    situ_and_mul.assert_called_once_with(
        sentinel.gateup_output,
        sentinel.down_input,
        4.0,
        25.0,
    )


def test_w4a16_deepep_ll_uses_masked_k3_situ():
    runner = _w4a16_runner()
    with patch("sglang.kernels.ops.activation.situ_and_mul_masked") as situ_and_mul:
        runner._apply_situ_and_mul(
            sentinel.gateup_output,
            sentinel.down_input,
            sentinel.masked_m,
            expected_m=3,
        )

    situ_and_mul.assert_called_once_with(
        sentinel.gateup_output,
        sentinel.down_input,
        sentinel.masked_m,
        4.0,
        25.0,
        8,
        3,
    )


@pytest.mark.parametrize("variant", ["mxfp4_mma", "mxfp4_valu", "int4"])
def test_w4a16_deepep_ll_passes_mask_to_activation(variant):
    runner = _w4a16_runner()
    quant_info = (
        _mxfp4_w4a16_quant_info()
        if variant == "mxfp4_mma"
        else _legacy_w4a16_quant_info(variant)
    )
    masked_m = torch.tensor([0, 2], dtype=torch.int32)
    runner_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((2, 3, 64), dtype=torch.bfloat16),
        hidden_states_scale=None,
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=2,
    )

    with (
        patch.object(deep_gemm_module, "_is_ppu", True),
        patch.object(deep_gemm_module, "SGLANG_PROFILE_NVTX", False),
        patch.object(
            deep_gemm_module.deep_gemm_wrapper,
            "grouped_gemm_nt_bf16i4bf16_masked",
        ) as grouped_gemm,
        patch.object(deep_gemm_module, "dispose_tensor"),
        patch("sglang.kernels.ops.activation.situ_and_mul_masked") as apply_situ,
    ):
        output = runner.run(
            runner_input,
            quant_info,
            {"hidden_states_device": torch.device("cpu")},
        ).hidden_states

    assert output.shape == (2, 3, 64)
    assert output.dtype == torch.bfloat16
    assert grouped_gemm.call_count == 2
    gemm1, gemm2 = grouped_gemm.call_args_list
    assert gemm1.args[0] is runner_input.hidden_states
    assert gemm1.args[1][0] is quant_info.w13_weight
    assert gemm1.args[1][1] is quant_info.w13_scale
    assert gemm1.args[2].shape == (2, 3, 128)
    assert gemm2.args[0].shape == (2, 3, 64)
    assert gemm2.args[0].dtype == torch.bfloat16
    assert gemm2.args[1][0] is quant_info.w2_weight
    assert gemm2.args[1][1] is quant_info.w2_scale
    assert gemm2.args[2] is output
    assert gemm1.args[3] is masked_m
    assert gemm2.args[3] is masked_m
    apply_situ.assert_called_once_with(ANY, ANY, masked_m, 4.0, 25.0, 8, 2)


@pytest.mark.parametrize("variant", ["mxfp4_mma", "mxfp4_valu", "int4"])
@pytest.mark.parametrize("num_tokens", [0, 4])
def test_w4a16_contiguous_preserves_layout(variant, num_tokens):
    runner = _w4a16_runner()
    quant_info = (
        _mxfp4_w4a16_quant_info()
        if variant == "mxfp4_mma"
        else _legacy_w4a16_quant_info(variant)
    )
    runner_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((num_tokens, 64), dtype=torch.bfloat16),
        hidden_states_scale=None,
        use_masked_gemm=False,
        m_indices=torch.zeros((num_tokens,), dtype=torch.int32),
    )
    m_rows = torch.tensor([num_tokens, 0], dtype=torch.int32)

    with (
        patch.object(deep_gemm_module, "_is_ppu", True),
        patch.object(deep_gemm_module, "SGLANG_PROFILE_NVTX", False),
        patch.object(
            deep_gemm_module.deep_gemm_wrapper,
            "grouped_gemm_nt_bf16i4bf16_nopad",
        ) as grouped_gemm,
        patch.object(deep_gemm_module, "dispose_tensor"),
        patch("sglang.kernels.ops.activation.situ_and_mul") as apply_situ,
    ):
        output = runner.run(
            runner_input,
            quant_info,
            {
                "all_tokens": num_tokens,
                "hidden_states_device": torch.device("cpu"),
                "hidden_states_shape": (num_tokens, 64),
                "m_rows": m_rows,
            },
        ).hidden_states

    assert output.shape == (num_tokens, 64)
    assert output.dtype == torch.bfloat16
    if num_tokens == 0:
        grouped_gemm.assert_not_called()
        apply_situ.assert_not_called()
        return

    assert grouped_gemm.call_count == 2
    gemm1, gemm2 = grouped_gemm.call_args_list
    assert gemm1.args[0] is runner_input.hidden_states
    assert gemm1.args[1][0] is quant_info.w13_weight
    assert gemm1.args[1][1] is quant_info.w13_scale
    assert gemm1.args[2].shape == (num_tokens, 128)
    assert gemm2.args[0].shape == (num_tokens, 64)
    assert gemm2.args[0].dtype == torch.bfloat16
    assert gemm2.args[1][0] is quant_info.w2_weight
    assert gemm2.args[1][1] is quant_info.w2_scale
    assert gemm2.args[2] is output
    for call in (gemm1, gemm2):
        assert call.args[3] is runner_input.m_indices
        assert call.args[4] is m_rows
    apply_situ.assert_called_once_with(ANY, ANY, 4.0, 25.0)


def test_w4a4_deepep_uses_k3_situ_mxfp4_post_quant():
    runner = _w4a16_runner()
    runner_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((2, 32), dtype=torch.uint8),
        hidden_states_scale=torch.empty((2, 2), dtype=torch.uint16),
        use_masked_gemm=False,
        m_indices=torch.zeros((2,), dtype=torch.int32),
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 128, 32), dtype=torch.uint8),
        w2_weight=torch.empty((2, 64, 32), dtype=torch.uint8),
        w13_scale=torch.empty((2, 2, 128), dtype=torch.uint8),
        w2_scale=torch.empty((2, 1, 128), dtype=torch.uint8),
        use_mxfp4=True,
    )
    with (
        patch.object(deep_gemm_module, "SGLANG_PROFILE_NVTX", False),
        patch.dict(
            sys.modules,
            {"deep_gemm": SimpleNamespace(preprocess_mxfp4_scales=lambda x: x)},
        ),
        patch.object(
            deep_gemm_module.deep_gemm_wrapper,
            "grouped_gemm_nt_f4f4bf16_nopad",
        ) as grouped_gemm,
        patch.object(deep_gemm_module, "dispose_tensor"),
        patch(
            "sglang.kernels.ops.moe.situ_and_mul_post_quant_mxfp4",
        ) as activation,
    ):
        output = runner.run(
            runner_input,
            quant_info,
            {
                "all_tokens": 2,
                "hidden_states_device": torch.device("cpu"),
                "hidden_states_dtype": torch.uint8,
                "hidden_states_shape": runner_input.hidden_states.shape,
            },
        ).hidden_states

    assert output.shape == (2, 64)
    activation.assert_called_once_with(ANY, ANY, ANY, 4.0, 25.0)
    assert activation.call_args.args[1].shape == (2, 32)
    assert activation.call_args.args[2].shape == (1, 2)
    assert grouped_gemm.call_args_list[1].args[0][1].shape == (2, 1)


def test_w4a4_deepep_ll_uses_masked_k3_situ_mxfp4_post_quant():
    runner = _w4a16_runner()
    masked_m = torch.tensor([1, 2], dtype=torch.int32)
    runner_input = DeepGemmRunnerInput(
        hidden_states=torch.empty((2, 3, 32), dtype=torch.uint8),
        hidden_states_scale=torch.empty((2, 1, 3), dtype=torch.uint16),
        use_masked_gemm=True,
        masked_m=masked_m,
        expected_m=2,
    )
    quant_info = DeepGemmMoeQuantInfo(
        w13_weight=torch.empty((2, 128, 32), dtype=torch.uint8),
        w2_weight=torch.empty((2, 64, 32), dtype=torch.uint8),
        w13_scale=torch.empty((2, 2, 128), dtype=torch.uint8),
        w2_scale=torch.empty((2, 1, 128), dtype=torch.uint8),
        use_mxfp4=True,
    )
    with (
        patch.object(deep_gemm_module, "SGLANG_PROFILE_NVTX", False),
        patch.object(
            deep_gemm_module.deep_gemm_wrapper,
            "grouped_gemm_nt_f4f4bf16_masked",
        ) as grouped_gemm,
        patch.object(deep_gemm_module, "dispose_tensor"),
        patch(
            "sglang.kernels.ops.moe.situ_and_mul_masked_post_quant_mxfp4",
        ) as activation,
    ):
        output = runner.run(
            runner_input,
            quant_info,
            {"hidden_states_device": torch.device("cpu")},
        ).hidden_states

    assert output.shape == (2, 3, 64)
    activation.assert_called_once_with(ANY, ANY, ANY, masked_m, 4.0, 25.0, 8, 2)
    assert activation.call_args.args[1].shape == (2, 3, 32)
    assert activation.call_args.args[2].shape == (2, 1, 3)
    assert grouped_gemm.call_args_list[1].args[0][1].shape == (2, 3, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
