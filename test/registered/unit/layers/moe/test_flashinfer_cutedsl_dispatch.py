import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.flashinfer_cutedsl as cutedsl_runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import (
    BypassedTopKOutput,
    StandardTopKOutput,
    TopK,
    TopKConfig,
    TopKOutputFormat,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _cutedsl_quant_info(wrapper, *, per_token=False, quant_mode="w4a4"):
    return SimpleNamespace(
        wrapper=wrapper,
        quant_mode=quant_mode,
        use_per_token_activation=per_token,
        a1_scale=torch.tensor(1.0),
        a2_scale=torch.tensor(1.0),
        w13_weight=object(),
        w13_weight_sf=object(),
        w1_alpha=object(),
        w2_weight=object(),
        w2_weight_sf=object(),
        w2_alpha=object(),
    )


def test_flashinfer_prefill_returns_standard_combine_input():
    dispatch_output = StandardDispatchOutput(
        hidden_states=torch.empty(2, 16, dtype=torch.bfloat16),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.empty(2, 1),
            topk_ids=torch.zeros(2, 1, dtype=torch.int32),
            router_logits=None,
        ),
    )
    expected_output = torch.empty(2, 16, dtype=torch.bfloat16)
    wrapper = Mock()
    wrapper.run.return_value = expected_output
    quant_info = _cutedsl_quant_info(wrapper)
    runner_config = SimpleNamespace(activation="silu")

    with patch(
        "sglang.srt.layers.quantization.fp4_utils.fp4_quantize",
        return_value=(
            torch.empty(2, 8, dtype=torch.uint8),
            torch.empty(2, 1, dtype=torch.float8_e4m3fn),
        ),
    ):
        result = cutedsl_runner.fused_experts_flashinfer_to_flashinfer_cutedsl_fp4(
            dispatch_output, quant_info, runner_config
        )

    assert isinstance(result, StandardCombineInput)
    assert result.hidden_states is expected_output
    wrapper.run.assert_called_once()


@pytest.mark.parametrize("num_tokens", [0, 3])
@pytest.mark.parametrize("per_token", [False, True])
def test_standard_cutedsl_forwards_fp4_dispatch(num_tokens, per_token):
    hidden = torch.empty(num_tokens, 16, dtype=torch.uint8)
    block_scales = torch.empty(num_tokens, 2, dtype=torch.uint8)
    token_scales = torch.ones(num_tokens, dtype=torch.float32) if per_token else None
    topk = StandardTopKOutput(
        topk_weights=torch.ones(num_tokens, 1),
        topk_ids=torch.zeros(num_tokens, 1, dtype=torch.int32),
        router_logits=None,
    )
    dispatch = StandardDispatchOutput(
        hidden, block_scales, topk, hidden_states_per_token_scale=token_scales
    )
    expected = torch.empty(num_tokens, 32, dtype=torch.bfloat16)
    wrapper = Mock(run=Mock(return_value=expected))
    # Neither static nor per-token quantization may run on packed FP4 input.
    with (
        patch("sglang.srt.layers.quantization.fp4_utils.fp4_quantize") as quantize,
        patch.dict(sys.modules, {"flashinfer": SimpleNamespace()}),
    ):
        result = cutedsl_runner.fused_experts_none_to_flashinfer_cutedsl_fp4(
            dispatch,
            _cutedsl_quant_info(wrapper, per_token=per_token),
            MoeRunnerConfig(),
        )
    quantize.assert_not_called()
    call = wrapper.run.call_args.kwargs
    assert call["x"] is hidden
    assert call["x_sf"].data_ptr() == block_scales.data_ptr()
    assert call["x_sf"].shape == (num_tokens, 2)
    assert call["x_sf"].dtype == torch.float8_e4m3fn
    assert call["per_token_scale"] is token_scales
    assert call["token_selected_experts"] is topk.topk_ids
    assert call["token_final_scales"] is topk.topk_weights
    assert result.hidden_states is expected


def test_standard_cutedsl_w4a16_keeps_bf16_activations():
    hidden = torch.empty(3, 32, dtype=torch.bfloat16)
    dispatch = StandardDispatchOutput(
        hidden,
        None,
        StandardTopKOutput(
            torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.int32), None
        ),
    )
    wrapper = Mock()
    cutedsl_runner.fused_experts_none_to_flashinfer_cutedsl_fp4(
        dispatch,
        _cutedsl_quant_info(wrapper, quant_mode="w4a16"),
        MoeRunnerConfig(),
    )
    call = wrapper.run.call_args.kwargs
    assert call["x"] is hidden
    assert call["x_sf"] is None
    assert call["per_token_scale"] is None
    assert call["fc2_input_scale"] is None


@pytest.mark.parametrize("fp4_dispatch", [False, True])
@pytest.mark.parametrize("explicit_standard", [False, True])
@pytest.mark.parametrize(
    "backend",
    [MoeRunnerBackend.FLASHINFER_TRTLLM, MoeRunnerBackend.FLASHINFER_TRTLLM_ROUTED],
)
def test_empty_topk_matches_fp4_dispatch_routing(
    fp4_dispatch, explicit_standard, backend
):
    import sglang.srt.layers.moe.topk as topk_module

    topk = TopK.__new__(TopK)
    torch.nn.Module.__init__(topk)
    topk.topk_config = TopKConfig(
        top_k=2,
        output_format=TopKOutputFormat.STANDARD if explicit_standard else None,
    )
    topk.is_fp4_experts = True
    topk.enable_waterfill = False
    topk.waterfill_balancer = None
    with (
        patch.object(
            topk_module,
            "should_use_flashinfer_moe_fp4_allgather",
            return_value=fp4_dispatch,
        ),
        patch.object(topk_module, "get_moe_runner_backend", return_value=backend),
        patch.object(topk_module, "get_tp_group"),
        patch.object(topk_module, "is_allocation_symmetric", return_value=False),
        patch.object(topk_module, "use_symmetric_memory", return_value=nullcontext()),
    ):
        empty = topk.empty_topk_output(torch.device("cpu"))
    if fp4_dispatch and not explicit_standard and backend.is_flashinfer_trtllm():
        assert isinstance(empty, BypassedTopKOutput)
        assert empty.topk_config is topk.topk_config
        assert empty.router_logits.shape == (0, 0)
        assert empty.router_logits.dtype == torch.float32
    else:
        assert isinstance(empty, StandardTopKOutput)
        assert empty.topk_ids.shape == (0, 2)


@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16])
def test_standard_trtllm_forwards_fp4_dispatch(routed, per_token, output_dtype):
    import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm as trtllm_runner

    hidden = torch.empty(3, 16, dtype=torch.uint8)
    block_scales = torch.empty(3, 2, dtype=torch.uint8)
    token_scales = torch.ones(3, dtype=torch.float32) if per_token else None
    router_logits = torch.empty(3, 2, dtype=torch.float32)
    topk = (
        StandardTopKOutput(torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.int32), None)
        if routed
        else BypassedTopKOutput(hidden, router_logits, TopKConfig(top_k=1))
    )
    dispatch = StandardDispatchOutput(
        hidden, block_scales, topk, hidden_states_per_token_scale=token_scales
    )
    quant_info = trtllm_runner.FlashInferTrtllmFp4MoeQuantInfo(
        w13_weight=hidden,
        w2_weight=hidden,
        w13_weight_scale=block_scales,
        w2_weight_scale=block_scales,
        g1_scale_c=torch.ones(1),
        g1_alphas=torch.ones(1),
        g2_alphas=torch.ones(1),
        w13_input_scale_quant=torch.ones(1),
        global_num_experts=2,
        local_expert_offset=0,
        local_num_experts=2,
        intermediate_size_per_partition=32,
        routing_method_type=0,
        use_per_token_activation=per_token,
    )
    raw_kernel = Mock(side_effect=lambda **kwargs: [kwargs["output"]])
    routed_kernel = Mock(side_effect=lambda **kwargs: [kwargs["output"]])
    packed_topk = torch.zeros(3, 1, dtype=torch.int32)
    with (
        patch.dict(
            sys.modules,
            {
                "flashinfer.fused_moe": SimpleNamespace(
                    trtllm_fp4_block_scale_moe=raw_kernel,
                    trtllm_fp4_block_scale_routed_moe=routed_kernel,
                )
            },
        ),
        patch.object(trtllm_runner, "quantize_hidden_states_fp4") as quantize,
        patch.object(trtllm_runner, "get_activation_type", return_value=3),
        patch.object(trtllm_runner, "get_tp_group"),
        patch.object(trtllm_runner, "is_allocation_symmetric", return_value=False),
        patch.object(trtllm_runner, "use_symmetric_memory", return_value=nullcontext()),
        patch.object(trtllm_runner, "trtllm_moe_enable_pdl", return_value=False),
        patch.object(
            trtllm_runner,
            "_get_packed_topk_ids_for_flashinfer_routed",
            return_value=packed_topk,
        ) as pack_topk,
        patch(
            "sglang.srt.runtime_context.get_forward",
            return_value=SimpleNamespace(moe_output_buffer=None),
        ),
    ):
        result = trtllm_runner.fused_experts_none_to_flashinfer_trtllm_fp4(
            dispatch,
            quant_info,
            MoeRunnerConfig(params_dtype=output_dtype),
            use_routed_topk=routed,
        )
    quantize.assert_not_called()
    selected, other = (
        (routed_kernel, raw_kernel) if routed else (raw_kernel, routed_kernel)
    )
    selected.assert_called_once()
    other.assert_not_called()
    call = selected.call_args.kwargs
    assert call["hidden_states"] is hidden
    assert call["hidden_states_scale"].data_ptr() == block_scales.data_ptr()
    assert call["hidden_states_scale"].shape == (3, 2)
    assert call["hidden_states_scale"].dtype == torch.float8_e4m3fn
    assert call["per_token_scale"] is token_scales
    assert result.hidden_states.shape == (3, 32)
    assert result.hidden_states.dtype == output_dtype
    if routed:
        pack_topk.assert_called_once_with(topk)
        assert call["topk_ids"] is packed_topk
    else:
        pack_topk.assert_not_called()
        assert call["routing_logits"] is router_logits


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
