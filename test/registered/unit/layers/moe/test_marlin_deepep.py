"""Initialization contracts for the resolved Marlin + DeepEP runner."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe import utils
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import validate_deepep_marlin
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DispatcherOutputDtype, MoeRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def args(**kwargs):
    values = dict(
        enable_lora=False,
        enable_eplb=False,
        elastic_ep_backend=None,
        enable_single_batch_overlap=False,
        enable_two_batch_overlap=False,
        init_expert_location="trivial",
        deepep_dispatcher_output_dtype="auto",
        enable_waterfill=False,
        ep_num_redundant_experts=0,
    )
    return SimpleNamespace(**(values | kwargs))


@pytest.mark.parametrize(
    "option,value",
    [
        ("enable_lora", True),
        ("enable_eplb", True),
        ("elastic_ep_backend", "mooncake"),
        ("enable_single_batch_overlap", True),
        ("enable_two_batch_overlap", True),
        ("init_expert_location", "random"),
        ("deepep_dispatcher_output_dtype", "fp8"),
    ],
)
def test_reject_unsupported_options(option, value):
    config = MoeRunnerConfig(
        runner_backend=MoeRunnerBackend.MARLIN, params_dtype=torch.bfloat16
    )
    with (
        patch(
            "sglang.srt.runtime_context.get_server_args",
            return_value=args(**{option: value}),
        ),
        patch("sglang.srt.utils.is_cuda", return_value=True),
    ):
        with pytest.raises(ValueError):
            validate_deepep_marlin(config)


def test_validate_resolved_runner():
    config = MoeRunnerConfig(
        runner_backend=MoeRunnerBackend.MARLIN, params_dtype=torch.bfloat16
    )
    with (
        patch("sglang.srt.runtime_context.get_server_args", return_value=args()),
        patch("sglang.srt.utils.is_cuda", return_value=True),
    ):
        validate_deepep_marlin(config)
        for changes in (
            dict(params_dtype=torch.float16),
            dict(num_fused_shared_experts=1),
            dict(apply_router_weight_on_input=True),
            dict(runner_backend=MoeRunnerBackend.EXPERIMENTAL_SGL_MARLIN),
        ):
            with pytest.raises(ValueError):
                validate_deepep_marlin(replace(config, **changes))


@pytest.mark.parametrize("requested", ["auto", "bf16", "fp8", "nvfp4"])
def test_dtype_uses_actual_runner(requested):
    dispatcher = SimpleNamespace(
        runner_backend=MoeRunnerBackend.MARLIN,
        quant_config={"input_global_scale": torch.ones(1)},
    )
    with (
        patch.object(
            utils,
            "get_exec",
            return_value=SimpleNamespace(
                moe=SimpleNamespace(deepep_dispatcher_output_dtype=requested)
            ),
        ),
        patch.object(utils, "get_server_args", return_value=None),
    ):
        if requested in {"auto", "bf16"}:
            assert (
                utils.get_deepep_output_dtype(dispatcher) == DispatcherOutputDtype.BF16
            )
        else:
            with pytest.raises(ValueError, match="bf16"):
                utils.get_deepep_output_dtype(dispatcher)


def test_normal_combine_without_deepgemm():
    impl = object.__new__(deepep._DeepEPDispatcherImplNormal)
    impl.runner_backend = MoeRunnerBackend.MARLIN
    impl.async_finish = False
    x = torch.randn(3, 8)
    with patch.object(deepep.deep_gemm_wrapper, "ENABLE_JIT_DEEPGEMM", False):
        output, event = impl.combine_a(x, None, None)
    assert output is x
    assert event is None


def test_auto_quantization_resolves_marlin():
    from sglang.srt.layers.moe.moe_runner import runner
    from sglang.srt.layers.moe.moe_runner.marlin import fused_experts_deepep_to_marlin
    from sglang.srt.layers.moe.utils import MoeA2ABackend

    config = MoeRunnerConfig(params_dtype=torch.bfloat16)
    with (
        patch.object(runner, "get_moe_a2a_backend", return_value=MoeA2ABackend.DEEPEP),
        patch.object(
            runner, "get_moe_runner_backend", return_value=MoeRunnerBackend.AUTO
        ),
        patch("sglang.srt.runtime_context.get_server_args", return_value=args()),
        patch("sglang.srt.utils.is_cuda", return_value=True),
    ):
        instance = runner.MoeRunner(MoeRunnerBackend.MARLIN, config)
        assert config.runner_backend == MoeRunnerBackend.MARLIN
        assert instance.fused_func is fused_experts_deepep_to_marlin
        with pytest.raises(ValueError, match="overlap"):
            instance.set_overlap_args(object(), {})


@pytest.mark.parametrize("mode", ["normal", "low_latency"])
@pytest.mark.parametrize("method_name", ["mxfp4", "mxfp4_marlin"])
def test_mxfp4_preserves_deepep_combine_format_and_width(mode, method_name):
    from unittest.mock import Mock

    from sglang.srt.layers.moe.token_dispatcher import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from sglang.srt.layers.quantization.mxfp4_marlin_moe import Mxfp4MarlinMoEMethod

    ids = torch.tensor([[0, 1]])
    weights = torch.tensor([[0.3, 0.7]])
    if mode == "normal":
        dispatch = DeepEPNormalDispatchOutput(
            torch.zeros(5, 128), None, ids, weights, []
        )
        combine_type = DeepEPNormalCombineInput
    else:
        dispatch = DeepEPLLDispatchOutput(
            torch.zeros(2, 5, 128), None, ids, weights, torch.tensor([1, 3]), 5
        )
        combine_type = DeepEPLLCombineInput
    layer = SimpleNamespace(
        w13_weight=torch.empty(2, 16, 32),
        w2_weight=torch.empty(2, 8, 64),
        w13_weight_scale=torch.empty(2, 1, 256),
        w2_weight_scale=torch.empty(2, 1, 128),
        dispatcher=SimpleNamespace(),
    )
    runner = Mock()

    def compute(dispatched, quant_info):
        assert dispatched.hidden_states.shape[-1] == 256
        return combine_type(torch.ones_like(dispatched.hidden_states), ids, weights)

    runner.run.side_effect = compute
    if method_name == "mxfp4":
        method = object.__new__(Mxfp4MoEMethod)
        method.hidden_size, method.hidden_pad, method.runner = 256, 128, runner
        output = method._apply_marlin(layer, dispatch)
    else:
        method = object.__new__(Mxfp4MarlinMoEMethod)
        method.runner = runner
        output = method.apply(layer, dispatch)
    assert isinstance(output, combine_type)
    assert output.hidden_states.shape == dispatch.hidden_states.shape
    assert output.hidden_states.is_contiguous()
    assert output.topk_ids is ids
    assert output.topk_weights is weights


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
