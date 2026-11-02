"""CPU coverage for Humming routing metadata, scale plumbing, and workspace reuse."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()
pytest.importorskip("humming")

from sglang.srt.layers.moe import fused_moe_triton  # noqa: E402
from sglang.srt.layers.moe.moe_runner import humming  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


@pytest.fixture
def runner_module(monkeypatch):
    monkeypatch.setattr(humming.HummingMethod, "may_process_input", Mock())
    return humming


def make_runner(module):
    runner = object.__new__(module.HummingRunnerCore)
    runner.num_experts = 3
    runner.activation = "silu"
    runner.swiglu_limit = None
    runner.config = SimpleNamespace(params_dtype=torch.bfloat16)
    runner.layer = SimpleNamespace(
        hidden_size=256,
        intermediate_size_per_partition=128,
        params_dtype=torch.bfloat16,
        humming_metas={},
    )
    for sublayer in ("w13", "w2"):
        runner.layer.humming_metas[sublayer] = SimpleNamespace(
            a_dtype=module.dtypes.float8e4m3,
            c_dtype=module.dtypes.bfloat16,
            input_quant_mode=module.InputQuantizationMode.DynamicGroup,
            input_scale_name=f"{sublayer}_input_scale",
            input_scale_2_name=f"{sublayer}_input_scale_2",
            check_device=Mock(),
        )
    return runner


@pytest.mark.parametrize("down_block_m", [40, 16])
def test_indexed_alignment_matches_each_gemm(runner_module, monkeypatch, down_block_m):
    runner = make_runner(runner_module)
    topk_ids = torch.tensor([[2, 0], [1, -1], [0, 2]], dtype=torch.int32)
    configs = {"compute_config_str": '{"gemm_type": "indexed"}'}
    for sublayer, block_m in (("w13", 40), ("w2", down_block_m)):
        tuning = [
            [0, 6, {"block_shape": [block_m, 128, 128]}],
            [6, 100, {"block_shape": [64, 128, 128]}],
        ]
        configs[f"{sublayer}_tuning_config"] = tuning
        configs[f"{sublayer}_tuning_config_str"] = json.dumps(tuning)
    runner.get_humming_gemm_configs = Mock(return_value=configs)
    alignments = {}

    def align(**kwargs):
        assert kwargs["topk_ids"] is topk_ids
        assert kwargs["ignore_invalid_expert"]
        assert kwargs["num_experts"] == 3
        result = tuple(torch.tensor([kwargs["block_size"]]) for _ in range(3))
        alignments[kwargs["block_size"]] = result
        return result

    align_mock = Mock(side_effect=align)
    monkeypatch.setattr(
        fused_moe_triton,
        "moe_align_block_size",
        align_mock,
        raising=False,
    )
    gate_up_kwargs, down_kwargs = runner._prepare_indexed_gemm_kwargs(topk_ids)
    assert align_mock.call_count == (1 if down_block_m == 40 else 2)
    for kwargs, sublayer, block_m, top_k in (
        (gate_up_kwargs, "w13", 40, 2),
        (down_kwargs, "w2", down_block_m, 1),
    ):
        assert kwargs["valid_shape_m"] == 6
        assert kwargs["top_k"] == top_k
        assert kwargs["tuning_config"] == configs[f"{sublayer}_tuning_config_str"]
        for name, expected in zip(
            ("sorted_ids", "expert_ids", "num_tokens_padded"), alignments[block_m]
        ):
            assert kwargs[name] is expected


@pytest.mark.parametrize(
    "mode_name,static_scale_name,first_scale_name,second_scale_name",
    [
        ("DynamicToken", None, "token", None),
        ("DynamicGroup", None, "group", None),
        ("StaticTensor", "w2_input_scale", "token", None),
        ("StaticTensorDynamicGroup", "w2_input_scale_2", "group", "token"),
        ("DynamicGroupToken", None, "group", "token"),
    ],
)
def test_input_scale_levels_are_preserved(
    runner_module, mode_name, static_scale_name, first_scale_name, second_scale_name
):
    runner = make_runner(runner_module)
    runner.layer.humming_metas["w2"].input_quant_mode = getattr(
        runner_module.InputQuantizationMode, mode_name
    )
    runner.layer.w2_input_scale = torch.tensor([0.25])
    runner.layer.w2_input_scale_2 = torch.tensor([0.5])
    inputs = torch.ones(3, 128, dtype=torch.bfloat16)
    quantized = torch.ones(3, 128, dtype=torch.int8)
    scales = {None: None, "group": torch.ones(3, 1), "token": torch.full((3, 1), 2.0)}
    process_input = runner_module.HummingMethod.may_process_input
    process_input.return_value = quantized, scales["group"], scales["token"]

    result, first_scale, second_scale = runner._prepare_input(inputs, "w2", quantized)
    assert result is quantized
    assert first_scale is scales[first_scale_name]
    assert second_scale is scales[second_scale_name]
    assert process_input.call_args.kwargs["m_major_scale"] is False
    assert process_input.call_args.kwargs["token_scales"] is (
        getattr(runner.layer, static_scale_name) if static_scale_name else None
    )


def test_deepep_input_is_not_requantized(runner_module):
    runner = make_runner(runner_module)
    inputs = torch.ones(2, 256).to(torch.float8_e4m3fn)
    scales = torch.ones(2, 2)
    result, first_scale, second_scale = runner._prepare_input(
        inputs, "w13", input_scale=scales
    )
    assert result is inputs and first_scale is scales and second_scale is None
    runner_module.HummingMethod.may_process_input.assert_not_called()


@pytest.mark.parametrize(
    "gate_up_dtype,down_dtype,gate_up_buffer,down_buffer",
    [
        ("bfloat16", "float4e2m1", None, (64, torch.uint8)),
        ("float8e5m2", "bfloat16", (256, torch.float8_e5m2), None),
        ("float4e0m3", "float8e4m3", (128, torch.uint8), (128, torch.float8_e4m3fn)),
    ],
)
def test_workspace_uses_each_sublayers_dtype(
    runner_module, gate_up_dtype, down_dtype, gate_up_buffer, down_buffer
):
    runner = make_runner(runner_module)
    for sublayer, dtype in (("w13", gate_up_dtype), ("w2", down_dtype)):
        meta = runner.layer.humming_metas[sublayer]
        meta.a_dtype = getattr(runner_module.dtypes, dtype)
        if dtype == "bfloat16":
            meta.input_quant_mode = runner_module.InputQuantizationMode.Disabled
    inputs = torch.ones(3, 256, dtype=torch.bfloat16)
    routes = torch.zeros(3, 2, dtype=torch.int32)
    buffers = runner.prepare_buffers(
        inputs, routes, runner_module.HummingGemmType.INDEXED
    )
    for name, expected in (
        ("quanted_gate_up_input", gate_up_buffer),
        ("quanted_down_input", down_buffer),
    ):
        if expected is None:
            assert name not in buffers
            continue
        width, dtype = expected
        assert buffers[name].shape[-1] == width
        assert buffers[name].dtype == dtype
    assert (
        buffers["activation_output"].data_ptr() != buffers["gate_up_output"].data_ptr()
    )
    if "quanted_down_input" in buffers:
        assert (
            buffers["activation_output"].data_ptr()
            != buffers["quanted_down_input"].data_ptr()
        )


def test_hadamard_matches_each_sublayers_weight_transform(runner_module):
    runner = make_runner(runner_module)
    runner.layer.weight_schemas = {
        "w13": SimpleNamespace(hadamard_block_size=32),
        "w2": SimpleNamespace(hadamard_block_size=64),
    }
    inputs = torch.ones(3, 128, dtype=torch.bfloat16)
    process_input = runner_module.HummingMethod.may_process_input
    process_input.return_value = inputs, torch.ones(3, 1), None
    for sublayer, block_size in (("w13", 32), ("w2", 64)):
        runner._prepare_input(inputs, sublayer)
        assert process_input.call_args.kwargs["hadamard_block_size"] == block_size

    with pytest.raises(ValueError, match="Hadamard inputs require BF16 DeepEP"):
        runner._prepare_input(inputs, "w13", input_scale=torch.ones(3, 1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
