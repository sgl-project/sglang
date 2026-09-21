"""H200 W4AFP8 adapter checks; no distributed process group is needed here."""

# ruff: noqa: E402

import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

pytest.importorskip("flashinfer.fused_moe")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
    pytest.skip("Requires SM90", allow_module_level=True)

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as runner
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
from sglang.srt.layers.quantization.w4afp8 import W4AFp8Config, W4AFp8MoEMethod

# Python's stdlib also has a `test` package; load the existing reference by path.
_spec = importlib.util.spec_from_file_location(
    "w4afp8_reference",
    Path(__file__).resolve().parents[4] / "manual/quant/test_cutlass_w4a8_moe.py",
)
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)
pack_int4_values_to_int8 = _reference.pack_int4_values_to_int8


def make_layer(ep_size=1, ep_rank=0, hidden=6144, intermediate=None, experts=256):
    """Build both adapters from identical checkpoint-layout tensors."""
    intermediate = intermediate or (256 if ep_size == 1 else 2048)
    torch.manual_seed(42)
    local_e = experts // ep_size
    w1 = torch.randint(
        -8, 8, (local_e, 2 * intermediate, hidden), device="cuda", dtype=torch.int8
    )
    w2 = torch.randint(
        -8, 8, (local_e, hidden, intermediate), device="cuda", dtype=torch.int8
    )
    s1 = (
        torch.randn(
            local_e,
            2 * intermediate,
            hidden // 128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.005
    )
    s2 = (
        torch.randn(
            local_e, hidden, intermediate // 128, device="cuda", dtype=torch.bfloat16
        )
        * 0.005
    )
    a1 = torch.tensor([0.75], device="cuda")
    a2 = torch.tensor([1.25], device="cuda")
    methods = []
    config = MoeRunnerConfig(
        activation="silu", is_gated=True, routed_scaling_factor=1.5
    )
    for backend in (MoeRunnerBackend.CUTLASS, MoeRunnerBackend.FLASHINFER_CUTLASS):
        method = W4AFp8MoEMethod(W4AFp8Config())
        layer = torch.nn.Module()
        layer.moe_tp_size = 8 if ep_size == 1 else 1
        layer.moe_tp_rank = 3 if ep_size == 1 else 0
        layer.moe_ep_size, layer.moe_ep_rank = ep_size, ep_rank
        with (
            torch.device("cuda"),
            patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=backend,
            ),
            patch(
                "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
                return_value=MoeA2ABackend.NONE,
            ),
        ):
            method.create_weights(
                layer,
                local_e,
                hidden,
                intermediate,
                torch.bfloat16,
                weight_loader=lambda *a: None,
            )
        packed = pack_int4_values_to_int8(w1)
        scale = s1
        if method.use_flashinfer:
            packed = torch.cat(packed.chunk(2, dim=1)[::-1], dim=1)
            scale = torch.cat(scale.chunk(2, dim=1)[::-1], dim=1)
            assert not hasattr(method, "expert_offsets")
        layer.w13_weight.data.copy_(packed)
        layer.w2_weight.data.copy_(pack_int4_values_to_int8(w2))
        layer.w13_weight_scale_inv.data.copy_(scale)
        layer.w2_weight_scale_inv.data.copy_(s2)
        layer.w13_input_scale.data.fill_(a1.item())
        layer.w2_input_scale.data.fill_(a2.item())
        method.moe_runner_config = config
        method.process_weights_after_loading(layer)
        methods.append((method, layer))
    return methods, (w1, w2, s1, s2, a1, a2), config


def invoke(method, dispatch, config):
    with (
        patch.object(runner, "get_tp_group", return_value=None),
        patch.object(runner, "is_allocation_symmetric", return_value=False),
        patch.object(runner, "use_symmetric_memory", return_value=nullcontext()),
    ):
        return runner.fused_experts_none_to_flashinfer_cutlass(
            dispatch, method.flashinfer_quant_info, config
        ).hidden_states


@pytest.mark.parametrize("tp_size,tp_rank", [(8, 3), (1, 0)])
def test_real_gate_up_loader(tp_size, tp_rank):
    config = MoeRunnerConfig(is_gated=True)
    for dtype in (torch.int8, torch.float32):
        layer = SimpleNamespace(
            moe_runner_config=config,
            quant_method=SimpleNamespace(load_up_proj_weight_first=True),
            moe_tp_size=tp_size,
            use_padded_loading=False,
            use_presharded_weights=False,
            use_triton_kernels=False,
        )
        out = torch.zeros(8, 16, device="cuda", dtype=dtype)
        gate = (
            torch.arange(4 * tp_size, device="cuda")
            .to(dtype)
            .view(-1, 1)
            .expand(-1, 16)
        )
        up = gate + 64
        for shard, values in (("w1", gate), ("w3", up)):
            FusedMoE._load_w13(layer, out, 0, shard, values, tp_rank)
        torch.testing.assert_close(out[:4], up[tp_rank * 4 : (tp_rank + 1) * 4])
        torch.testing.assert_close(out[4:], gate[tp_rank * 4 : (tp_rank + 1) * 4])


def test_payload_and_fail_fast():
    methods, _, config = make_layer(8, 3)
    method, layer = methods[1]
    q = method.flashinfer_quant_info
    assert (q.moe_tp_size, q.moe_tp_rank, q.moe_ep_size, q.moe_ep_rank) == (1, 0, 8, 3)
    assert len(q.quant_scales) == 8
    assert [t.numel() for t in q.quant_scales[2:]] == [6144, 2048, 0, 0, 32, 32]
    assert all(t.is_contiguous() for t in q.quant_scales)
    assert all(t.dtype == torch.bfloat16 for t in q.quant_scales[:6])
    assert all(t.dtype == torch.float32 for t in q.quant_scales[6:])
    for index, value in ((2, 1 / 0.75), (3, 1 / 1.25), (6, 0.75), (7, 1.25)):
        torch.testing.assert_close(
            q.quant_scales[index],
            torch.full_like(q.quant_scales[index], value),
            rtol=0,
            atol=0,
        )
    with (
        patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=MoeA2ABackend.FLASHINFER,
        ),
    ):
        with pytest.raises(ValueError, match="none or deepep"):
            method._validate_flashinfer_config()
    for name, value in (
        ("activation", "gelu"),
        ("is_gated", False),
        ("apply_router_weight_on_input", True),
    ):
        bad_config = MoeRunnerConfig(**{name: value})
        with pytest.raises(ValueError, match="gated SiLU"):
            method.create_moe_runner(layer, bad_config)
    with patch.object(torch.cuda, "get_device_capability", return_value=(10, 0)):
        with patch(
            "sglang.srt.layers.moe.utils.get_moe_a2a_backend",
            return_value=MoeA2ABackend.NONE,
        ):
            with pytest.raises(ValueError, match="SM90"):
                method._validate_flashinfer_config()


@pytest.mark.parametrize("ep_size,ep_rank", [(1, 0), (8, 3)])
@pytest.mark.parametrize("finalize", [False, True])
def test_direct_flashinfer_and_empty_expert_replay(ep_size, ep_rank, finalize):
    """Separate adapter/replay correctness from cross-kernel rounding differences."""
    from flashinfer.fused_moe import (
        cutlass_fused_moe,
        interleave_moe_scales_for_sm90_mixed_gemm,
        interleave_moe_weights_for_sm90_mixed_gemm,
    )
    from flashinfer.fused_moe.core import ActivationType

    methods, raw, config = make_layer(ep_size, ep_rank)
    method, layer = methods[1]
    w1, w2, s1, s2, _, _ = raw
    packed1 = pack_int4_values_to_int8(torch.cat(w1.chunk(2, dim=1)[::-1], dim=1))
    packed2 = pack_int4_values_to_int8(w2)
    direct_w1 = interleave_moe_weights_for_sm90_mixed_gemm(
        packed1.view(torch.uint8), "int4"
    )
    direct_w2 = interleave_moe_weights_for_sm90_mixed_gemm(
        packed2.view(torch.uint8), "int4"
    )
    direct_s1 = interleave_moe_scales_for_sm90_mixed_gemm(
        torch.cat(s1.chunk(2, dim=1)[::-1], dim=1), 128
    )
    direct_s2 = interleave_moe_scales_for_sm90_mixed_gemm(s2, 128)
    torch.testing.assert_close(
        method.flashinfer_quant_info.w13_weight, direct_w1, rtol=0, atol=0
    )
    torch.testing.assert_close(
        method.flashinfer_quant_info.w2_weight, direct_w2, rtol=0, atol=0
    )
    scales = [direct_s1, direct_s2, *method.flashinfer_quant_info.quant_scales[2:]]
    x = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(8, device="cuda", dtype=torch.int32).expand(8, 8).contiguous()
    scores = torch.full((8, 8), 0.125, device="cuda")
    dispatch = StandardDispatchOutput(x, None, StandardTopKOutput(scores, ids, None))
    with patch.object(
        runner.envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE, "get", return_value=finalize
    ):
        for _ in range(3):
            invoke(method, dispatch, config)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = invoke(method, dispatch, config)
        for start in (0, 96, 128, 96, 0):
            ids.copy_(
                (torch.arange(8, device="cuda", dtype=torch.int32) + start).expand(8, 8)
            )
            x.normal_()
            output.fill_(float("nan"))
            graph.replay()
            direct = torch.full_like(x, float("nan"))
            cutlass_fused_moe(
                input=x,
                output=direct,
                token_selected_experts=ids,
                token_final_scales=scores,
                fc1_expert_weights=direct_w1,
                fc2_expert_weights=direct_w2,
                quant_scales=scales,
                output_dtype=torch.bfloat16,
                tp_size=layer.moe_tp_size,
                tp_rank=layer.moe_tp_rank,
                ep_size=ep_size,
                ep_rank=ep_rank,
                activation_type=ActivationType.Swiglu,
                use_w4_group_scaling=True,
                use_packed_weights=True,
                use_fused_finalize=finalize,
                tune_max_num_tokens=8,
            )
            direct.mul_(config.routed_scaling_factor)
            torch.testing.assert_close(output, direct, rtol=0, atol=0)
            if ep_size == 8 and start != 96:
                assert torch.count_nonzero(output).item() == 0


def test_empty_input_does_not_launch_kernel():
    methods, _, config = make_layer(8, 3)
    method, _ = methods[1]
    x = torch.empty(0, 6144, device="cuda", dtype=torch.bfloat16)
    ids = torch.empty(0, 8, device="cuda", dtype=torch.int32)
    scores = torch.empty(0, 8, device="cuda")
    dispatch = StandardDispatchOutput(x, None, StandardTopKOutput(scores, ids, None))
    with patch.object(runner, "_flashinfer_cutlass_fused_moe") as kernel:
        output = invoke(method, dispatch, config)
    kernel.assert_not_called()
    assert output.shape == x.shape
    assert output.dtype == x.dtype


@pytest.mark.parametrize("ep_rank", [0, 3, 7])
@pytest.mark.parametrize("finalize", [False, True])
def test_deepep_normal_matches_standard_local_contribution(ep_rank, finalize):
    from dataclasses import replace

    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput

    methods, _, config = make_layer(8, ep_rank)
    method, _ = methods[1]
    x = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)
    local_ids = torch.arange(8, device="cuda").expand(8, 8).clone()
    local_ids[:, 4:] = -1
    local_ids[-1] = -1
    scores = torch.full((8, 8), 0.125, device="cuda")
    global_ids = torch.where(
        local_ids >= 0, local_ids + ep_rank * 32, 0 if ep_rank else 32
    ).int()
    expected_dispatch = StandardDispatchOutput(
        x, None, StandardTopKOutput(scores, global_ids, None)
    )
    dispatch = DeepEPNormalDispatchOutput(
        x, None, local_ids, scores, [8] * 4 + [0] * 28
    )
    with patch.object(
        runner.envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE, "get", return_value=finalize
    ):
        expected = invoke(
            method, expected_dispatch, replace(config, routed_scaling_factor=None)
        )
        with (
            patch.object(runner, "get_tp_group", return_value=None),
            patch.object(runner, "is_allocation_symmetric", return_value=False),
            patch.object(runner, "use_symmetric_memory", return_value=nullcontext()),
        ):
            actual = runner.fused_experts_deepep_to_flashinfer_cutlass(
                dispatch, method.flashinfer_quant_info, config
            )
        torch.testing.assert_close(actual.hidden_states, expected, rtol=0, atol=0)
        assert actual.hidden_states[-1].count_nonzero().item() == 0


@pytest.mark.parametrize("ep_rank", [0, 3])
@pytest.mark.parametrize("finalize", [False, True])
def test_deepep_low_latency_graph_replay(ep_rank, finalize):
    from dataclasses import replace
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPLLDispatchOutput

    methods, _, config = make_layer(8, ep_rank)
    method, _ = methods[1]
    x = torch.randn(32, 3, 6144, device="cuda", dtype=torch.bfloat16)
    counts = torch.zeros(32, device="cuda", dtype=torch.int32)
    ids = torch.tensor([[ep_rank * 32, ep_rank * 32 + 1]], device="cuda")
    scores = torch.tensor([[0.25, 0.75]], device="cuda")
    dispatch = DeepEPLLDispatchOutput(x, None, ids, scores, counts, 1)
    with (
        patch.object(
            runner.envs.SGLANG_FLASHINFER_MOE_FUSED_FINALIZE,
            "get",
            return_value=finalize,
        ),
        patch.object(runner, "get_tp_group", return_value=None),
        patch.object(runner, "is_allocation_symmetric", return_value=False),
        patch.object(runner, "use_symmetric_memory", return_value=nullcontext()),
    ):

        def run():
            return runner.fused_experts_deepep_to_flashinfer_cutlass(
                dispatch, method.flashinfer_quant_info, config
            ).hidden_states

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()
        for first, second in ((3, 1), (0, 0), (1, 2)):
            counts.zero_()
            counts[:2].copy_(
                torch.tensor([first, second], device="cuda", dtype=torch.int32)
            )
            x.normal_()
            live = torch.arange(3, device="cuda")[None, :] < counts[:, None]
            x.masked_fill_(~live[:, :, None], float("nan"))
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            # Independently run only live rows, one expert at a time. DeepEP
            # combines these unweighted outputs using its original scores.
            for expert, count in enumerate((first, second)):
                if count:
                    standard = StandardDispatchOutput(
                        x[expert, :count],
                        None,
                        StandardTopKOutput(
                            torch.ones(count, 1, device="cuda"),
                            torch.full(
                                (count, 1),
                                ep_rank * 32 + expert,
                                device="cuda",
                                dtype=torch.int32,
                            ),
                            None,
                        ),
                    )
                    expected = invoke(
                        method, standard, replace(config, routed_scaling_factor=None)
                    )
                    torch.testing.assert_close(
                        output[expert, :count], expected, rtol=1e-2, atol=0.1
                    )
            assert torch.isfinite(output).all()
            assert output[~live].count_nonzero().item() == 0
