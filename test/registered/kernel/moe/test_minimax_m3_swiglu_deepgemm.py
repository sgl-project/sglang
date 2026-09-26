# SPDX-License-Identifier: Apache-2.0
import sys

import pytest
import torch

from sglang.kernels.ops.moe.minimax_m3_swiglu_deepgemm import swiglu_quant
from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_group_quant_fp8
from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
    swiglu_no_interleaved_with_alpha_and_limit,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("intermediate", [384, 768, 1536, 3072, 12288])
@pytest.mark.parametrize("rows", [1, 31, 240])
@pytest.mark.parametrize("magnitude", [0.0, 1.0, 16.0])
@torch.no_grad()
def test_swiglu_preserves_bf16_rounding_and_packed_scale(intermediate, rows, magnitude):
    torch.manual_seed(intermediate + rows)
    x = (
        torch.randn(rows, 2 * intermediate, device="cuda", dtype=torch.bfloat16)
        * magnitude
    )
    activated = swiglu_no_interleaved_with_alpha_and_limit(x, 1.702, 7.0)
    expected_q, expected_s = sglang_per_token_group_quant_fp8(
        activated,
        32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    q, s = swiglu_quant(x, 1.702, 7.0)
    torch.testing.assert_close(q.float(), expected_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
    assert s.stride() == expected_s.stride()


@pytest.mark.parametrize("alpha,limit", [(1.702, float("inf")), (1.0, 7.0), (2.0, 5.0)])
@torch.no_grad()
def test_activation_parameters(alpha, limit):
    x = torch.randn(31, 1280, device="cuda", dtype=torch.bfloat16) * 8
    activated = swiglu_no_interleaved_with_alpha_and_limit(x, alpha, limit)
    expected = sglang_per_token_group_quant_fp8(
        activated,
        32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    actual = swiglu_quant(x, alpha, limit)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.float(), b.float(), rtol=0, atol=0)


def test_empty_and_invalid_shape():
    q, scales = swiglu_quant(torch.empty(0, 768, device="cuda", dtype=torch.bfloat16))
    assert q.shape == (0, 384) and scales.shape == (0, 3)
    with pytest.raises(AssertionError):
        swiglu_quant(torch.empty(1, 769, device="cuda", dtype=torch.bfloat16))


@pytest.fixture(scope="module")
def model_parallel():
    from sglang.srt.distributed.parallel_state import destroy_model_parallel
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs
    from sglang.test.layer_ut_utils import init_single_process_dist

    publish(ServerArgs(model_path="dummy"), role="test")
    init_single_process_dist(master_port=29684)
    try:
        yield
    finally:
        destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    "hidden,intermediate,tp",
    [(6144, i, tp) for i in (3072, 12288) for tp in (1, 4, 8)],
)
@torch.no_grad()
def test_actual_mlp_and_graph_weight_update(
    hidden, intermediate, tp, model_parallel, monkeypatch
):
    from types import SimpleNamespace

    from sglang.srt.layers.deep_gemm_wrapper import compile_utils
    from sglang.srt.layers.quantization import fp8_utils
    from sglang.srt.layers.quantization.fp8 import Fp8Config
    from sglang.srt.models.minimax_m3 import MiniMaxM3MLP

    monkeypatch.setattr(
        fp8_utils, "FP8_GEMM_RUNNER_BACKEND", fp8_utils.Fp8GemmRunnerBackend.DEEP_GEMM
    )
    monkeypatch.setattr(compile_utils, "_ENABLE_JIT_DEEPGEMM_PRECOMPILE", False)
    monkeypatch.setenv("SGLANG_OPT_MINIMAX_M3_FUSED_SWIGLU_MXFP8", "1")
    config = SimpleNamespace(
        hidden_size=hidden, hidden_act="swigluoai", swiglu_alpha=1.702, swiglu_limit=7.0
    )
    layer = MiniMaxM3MLP(
        config,
        quant_config=Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            use_mxfp8=True,
        ),
        intermediate_size=intermediate,
        tp_rank=0,
        tp_size=tp,
        reduce_results=False,
    ).cuda()
    assert layer._fuse_swiglu_mxfp8
    for proj in [layer.gate_up_proj, layer.down_proj]:
        n, k = proj.weight.shape
        raw = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
        q, s = sglang_per_token_group_quant_fp8(
            raw, 32, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=True
        )
        proj.weight.copy_(q)
        raw_scales = (
            s.T.contiguous()
            .view(torch.uint8)
            .reshape(k // 128, n, 4)
            .permute(1, 0, 2)
            .reshape(n, k // 32)
        )
        proj.weight_scale_inv.copy_(raw_scales)
        proj.quant_method.process_weights_after_loading(proj)
    for m in [0, 1, 31, 240, 1024, 1025]:
        x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
        layer._fuse_swiglu_mxfp8 = False
        expected = layer(x)
        layer._fuse_swiglu_mxfp8 = True
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)

    x = x[:240]

    # Sequence-parallel linears consume tensors, not pre-quantized tuples.
    from sglang.srt.runtime_context import get_forward

    with get_forward().scoped(sp_active=True):
        layer._fuse_swiglu_mxfp8 = False
        expected = layer(x, should_allreduce_fusion=True)
        layer._fuse_swiglu_mxfp8 = True
        torch.testing.assert_close(
            layer(x, should_allreduce_fusion=True), expected, rtol=0, atol=0
        )

    x_3d = x.unsqueeze(0)
    layer._fuse_swiglu_mxfp8 = False
    expected = layer(x_3d)
    layer._fuse_swiglu_mxfp8 = True
    torch.testing.assert_close(layer(x_3d), expected, rtol=0, atol=0)

    if intermediate == 3072 and tp == 1:
        compiled = torch.compile(layer, backend="eager")
        layer._fuse_swiglu_mxfp8 = False
        expected = compiled(x)
        layer._fuse_swiglu_mxfp8 = True
        torch.testing.assert_close(compiled(x), expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = layer(x)
    before = layer(x).clone()
    scale_ptr = layer.down_proj.weight_scale_inv_deepgemm.data_ptr()
    layer.down_proj.weight_scale_inv.add_(1)
    layer.down_proj.quant_method.process_weights_after_loading(layer.down_proj)
    graph.replay()
    torch.cuda.synchronize()
    layer._fuse_swiglu_mxfp8 = False
    expected = layer(x)
    assert scale_ptr == layer.down_proj.weight_scale_inv_deepgemm.data_ptr()
    assert not torch.equal(expected, before)
    torch.testing.assert_close(captured, expected, rtol=0, atol=0)


@pytest.mark.parametrize("hidden_act", ["swigluoai", "silu"])
@torch.no_grad()
def test_unquantized_fallback(hidden_act, model_parallel, monkeypatch):
    from types import SimpleNamespace

    from sglang.srt.models.minimax_m3 import MiniMaxM3MLP

    monkeypatch.setenv("SGLANG_OPT_MINIMAX_M3_FUSED_SWIGLU_MXFP8", "1")
    config = SimpleNamespace(
        hidden_size=256, hidden_act=hidden_act, swiglu_alpha=1.702, swiglu_limit=7.0
    )
    layer = (
        MiniMaxM3MLP(config, intermediate_size=128, tp_rank=0, tp_size=1)
        .cuda()
        .bfloat16()
    )
    for p in layer.parameters():
        p.normal_(0, 0.02)
    assert not layer._fuse_swiglu_mxfp8
    x = torch.randn(31, 256, device="cuda", dtype=torch.bfloat16)
    expected = layer.down_proj(layer.act_fn(layer.gate_up_proj(x)[0]))[0]
    torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["dtype", "layout", "shape", "fallback"])
@torch.no_grad()
def test_prequantized_linear_rejects_incompatible_scales(invalid):
    from sglang.srt.layers.quantization.fp8_utils import (
        _deepgemm_w8a8_mxfp8_linear_with_fallback,
    )

    x = torch.randn(31, 768, device="cuda", dtype=torch.bfloat16)
    q, scale = swiglu_quant(x)
    weight = torch.zeros(64, 384, device="cuda", dtype=torch.float8_e4m3fn)
    if invalid == "dtype":
        scale = scale.to(torch.float32)
    elif invalid == "layout":
        scale = scale.contiguous()
    elif invalid == "shape":
        scale = scale[:, :2]
    else:
        weight = weight[:63]
    with pytest.raises(ValueError, match="packed"):
        _deepgemm_w8a8_mxfp8_linear_with_fallback(
            input=q,
            weight=weight,
            weight_scale=torch.empty(0, device="cuda"),
            input_scale=scale,
        )


@torch.no_grad()
def test_prequantized_linear_output_and_bias():
    from sglang.srt.layers.quantization.fp8_utils import (
        _deepgemm_w8a8_mxfp8_linear_with_fallback,
    )

    x = torch.randn(31, 768, device="cuda", dtype=torch.bfloat16)
    activated = swiglu_no_interleaved_with_alpha_and_limit(x, 1.702, 7.0)
    q, scale = swiglu_quant(x)
    weight, weight_scale = sglang_per_token_group_quant_fp8(
        torch.randn(64, 384, device="cuda", dtype=torch.bfloat16),
        32,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    bias = torch.randn(64, device="cuda", dtype=torch.bfloat16)
    expected = _deepgemm_w8a8_mxfp8_linear_with_fallback(
        input=activated, weight=weight, weight_scale=weight_scale, bias=bias
    )
    actual = _deepgemm_w8a8_mxfp8_linear_with_fallback(
        input=q, weight=weight, weight_scale=weight_scale, input_scale=scale, bias=bias
    )
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
