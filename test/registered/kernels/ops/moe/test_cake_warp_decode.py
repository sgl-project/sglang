"""Cake NVFP4 warp-decode MoE runner through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend for the four
warp-decode entries, that the facade config / weight / activation preparation
equals the direct FlashInfer calls, that ``supports_warp_decode`` mirrors the
calibrated geometry table, and that a ``MoELayer`` built on
``CakeWarpDecodeConfig`` matches FlashInfer's TRT-LLM NVFP4 routed MoE within
FP4 tolerance. Skips (with the reason) when FlashInfer lacks the module or the
GPU is not sm_100a / sm_103a.
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_warp_decode as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import (
    cake_warp_decode_config,
    cake_warp_decode_prepare_activations,
    cake_warp_decode_prepare_weights,
    cake_warp_decode_runner,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "moe.warp_decode_config",
    "moe.warp_decode_runner",
    "moe.warp_decode_prepare_weights",
    "moe.warp_decode_prepare_activations",
)
# SwiGLU(), hidden 2048, intermediate 1536, 60 experts, top-4.
H, INTER, E, TOP_K = 2048, 1536, 60, 4


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_warp_decode:")


def test_geometry_table_matches_flashinfer_runner():
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks the Cake warp-decode runner")
    from flashinfer.fused_moe import CakeWarpDecodeRunner

    table = {
        (adapter.activation_key(act), h, i, e, k)
        for act, h, i, e, k in CakeWarpDecodeRunner._SUPPORTED_CONFIGURATIONS
    }
    assert table == set(adapter.GEOMETRIES)


def test_supports_rejects_outside_table():
    assert not adapter.supports_warp_decode(
        activation="swiglu",
        hidden_size=2048,
        intermediate_size=1536,
        num_experts=61,
        top_k=4,
    )
    assert not adapter.supports_warp_decode(
        activation="relu",
        hidden_size=2048,
        intermediate_size=1536,
        num_experts=60,
        top_k=4,
    )
    assert not adapter.supports_warp_decode(
        activation="swiglu",
        hidden_size=2048,
        intermediate_size=1536,
        num_experts=60,
        top_k=4,
        num_tokens=33,
    )


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip(
            "installed FlashInfer lacks flashinfer.jit.cake_fused_moe_warp_decode"
        )
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(f"Cake warp decode is built for sm_100a/sm_103a, device is {cc}")
    return torch.device("cuda", torch.cuda.current_device())


def test_config_and_runner_match_flashinfer():
    device = _skip_unless_supported()
    from flashinfer.fused_moe import (
        BackendOptions,
        CakeWarpDecodeConfig,
        CakeWarpDecodeRunner,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )

    assert cake_warp_decode_config() == CakeWarpDecodeConfig(backend="cake")
    config = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=TOP_K),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=INTER),
        activation=SwiGLU(),
        backend=BackendOptions((cake_warp_decode_config(),)),
        execution=ExecutionConfig(enable_pdl=True),
    )
    runner = cake_warp_decode_runner(config, device)
    assert isinstance(runner, CakeWarpDecodeRunner)
    runner.check_support()
    assert adapter.supports_warp_decode(
        activation=SwiGLU(),
        hidden_size=H,
        intermediate_size=INTER,
        num_experts=E,
        top_k=TOP_K,
        num_tokens=8,
        device=device,
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32])
def test_layer_matches_flashinfer_and_reference(num_tokens):
    device = _skip_unless_supported()
    from flashinfer.fused_moe import (
        BackendOptions,
        CakeWarpDecodeConfig,
        ExecutionConfig,
        ExpertConfig,
        MoEActivationPack,
        MoEConfig,
        MoELayer,
        MoEWeightPack,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        RoutingInputMode,
        SwiGLU,
        TrtllmFp4Config,
        trtllm_fp4_block_scale_routed_moe,
    )
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType
    from flashinfer.utils import device_support_pdl

    gen = torch.Generator(device=device).manual_seed(1000 + num_tokens)
    w1 = (
        torch.randn(E, 2 * INTER, H, device=device, dtype=torch.bfloat16, generator=gen)
        * 0.125
    )
    w2 = (
        torch.randn(E, H, INTER, device=device, dtype=torch.bfloat16, generator=gen)
        * 0.125
    )
    view = cake_warp_decode_prepare_weights(
        w1, w2, num_local_experts=E, hidden_size=H, intermediate_size=INTER
    )
    view_fi = CakeWarpDecodeConfig.prepare_weights(
        w1, w2, num_local_experts=E, hidden_size=H, intermediate_size=INTER
    )
    assert set(view) == set(view_fi)
    for key in view:
        assert torch.equal(view[key], view_fi[key]), key

    x = torch.randn(num_tokens, H, device=device, dtype=torch.bfloat16, generator=gen)
    hidden_q, hidden_scale = cake_warp_decode_prepare_activations(x)
    hidden_q_fi, hidden_scale_fi = CakeWarpDecodeConfig.prepare_activations(x)
    assert torch.equal(hidden_q, hidden_q_fi)
    assert torch.equal(hidden_scale, hidden_scale_fi)

    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
    ids = ((tokens[:, None] * TOP_K + slots[None, :]) % E).contiguous()
    weights = (
        torch.randn(num_tokens, TOP_K, device=device, generator=gen)
        .softmax(dim=-1)
        .to(torch.bfloat16)
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=E, top_k=TOP_K),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=INTER),
        activation=SwiGLU(),
        backend=BackendOptions((cake_warp_decode_config(),)),
        execution=ExecutionConfig(enable_pdl=True),
    )
    layer = MoELayer(config, device)
    weight_pack = MoEWeightPack()
    weight_pack.prepare_for("cake", view)
    act_pack = MoEActivationPack(
        hidden_q,
        hidden_scale,
        ids,
        weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    out = layer(act_pack, weight_pack)
    out_again = layer(act_pack, weight_pack)
    torch.cuda.synchronize()
    assert torch.equal(out, out_again)

    quantized, scales = TrtllmFp4Config.prepare_activations(x)
    expected = torch.empty_like(x)
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, weights),
        routing_bias=None,
        hidden_states=quantized,
        hidden_states_scale=scales,
        gemm1_weights=view["gemm1_weights"],
        gemm1_weights_scale=view["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=view["gemm2_weights"],
        gemm2_weights_scale=view["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=view["output1_scale_scalar"],
        output1_scale_gate_scalar=view["output1_scale_gate_scalar"],
        output2_scale_scalar=view["output2_scale_scalar"],
        num_experts=E,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        activation_type=ActivationType.Swiglu.value,
        do_finalize=True,
        enable_pdl=device_support_pdl(device),
        per_token_scale=None,
        output=expected,
        tune_max_num_tokens=32,
    )
    if isinstance(result, (list, tuple)):
        result = result[0]
    torch.cuda.synchronize()
    assert expected.abs().max() > 1.0
    torch.testing.assert_close(out.float(), expected.float(), atol=1.0, rtol=0.1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
