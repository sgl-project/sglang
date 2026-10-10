"""Cake Kimi-K3 SiTU fused MoE through sglang.kernels.

Checks that the registry resolves the explicit FlashInfer backend for the
three SiTU entries, that the facade output is bitwise identical to calling
FlashInfer directly, and that the fused result matches FlashInfer's TRT-LLM
NVFP4 routed MoE reference within FP4 block-scaled tolerance. Skips (with the
reason) when FlashInfer lacks the Cake module or the GPU is not sm_100a /
sm_103a. The fixture quantizes the full 896-expert weight set once (multi-GiB).
"""

import sys

import pytest
import torch

from sglang.kernels import KernelBackend, select_kernel
from sglang.kernels.cake_kernels import moe_kimi_k3_situ as adapter
from sglang.kernels.cake_kernels._support import flashinfer_module_available
from sglang.kernels.ops.moe.cake import (
    cake_kimi_k3_situ_fused_moe,
    cake_kimi_k3_situ_fused_moe_prepare_workspace,
    cake_kimi_k3_situ_fused_moe_workspace_size,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

OPS = (
    "moe.kimi_k3_situ_fused_moe_workspace_size",
    "moe.kimi_k3_situ_fused_moe_prepare_workspace",
    "moe.kimi_k3_situ_fused_moe",
)
H, INTER, E, TOP_K = (
    adapter.HIDDEN,
    adapter.INTERMEDIATE,
    adapter.NUM_EXPERTS,
    adapter.TOP_K,
)


@pytest.mark.parametrize("op", OPS)
def test_registry_resolves_flashinfer_backend(op):
    spec = select_kernel(op, backend=KernelBackend.FLASHINFER)
    assert spec.backend is KernelBackend.FLASHINFER
    assert spec.target.startswith("sglang.kernels.cake_kernels.moe_kimi_k3_situ:")


def _skip_unless_supported():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not flashinfer_module_available(adapter.FI_MODULE, adapter.FI_JIT_MODULE):
        pytest.skip("installed FlashInfer lacks flashinfer.fused_moe.cake_kimi_k3_situ")
    cc = torch.cuda.get_device_capability()
    if cc not in adapter.ARCHS:
        pytest.skip(f"Cake SiTU is built for sm_100a/sm_103a, device is {cc}")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="module")
def situ_weights():
    device = _skip_unless_supported()
    from flashinfer.fused_moe import QuantConfig, QuantFormat, SiTU, TrtllmFp4Config

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH", "1")
        patch.setenv("FLASHINFER_NVFP4_4OVER6", "0")
        gen = torch.Generator(device=device).manual_seed(4568)
        w1 = torch.randn(
            E, 2 * INTER, H, device=device, dtype=torch.bfloat16, generator=gen
        )
        w1.mul_(0.125)
        w2 = torch.randn(
            E, H, INTER, device=device, dtype=torch.bfloat16, generator=gen
        )
        w2.mul_(0.125)
        prepared = TrtllmFp4Config.prepare_weights(
            w1,
            w2,
            quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
            num_local_experts=E,
            hidden_size=H,
            intermediate_size=INTER,
            activation=SiTU(),
        )
        del w1, w2
        yield prepared


def _trtllm_reference(x, ids, route_weights, prepared):
    from flashinfer.fused_moe import (
        QuantConfig,
        QuantFormat,
        TrtllmFp4Config,
        trtllm_fp4_block_scale_routed_moe,
    )
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType
    from flashinfer.utils import device_support_pdl

    quantized, scales = TrtllmFp4Config.prepare_activations(
        x, quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
    )
    output = torch.empty_like(x)
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(ids, route_weights),
        routing_bias=None,
        hidden_states=quantized,
        hidden_states_scale=scales,
        gemm1_weights=prepared["gemm1_weights"],
        gemm1_weights_scale=prepared["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=prepared["gemm1_alpha"],
        gemm1_beta=prepared["gemm1_beta"],
        gemm1_clamp_limit=None,
        gemm2_weights=prepared["gemm2_weights"],
        gemm2_weights_scale=prepared["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=prepared["output1_scale_scalar"],
        output1_scale_gate_scalar=prepared["output1_scale_gate_scalar"],
        output2_scale_scalar=prepared["output2_scale_scalar"],
        num_experts=E,
        top_k=TOP_K,
        n_group=None,
        topk_group=None,
        intermediate_size=INTER,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.Renormalize.value,
        activation_type=ActivationType.Situ.value,
        do_finalize=True,
        enable_pdl=device_support_pdl(x.device),
        per_token_scale=None,
        output=output,
        tune_max_num_tokens=16384,
    )
    if isinstance(result, (list, tuple)):
        result = result[0]
    assert result.data_ptr() == output.data_ptr()
    return output


@pytest.mark.parametrize("num_tokens", [64, 256])
def test_matches_flashinfer_and_reference(num_tokens, situ_weights):
    device = _skip_unless_supported()
    prepared = situ_weights
    gen = torch.Generator(device=device).manual_seed(9000 + num_tokens)
    x = torch.randn(num_tokens, H, device=device, dtype=torch.bfloat16, generator=gen)
    slots = torch.arange(TOP_K, dtype=torch.int32, device=device)
    tokens = torch.arange(num_tokens, dtype=torch.int32, device=device)
    ids = ((tokens[:, None] * TOP_K + slots[None, :]) % E).contiguous()
    route_weights = (
        torch.randn(num_tokens, TOP_K, device=device, generator=gen)
        .softmax(dim=-1)
        .to(torch.bfloat16)
    )
    quant_scales = [
        torch.ones(1, device=device, dtype=torch.float32),
        prepared["gemm1_weights_scale"],
        prepared["output1_scale_gate_scalar"],
        prepared["output1_scale_scalar"],
        prepared["gemm2_weights_scale"],
        prepared["output2_scale_scalar"],
    ]
    assert adapter.supports_kimi_k3_situ_fused_moe(
        x,
        ids,
        route_weights,
        prepared["gemm1_weights"],
        prepared["gemm2_weights"],
        quant_scales,
    )

    nbytes = cake_kimi_k3_situ_fused_moe_workspace_size(num_tokens, device=device)
    from flashinfer.fused_moe import cutlass_fused_moe, cutlass_fused_moe_workspace_size
    from flashinfer.tllm_enums import ActivationType

    assert nbytes == cutlass_fused_moe_workspace_size(
        num_tokens,
        H,
        INTER,
        E,
        TOP_K,
        x_dtype=torch.bfloat16,
        weight_dtype=torch.uint8,
        activation_type=ActivationType.Situ,
        tp_size=8,
        backend="cake",
    )
    workspace = torch.empty(nbytes, dtype=torch.uint8, device=device)
    assert adapter.supports_kimi_k3_situ_workspace(workspace, num_tokens)
    assert (
        cake_kimi_k3_situ_fused_moe_prepare_workspace(workspace, num_tokens)
        is workspace
    )

    output = torch.full_like(x, float("nan"))
    result = cake_kimi_k3_situ_fused_moe(
        x,
        ids,
        route_weights,
        prepared["gemm1_weights"],
        prepared["gemm2_weights"],
        quant_scales,
        output=output,
        workspace_buffer=workspace,
    )
    assert result is output

    output_fi = torch.full_like(x, float("nan"))
    cutlass_fused_moe(
        x,
        ids,
        route_weights,
        prepared["gemm1_weights"],
        prepared["gemm2_weights"],
        torch.bfloat16,
        quant_scales,
        activation_type=ActivationType.Situ,
        tp_size=8,
        backend="cake",
        output=output_fi,
        workspace_buffer=workspace,
    )
    torch.cuda.synchronize()
    assert torch.equal(output, output_fi)

    expected = _trtllm_reference(x, ids, route_weights, prepared)
    assert expected.abs().max() > 2.0
    torch.testing.assert_close(output, expected, atol=1.0, rtol=0.1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
