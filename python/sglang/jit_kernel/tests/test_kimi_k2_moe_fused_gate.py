import pytest
import torch

from sglang.jit_kernel.kimi_k2_moe_fused_gate import (
    kimi_k2_moe_fused_gate as jit_kimi_k2_moe_fused_gate,
)
from sglang.srt.layers.moe.topk import kimi_k2_biased_topk_impl
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="base-b-kernel-unit-1-gpu-large")

# (num_experts, topk, routed_scaling_factor)
_CONFIGS = [
    (384, 6, 2.872),  # Kimi K2
    (256, 8, 1.0),  # MiMo V2.5
]

# Cross both the small-token (<=512 rows) and large-token kernels.
_SEQ_LENGTHS = list(range(1, 10)) + [16, 64, 128, 512, 513, 1024, 4096, 16384]


@pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
@pytest.mark.parametrize("config", _CONFIGS, ids=["kimi384", "mimo256"])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_jit_kimi_matches_biased_topk_reference(
    seq_length, config, apply_routed_scaling_factor_on_output
):
    """JIT kimi K2 fused gate weights match the kimi_k2_biased_topk_impl reference."""
    num_experts, topk, routed_scaling_factor = config
    renormalize = True

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=torch.float32, device="cuda")
    scores = tensor.clone()
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    output, indices = jit_kimi_k2_moe_fused_gate(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    ref_output, ref_indices = kimi_k2_biased_topk_impl(
        scores,
        scores,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )

    # Weights drive the MoE output; compare them after sorting (selection order differs).
    assert torch.allclose(
        ref_output.sort()[0].to(torch.float32),
        output.sort()[0].to(torch.float32),
        rtol=1e-2,
        atol=1e-3,
    ), (
        f"Weight mismatch at seq_length {seq_length}, num_experts {num_experts}, "
        f"topk {topk}, apply_scaling {apply_routed_scaling_factor_on_output}"
    )


@pytest.mark.parametrize("seq_length", _SEQ_LENGTHS)
@pytest.mark.parametrize("config", _CONFIGS, ids=["kimi384", "mimo256"])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_jit_kimi_matches_aot_kernel(
    seq_length, config, apply_routed_scaling_factor_on_output
):
    """JIT kimi K2 kernel reproduces the AOT sgl_kernel output (device code is verbatim)."""
    aot = pytest.importorskip("sgl_kernel")
    aot_kimi_k2_moe_fused_gate = getattr(aot, "kimi_k2_moe_fused_gate", None)
    if aot_kimi_k2_moe_fused_gate is None:
        pytest.skip("sgl_kernel.kimi_k2_moe_fused_gate not available")

    num_experts, topk, routed_scaling_factor = config
    renormalize = True

    torch.manual_seed(seq_length)
    tensor = torch.rand((seq_length, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    jit_out, jit_idx = jit_kimi_k2_moe_fused_gate(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    try:
        aot_out, aot_idx = aot_kimi_k2_moe_fused_gate(
            tensor,
            bias,
            topk=topk,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        )
    except RuntimeError as e:
        # Older AOT builds only support 384 experts; nothing to compare against then.
        pytest.skip(f"AOT kernel rejected this config: {e}")

    # Verbatim device kernels -> identical results. Sort to ignore selection order.
    torch.testing.assert_close(jit_out.sort(dim=-1)[0], aot_out.sort(dim=-1)[0])
    torch.testing.assert_close(jit_idx.sort(dim=-1)[0], aot_idx.sort(dim=-1)[0])


def test_jit_kimi_output_shapes_and_renorm():
    """Output shapes/dtypes are correct and renormalized weights sum to 1 per token."""
    num_experts, topk = 384, 6
    seq_length = 1024

    torch.manual_seed(42)
    tensor = torch.rand((seq_length, num_experts), dtype=torch.float32, device="cuda")
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda")

    output, indices = jit_kimi_k2_moe_fused_gate(
        tensor, bias, topk=topk, renormalize=True
    )

    assert output.shape == (seq_length, topk)
    assert indices.shape == (seq_length, topk)
    assert output.dtype == torch.float32
    assert indices.dtype == torch.int32

    weight_sums = output.sum(dim=-1)
    assert torch.allclose(
        weight_sums, torch.ones_like(weight_sums), rtol=1e-3, atol=1e-4
    )


_DTYPES = [torch.float32, torch.bfloat16, torch.float16]
_DTYPE_IDS = ["fp32", "bf16", "fp16"]


@pytest.mark.parametrize("seq_length", [1, 7, 64, 512, 513, 4096])
@pytest.mark.parametrize("config", _CONFIGS, ids=["kimi384", "mimo256"])
@pytest.mark.parametrize("input_dtype", _DTYPES, ids=[f"in_{d}" for d in _DTYPE_IDS])
@pytest.mark.parametrize("bias_dtype", _DTYPES, ids=[f"bias_{d}" for d in _DTYPE_IDS])
def test_jit_kimi_low_precision_input_matches_fp32(
    seq_length, config, input_dtype, bias_dtype
):
    """Any fp32/bf16/fp16 input+bias combo is widened exactly -> matches the fp32 path bitwise."""
    num_experts, topk, routed_scaling_factor = config
    renormalize = True

    torch.manual_seed(seq_length)
    tensor = torch.rand(
        (seq_length, num_experts), dtype=torch.float32, device="cuda"
    ).to(input_dtype)
    bias = torch.rand(num_experts, dtype=torch.float32, device="cuda").to(bias_dtype)

    output, indices = jit_kimi_k2_moe_fused_gate(
        tensor,
        bias,
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )
    # Host-side upcast is the exact same widening the kernel does internally.
    ref_output, ref_indices = jit_kimi_k2_moe_fused_gate(
        tensor.to(torch.float32),
        bias.to(torch.float32),
        topk=topk,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

    torch.testing.assert_close(output, ref_output, rtol=0, atol=0)
    torch.testing.assert_close(indices, ref_indices, rtol=0, atol=0)


def test_jit_kimi_unsupported_num_experts():
    """num_experts outside {256, 384} raises instead of silently mis-routing."""
    tensor = torch.rand((4, 128), dtype=torch.float32, device="cuda")
    bias = torch.rand(128, dtype=torch.float32, device="cuda")
    with pytest.raises(Exception):
        jit_kimi_k2_moe_fused_gate(tensor, bias, topk=4, renormalize=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
