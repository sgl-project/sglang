"""Fused static-FP8 producers for Qwen3.5 ModelOpt checkpoints: each must give
the same FP8 codes as the unfused producer followed by static_quant_fp8."""

import pytest
import torch

from sglang.kernels.ops.attention.fla.layernorm_gated import _layer_norm_fwd
from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul
from sglang.kernels.ops.layernorm.gemma_norm_fp8 import gemma_rmsnorm_bf16_fp8
from sglang.kernels.ops.quantization.fp8_kernel import static_quant_fp8
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

TOKENS = [1, 7, 64, 513]


def _scale(x: torch.Tensor) -> torch.Tensor:
    return (x.float().abs().amax() / 448.0 * 0.8).reshape(1).contiguous()


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("activation", ["swish", "sigmoid"])
def test_gated_norm_static_fp8_matches_chain(tokens, activation):
    torch.manual_seed(tokens)
    heads, head_dim = 48, 128
    x = torch.randn(tokens * heads, head_dim, device="cuda", dtype=torch.bfloat16)
    z = torch.randn_like(x)
    w = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
    ref, _, _ = _layer_norm_fwd(
        x, w, None, 1e-6, z=z, is_rms_norm=True, activation=activation
    )
    scale = _scale(ref)
    ref_q, _ = static_quant_fp8(ref.view(tokens, -1).contiguous(), scale)
    q, _, _ = _layer_norm_fwd(
        x,
        w,
        None,
        1e-6,
        z=z,
        is_rms_norm=True,
        activation=activation,
        static_quant_scale=scale,
    )
    _assert_codes_close(q.view(tokens, -1), ref_q, f"gated-{activation}-{tokens}")


def _assert_codes_close(q, ref_q, tag):
    """FP8 E4M3 codes as signed steps; at most one step apart on <=1e-4 of them."""
    to_steps = lambda t: t.float().sign() * t.view(torch.uint8).bitwise_and(0x7F).int()
    diff = (to_steps(q) - to_steps(ref_q)).abs()
    frac = (diff != 0).float().mean().item()
    print(f"{tag}: max step diff {diff.max().item():.0f}, mismatched {frac:.2e}")
    assert diff.max().item() <= 1 and frac <= 1e-4, (tag, diff.max().item(), frac)


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("strided_gate", [False, True])
def test_sigmoid_gate_static_fp8_matches_chain(tokens, strided_gate):
    torch.manual_seed(tokens)
    heads, head_dim = 24, 256
    attn = torch.randn(tokens, heads * head_dim, device="cuda", dtype=torch.bfloat16)
    if strided_gate:
        packed = torch.randn(
            tokens, heads, 2 * head_dim, device="cuda", dtype=torch.bfloat16
        )
        gate = packed[..., head_dim:]
    else:
        gate = torch.randn_like(attn)
    ref = fused_sigmoid_mul(attn, gate, inplace=False)
    scale = _scale(ref)
    ref_q, _ = static_quant_fp8(ref, scale)
    q = fused_sigmoid_mul(attn, gate, static_quant_scale=scale)
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("with_residual", [True, False])
def test_gemma_rmsnorm_bf16_fp8_matches_sgl_kernel(tokens, with_residual):
    from sgl_kernel import gemma_fused_add_rmsnorm, gemma_rmsnorm

    torch.manual_seed(tokens)
    hidden = 5120
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w = (torch.randn(hidden, device="cuda") * 0.1).to(torch.bfloat16)
    res = torch.randn_like(x)

    ref_x, ref_res = x.clone(), res.clone()
    if with_residual:
        gemma_fused_add_rmsnorm(ref_x, ref_res, w, 1e-6)
        ref_y = ref_x
    else:
        ref_y = gemma_rmsnorm(ref_x, w, 1e-6)
    scale = _scale(ref_y)
    ref_q, _ = static_quant_fp8(ref_y, scale)

    x_in, res_in = x.clone(), res.clone()
    y, q = gemma_rmsnorm_bf16_fp8(
        x_in, w, 1e-6, scale, res_in if with_residual else None
    )
    if with_residual:
        assert torch.equal(res_in, ref_res)
    else:
        assert torch.equal(x_in, x), "x must stay readable without a residual"
    # Only the sum-of-squares reduction order differs from the CUDA kernel.
    diff = (y.view(torch.int16).int() - ref_y.view(torch.int16).int()).abs()
    assert diff.max().item() <= 1, f"bf16 ulp diff {diff.max().item()}"
    code_diff = (q.view(torch.uint8).int() - ref_q.view(torch.uint8).int()).abs()
    assert code_diff.max().item() <= 1
    assert (code_diff != 0).float().mean().item() <= 1e-3
