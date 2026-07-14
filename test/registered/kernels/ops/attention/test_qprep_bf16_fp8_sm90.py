"""Tests for the SM90 Q8KV8 born-fp8 q-prep JIT kernel.

Gates (mirroring scripts/pr3_qprep_microbench.py conventions):
  (a) vs the Triton absorbed_bmm_concat_cast_q_fp8 "two_dot" variant with
      atol/rtol=2e-2 on the fp32 view (the accumulation order matches, so the
      output is empirically bitwise identical on SM90, but only the tolerance
      is contractual);
  (b) vs an fp64 bmm reference: mean |err| must match two_dot's;
  rope half must be bit-exact (identical bf16 -> fp8 conversion chain).
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")

N_LORA = 512  # kv_lora_rank
ROPE = 64  # qk_rope_head_dim


def _is_sm90() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0)


requires_sm90 = pytest.mark.skipif(not _is_sm90(), reason="requires SM90 (Hopper)")


def _make_inputs(T: int, H: int, K: int, seed: int = 1234, magnitude: float = 1.0):
    # Production layout: q_nope/q_rope are strided views of one [T, H, K+R]
    # q_b_proj output; w_kc is the N-major absorbed weight [H, K, N] with
    # strides (K*N, 1, K).
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn((T, H, K + ROPE), generator=g, device="cuda", dtype=torch.float32)
        * magnitude
    ).to(torch.bfloat16)
    w = (
        torch.randn((H, N_LORA, K), generator=g, device="cuda", dtype=torch.float32)
        / K**0.5
    ).to(torch.bfloat16)
    return q[..., :K], q[..., K:], w.transpose(1, 2)


@requires_sm90
@pytest.mark.parametrize("T", [1, 437, 1024])
@pytest.mark.parametrize(
    "h_k", [(64, 192), (128, 128)], ids=["glm_h64_k192", "ds_h128_k128"]
)
@pytest.mark.parametrize("pad_heads_extra", [0, 2])
def test_qprep_vs_triton_two_dot(T, h_k, pad_heads_extra):
    from sglang.jit_kernel.qprep_bf16_fp8_sm90 import q8kv8_qprep_fwd
    from sglang.srt.layers.attention.triton_ops.cache_ops import (
        absorbed_bmm_concat_cast_q_fp8,
    )

    H, K = h_k
    q_nope, q_rope, w_kc = _make_inputs(T, H, K)
    ph = H + pad_heads_extra
    ref = torch.zeros((T, ph, N_LORA + ROPE), dtype=torch.float8_e4m3fn, device="cuda")
    out = torch.zeros_like(ref)
    absorbed_bmm_concat_cast_q_fp8(ref, q_nope, w_kc, q_rope, H, variant="two_dot")
    q8kv8_qprep_fwd(out, q_nope, w_kc, q_rope, H)
    torch.cuda.synchronize()

    # rope half: identical conversion chain -> bit-exact.
    assert torch.equal(
        ref[:, :H, N_LORA:].contiguous().view(torch.uint8),
        out[:, :H, N_LORA:].contiguous().view(torch.uint8),
    ), "rope half must be bit-exact vs the Triton kernel"

    # gate (a): nope half within tolerance on the fp32 view.
    torch.testing.assert_close(
        out[:, :H].to(torch.float32),
        ref[:, :H].to(torch.float32),
        atol=2e-2,
        rtol=2e-2,
    )

    # padded head slice must stay untouched.
    if pad_heads_extra:
        assert out[:, H:].view(torch.uint8).max().item() == 0


@requires_sm90
@pytest.mark.parametrize(
    "h_k", [(64, 192), (128, 128)], ids=["glm_h64_k192", "ds_h128_k128"]
)
def test_qprep_fp64_reference_parity(h_k):
    from sglang.jit_kernel.qprep_bf16_fp8_sm90 import q8kv8_qprep_fwd
    from sglang.srt.layers.attention.triton_ops.cache_ops import (
        absorbed_bmm_concat_cast_q_fp8,
    )

    H, K = h_k
    T = 437
    q_nope, q_rope, w_kc = _make_inputs(T, H, K, seed=5678)
    ref8 = torch.zeros((T, H, N_LORA + ROPE), dtype=torch.float8_e4m3fn, device="cuda")
    out8 = torch.zeros_like(ref8)
    absorbed_bmm_concat_cast_q_fp8(ref8, q_nope, w_kc, q_rope, H, variant="two_dot")
    q8kv8_qprep_fwd(out8, q_nope, w_kc, q_rope, H)
    torch.cuda.synchronize()

    # gate (b): the CUDA kernel's fp8 must land as close to the exact bmm as
    # the Triton kernel's (same quantization noise floor, ~1.8e-2 mean).
    ref64 = torch.bmm(
        q_nope.transpose(0, 1).to(torch.float64), w_kc.to(torch.float64)
    ).transpose(0, 1)
    err_tri = (ref8[..., :N_LORA].to(torch.float64) - ref64).abs().mean().item()
    err_cuda = (out8[..., :N_LORA].to(torch.float64) - ref64).abs().mean().item()
    assert (
        err_cuda <= 1.05 * err_tri
    ), f"CUDA fp64-ref mean |err| {err_cuda:.4e} exceeds Triton's {err_tri:.4e}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
