"""Equivalence of the fused qkv+index GEMM with two separate GEMMs (mxfp8).

The MiniMax-M3 fused projection concatenates the main ``qkv_proj`` and sparse
``index_qkv_proj`` weights (and raw UE8M0 scales) along the output dim and runs
one mxfp8 deep_gemm matmul. Because each output row is independent and the
activation is quantized once from the shared input, the fused output must equal
the row-concatenation of the two separate outputs -- bit for bit. This drives
the exact production apply (``_deepgemm_w8a8_mxfp8_linear_with_fallback``) plus
the exact scale-packing used by ``_process_mxfp8_linear_weight_scale``.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-b200")

dev = "cuda"


def _pack_weight_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Mirror Fp8LinearMethod._process_mxfp8_linear_weight_scale (deep_gemm)."""
    from sglang.srt.layers.deep_gemm_wrapper.configurer import DEEPGEMM_SCALE_UE8M0

    n, kk = scale_u8.shape
    scale_fp32 = (
        (scale_u8.contiguous().view(-1).to(torch.int32) << 23)
        .view(torch.float32)
        .view(n, kk)
    )
    if DEEPGEMM_SCALE_UE8M0:
        import deep_gemm.utils.layout

        return deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor(
            scale_fp32
        )
    return scale_fp32


@pytest.mark.parametrize("T", [1, 7, 64, 256, 1024])
@pytest.mark.parametrize("N1,N2,K", [(1280, 256, 6144), (1024, 128, 2048)])
def test_fused_equals_separate(T, N1, N2, K):
    from sglang.srt.layers.quantization.fp8_utils import (
        _deepgemm_w8a8_mxfp8_linear_with_fallback as mxfp8_linear,
    )

    torch.manual_seed(T * 7 + N1 + K)
    G = K // 32

    def rand_w(n):
        return (torch.randn(n, K, device=dev) * 0.2).to(torch.float8_e4m3fn)

    def rand_s(n):
        # mid-range UE8M0 exponents (~2^-3..2^3) to avoid inf/zero blowups.
        return torch.randint(124, 131, (n, G), device=dev, dtype=torch.uint8)

    w1, w2 = rand_w(N1), rand_w(N2)
    s1, s2 = rand_s(N1), rand_s(N2)
    x = torch.randn(T, K, dtype=torch.bfloat16, device=dev)

    s1p, s2p = _pack_weight_scale(s1), _pack_weight_scale(s2)
    out1 = mxfp8_linear(x, w1, s1p, weight_scale_fallback=s1)
    out2 = mxfp8_linear(x, w2, s2p, weight_scale_fallback=s2)
    ref = torch.cat([out1, out2], dim=-1)

    w = torch.cat([w1, w2], dim=0).contiguous()
    s = torch.cat([s1, s2], dim=0).contiguous()
    sp = _pack_weight_scale(s)
    fused = mxfp8_linear(x, w, sp, weight_scale_fallback=s)

    assert fused.shape == ref.shape
    assert torch.equal(
        fused, ref
    ), f"max abs diff {(fused.float() - ref.float()).abs().max().item()}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
