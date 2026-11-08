"""The AWQ -> skinny int4 repack against a torch reference, and the Triton skinny W4A16 GEMM
(the Quark int4 W4A16 path on ROCm) against a dequantized matmul."""

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

rocm = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_hip()), reason="needs a ROCm GPU"
)


def _to_int32(words: torch.Tensor) -> torch.Tensor:
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _awq_pack(codes: torch.Tensor) -> torch.Tensor:
    """``codes[K, N]`` (nibbles 0..15) -> AWQ words ``[K, N / 8]``: logical column ``j`` of
    each group of eight sits at nibble ``(j % 2) * 4 + j // 2``."""
    k, n = codes.shape
    c = codes.view(k, n // 8, 8).long()
    j = torch.arange(8, device=codes.device)
    shift = ((j % 2) * 4 + j // 2) * 4
    return _to_int32((c << shift).sum(-1))


def _skinny_pack(codes_nk: torch.Tensor) -> torch.Tensor:
    """``codes[N, K]`` -> skinny words ``[N, K / 8]``: lane ``j`` of each group of eight K
    values sits at nibble ``(j % 2) * 4 + j // 2``."""
    n, k = codes_nk.shape
    c = codes_nk.view(n, k // 8, 8).long()
    j = torch.arange(8, device=codes_nk.device)
    shift = ((j % 2) * 4 + j // 2) * 4
    return _to_int32((c << shift).sum(-1))


@rocm
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("k,n", [(256, 64), (5120, 1024), (384, 136)])
def test_repack_awq_w4_to_skinny_matches_reference(signed, k, n):
    from sglang.kernels.ops.quantization.w4a16_skinny_triton import (
        repack_awq_w4_to_skinny,
    )

    torch.manual_seed(0)
    codes = torch.randint(0, 16, (k, n), device="cuda")
    out = repack_awq_w4_to_skinny(_awq_pack(codes), signed=signed)
    expected = codes ^ 8 if signed else codes  # signed codes -> zero point 8
    assert torch.equal(out, _skinny_pack(expected.t().contiguous()))


@rocm
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("m", [1, 8, 33, 300])
def test_skinny_gemm_matches_dequantized_matmul(m, padded):
    from sglang.kernels.ops.quantization.w4a16_skinny_triton import (
        pad_rows,
        repack_awq_w4_to_skinny,
        w4a16_skinny_gemm,
    )

    torch.manual_seed(0)
    k, n, g = 1024, 512, 128
    signed = torch.randint(-8, 8, (k, n), device="cuda")  # Quark: signed int4 codes
    w = repack_awq_w4_to_skinny(_awq_pack(signed & 0xF), signed=True)
    if padded:
        w = pad_rows(w)
    scales = torch.rand(n, k // g, device="cuda", dtype=torch.bfloat16) * 0.02 + 0.001
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

    out = w4a16_skinny_gemm(input=x, qweight=w, scales=scales, group_size=g)
    deq = signed.t().float() * scales.float().repeat_interleave(g, dim=1)  # [n, k]
    ref = x.float() @ deq.t()
    err = (out.float() - ref).abs().max().item() / ref.abs().max().item()
    assert err < 2e-2, err


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
