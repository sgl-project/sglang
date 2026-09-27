"""Fused inverse-RoPE + WO-A + MXFP8 against a torch reference."""

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.wo_a import fused_rope_wo_a_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="wo_a_fused requires SM100/SM103",
)

LOCAL_HEADS = 16
HEAD_DIM = 512
ROPE_DIM = 64
MAX_POS = 4096
N_OUT = 2048


def _inputs(tokens: int, padded: bool):
    device = "cuda"
    torch.manual_seed(1234)
    heads = 64 if padded else LOCAL_HEADS
    backing = torch.randn(tokens, heads, HEAD_DIM, dtype=torch.bfloat16, device=device)
    o = backing[:, :LOCAL_HEADS, :]
    weight = torch.randn(2, 1024, 4096, dtype=torch.bfloat16, device=device).mul_(0.05)
    angles = torch.outer(
        torch.arange(MAX_POS, device=device, dtype=torch.float32),
        1.0 / 10000.0 ** (torch.arange(32, device=device, dtype=torch.float32) / 32),
    )
    freqs = torch.stack([angles.cos(), angles.sin()], dim=-1).reshape(MAX_POS, ROPE_DIM)
    positions = torch.randint(0, MAX_POS, (tokens,), dtype=torch.int32, device=device)
    return o, weight, freqs, positions


def _reference(o, weight, freqs, positions):
    """Inverse RoPE (fp32, rounded to bf16) then the grouped GEMM in fp64."""
    rotated = o.float().clone()
    f = freqs[positions.long()]  # [T, 64] interleaved (cos0, sin0, ...)
    cos, sin = f[:, 0::2], f[:, 1::2]  # [T, 32]
    tail = rotated[..., -ROPE_DIM:]
    # Clone first: these are views into `tail`, so writing the real part in
    # place would feed the updated values into the imaginary part.
    re = tail[..., 0::2].clone()
    im = tail[..., 1::2].clone()
    tail[..., 0::2] = (re * cos[:, None, :] + im * sin[:, None, :]).bfloat16().float()
    tail[..., 1::2] = (im * cos[:, None, :] - re * sin[:, None, :]).bfloat16().float()
    x = rotated.reshape(o.shape[0], 2, -1).double()
    return torch.einsum("tgd,grd->tgr", x, weight.double()).reshape(o.shape[0], N_OUT)


def _ue8m0(amax: torch.Tensor):
    """Bit-identical to the kernel's ue8m0_scale / flashinfer's mxfp8_quantize."""
    bits = (amax / 448.0).float().view(torch.int32)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = torch.where(
        bits == 0, torch.zeros_like(exponent), (exponent + bump).clamp(max=254)
    )
    inv = torch.where(bits == 0, torch.zeros_like(sf), (254 - sf) << 23).view(
        torch.float32
    )
    return sf.to(torch.uint8), inv


def _sf_index(block: torch.Tensor, row: torch.Tensor) -> torch.Tensor:
    return (block >> 2) * 512 + ((row % 32) * 4 + ((row // 32) % 4)) * 4 + (block & 3)


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 13, 16, 17, 32])
@pytest.mark.parametrize("padded", [False, True])
def test_matches_reference(tokens: int, padded: bool):
    o, weight, freqs, positions = _inputs(tokens, padded)
    grouped = o.view(tokens, 2, -1)
    # The two output forms are mutually exclusive, so this also checks that the
    # bf16 path and the quantized path agree on the same inputs.
    (y,) = fused_rope_wo_a_bf16(grouped, weight, freqs, positions, out_mxfp8=False)
    q, scales = fused_rope_wo_a_bf16(grouped, weight, freqs, positions)

    ref = _reference(o, weight, freqs, positions)
    # bf16 accumulation noise only; the GEMM carries no exactness contract.
    err = (y.float() - ref.float()).norm() / ref.float().norm()
    assert err < 1e-2, f"relative error {err:.3g}"

    # The quantizer does: quantize(y) must match bit for bit.
    yf = y.float().reshape(tokens, N_OUT // 32, 32)
    sf, inv = _ue8m0(yf.abs().amax(-1))
    want = (yf * inv[..., None]).clamp(max=448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        q.reshape(tokens, -1, 32).float(), want.float(), rtol=0, atol=0
    )

    blocks = torch.arange(N_OUT // 32)
    rows = torch.arange(tokens)[:, None]
    torch.testing.assert_close(
        scales.cpu()[_sf_index(blocks, rows)], sf.cpu(), rtol=0, atol=0
    )


def test_scale_padding_is_zeroed():
    """Rows >= T must be zeroed every launch: the buffer is reused by CUDA graphs."""
    tokens = 4
    o, weight, freqs, positions = _inputs(tokens, padded=False)
    _, scales = fused_rope_wo_a_bf16(o.view(tokens, 2, -1), weight, freqs, positions)
    blocks = torch.arange(N_OUT // 32)
    rows = torch.arange(tokens, 128)[:, None]
    assert torch.count_nonzero(scales.cpu()[_sf_index(blocks, rows)]) == 0


def test_int64_positions_match_int32():
    """DSV4 decode supplies int32 positions but the DSpark verify path supplies
    int64; accepting only int32 made the server fail CUDA graph capture."""
    tokens = 6
    o, weight, freqs, positions = _inputs(tokens, padded=True)
    grouped = o.view(tokens, 2, -1)
    q32, s32 = fused_rope_wo_a_bf16(grouped, weight, freqs, positions)
    q64, s64 = fused_rope_wo_a_bf16(grouped, weight, freqs, positions.to(torch.int64))
    assert torch.equal(q32, q64), "int64 positions changed the quantized output"
    assert torch.equal(s32, s64), "int64 positions changed the scales"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
