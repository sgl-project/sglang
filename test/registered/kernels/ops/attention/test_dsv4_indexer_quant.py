"""Correctness tests for the DeepSeek-V4 DSA indexer fp8-quant Q kernel and its
V3.2/GLM rope-first variant, after the grid-stride + occupancy scheduling
optimization of ``fused_q_indexer_rope_hadamard_quant``.

Covers both template configs that share the kernel:
  - fused_q_indexer_rope_hadamard_quant  (V4: rope on trailing 64 dims + 128-pt
                                           Hadamard + dynamic fp8-e4m3 quant)
  - fused_q_indexer_rope_first_quant      (V3.2/GLM: rope on leading 64 dims,
                                           no Hadamard, + fp8-e4m3 quant)

Each kernel is checked against a torch reference (dequantized q within fp8-e4m3
precision; weights_out to tight tolerance). Multiple batch sizes exercise both
the straight-line (small/medium batch) and grid-stride (large batch) launch
branches introduced by the scheduling optimization.
"""

from __future__ import annotations

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import (
    fused_q_indexer_rope_first_quant,
    fused_q_indexer_rope_hadamard_quant,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

_is_hip = is_hip()

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=45, suite="jit-kernel-unit-test-amd")

HEAD_DIM = 128
ROPE_DIM = 64
HALF = ROPE_DIM // 2
FP8_MAX = 448.0
MAX_POS = 8192
N_HEADS = 64

# Batch sizes spanning both launch branches. Grid-stride kicks in once
# rows_blocks (=B*H/kNumWarps) exceeds one full wave (num_sm * kBlocksPerSM);
# the large sizes here (>=512) are firmly in the grid-stride regime on any GPU,
# the small ones exercise the straight-line branch.
BATCHES = [1, 8, 64, 256, 512, 2048]


def _skip_if_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if _is_hip:
        pytest.skip("Indexer fused Q kernel is CUDA-specific")


def _hadamard_matrix(n, device):
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.to(device=device, dtype=torch.float32)


def _fp8_dequant_ok(q_fp8, ref, scale):
    """fp8-e4m3 round-to-nearest: <= 1/16 relative + one scale step at the
    bottom of the range."""
    deq = q_fp8.float() * scale
    err = (deq - ref).abs()
    return (err <= 0.0625 * ref.abs() + scale).all()


# ----------------------------------------------------------------------------
# V4 path: rope on trailing 64 dims (interleaved) + 128-pt Hadamard + fp8 quant
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("batch", BATCHES)
def test_v4_rope_hadamard_quant_matches_reference(batch, pos_dtype):
    _skip_if_unavailable()
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(
        batch, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g
    )
    weight = torch.randn(batch, N_HEADS, dtype=torch.bfloat16, device=dev, generator=g)
    weight_scale = 0.137
    angles = torch.rand(MAX_POS, HALF, device=dev, generator=g) * 6.2831853
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex, (max_pos, 32)
    positions = torch.randint(
        0, 4096, (batch,), device=dev, dtype=pos_dtype, generator=g
    )

    q_fp8, weights_out = fused_q_indexer_rope_hadamard_quant(
        q, weight, weight_scale, freqs_cis, positions
    )
    torch.cuda.synchronize()

    # Reference: rope (trailing 64, interleaved) -> 128-pt Hadamard * rsqrt(128)
    # -> per-(token,head) abs-max fp8-e4m3 dynamic quant.
    qf = q.float()
    fc = freqs_cis[positions.long()]
    cos = fc.real[:, None, :]
    sin = fc.imag[:, None, :]
    tail = qf[..., ROPE_DIM:]
    re, im = tail[..., 0::2], tail[..., 1::2]
    ntail = torch.stack([re * cos - im * sin, re * sin + im * cos], dim=-1).flatten(-2)
    qrot = torch.cat([qf[..., :ROPE_DIM], ntail], dim=-1)
    y = torch.matmul(qrot, _hadamard_matrix(HEAD_DIM, dev)) * (HEAD_DIM**-0.5)
    scale = torch.clamp(y.abs().amax(dim=-1, keepdim=True), min=1e-4) / FP8_MAX

    w_ref = weight.float() * weight_scale * scale.squeeze(-1)
    torch.testing.assert_close(weights_out.squeeze(-1), w_ref, atol=1e-3, rtol=1e-3)
    assert _fp8_dequant_ok(q_fp8, y, scale), "V4 fp8 dequant error out of tolerance"
    assert torch.isfinite(q_fp8.float()).all() and torch.isfinite(weights_out).all()


# ----------------------------------------------------------------------------
# V3.2/GLM path: rope on leading 64 dims (interleaved), NO Hadamard, + fp8 quant
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("batch", BATCHES)
def test_v32_rope_first_quant_matches_reference(batch):
    _skip_if_unavailable()
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(
        batch, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g
    )
    weight = torch.randn(batch, N_HEADS, dtype=torch.bfloat16, device=dev, generator=g)
    weight_scale = 0.137
    cos = torch.randn(MAX_POS, HALF, device=dev, generator=g)
    sin = torch.randn(MAX_POS, HALF, device=dev, generator=g)
    cos_sin_cache = torch.cat((cos, sin), dim=-1)  # (max_pos, 64)
    positions = torch.randint(
        0, 4096, (batch,), device=dev, dtype=torch.int32, generator=g
    )

    q_fp8, weights_out = fused_q_indexer_rope_first_quant(
        q, weight, weight_scale, cos_sin_cache, positions
    )
    torch.cuda.synchronize()

    # Reference: rope on leading 64 dims (interleaved), no Hadamard.
    qf = q.float()
    cp = cos[positions.long()][:, None, :]
    sp = sin[positions.long()][:, None, :]
    ref = qf.clone()
    xr = qf[..., 0:ROPE_DIM:2]
    xi = qf[..., 1:ROPE_DIM:2]
    ref[..., 0:ROPE_DIM:2] = xr * cp - xi * sp
    ref[..., 1:ROPE_DIM:2] = xr * sp + xi * cp
    scale = torch.clamp(ref.abs().amax(dim=-1, keepdim=True), min=1e-4) / FP8_MAX

    w_ref = weight.float() * weight_scale * scale.squeeze(-1)
    torch.testing.assert_close(weights_out.squeeze(-1), w_ref, atol=1e-3, rtol=1e-3)
    assert _fp8_dequant_ok(q_fp8, ref, scale), "V3.2 fp8 dequant error out of tolerance"
    assert torch.isfinite(q_fp8.float()).all() and torch.isfinite(weights_out).all()


# ----------------------------------------------------------------------------
# Strided weight (the non-contiguous wk slice) matches contiguous (V4 path).
# ----------------------------------------------------------------------------
def test_v4_strided_weight_matches_contiguous():
    _skip_if_unavailable()
    dev = "cuda"
    B = 512  # grid-stride regime
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(B, N_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    angles = torch.rand(MAX_POS, HALF, device=dev, generator=g) * 6.2831853
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randint(0, 4096, (B,), device=dev, dtype=torch.int32, generator=g)
    kw = torch.randn(
        B, HEAD_DIM + N_HEADS, dtype=torch.bfloat16, device=dev, generator=g
    )
    w_strided = kw[:, HEAD_DIM:]
    w_contig = w_strided.contiguous()
    assert not w_strided.is_contiguous()

    a_fp8, a_w = fused_q_indexer_rope_hadamard_quant(
        q, w_strided, 0.137, freqs_cis, positions
    )
    b_fp8, b_w = fused_q_indexer_rope_hadamard_quant(
        q, w_contig, 0.137, freqs_cis, positions
    )
    torch.cuda.synchronize()
    assert torch.equal(a_fp8, b_fp8)
    assert torch.equal(a_w, b_w)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
