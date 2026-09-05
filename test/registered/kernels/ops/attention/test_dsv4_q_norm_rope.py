"""Registered CI test for the DSv4 fused Q norm+RoPE kernel (fp8 + bf16).

Covers the fp8 optimization of `fused_q_norm_rope` (block-per-token freq
sharing + shape dispatch). Validates against a pure-torch fp32 reference with
per-dtype tolerances, exercising BOTH launcher dispatch paths (small-N
warp-per-work and large-N block-per-token, crossover at total_works=4096) and
the block-per-token remainder-block boundary (num_q_heads not a multiple of 8).

Run:
    export PYTHONPATH=<repo>/python:$PYTHONPATH
    python -m pytest test/registered/kernels/ops/attention/test_dsv4_q_norm_rope.py -v
"""

import torch

from sglang.kernels.ops.attention.dsv4 import fused_q_norm_rope
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
EPS = 1e-6
MAX_POS = 4096

# Per-dtype tolerance. bf16: 8-bit mantissa -> 2e-2 (project parity tol).
# fp8_e4m3: 3-bit mantissa -> golden<->kernel disagree by up to ~1 ULP, whose
# relative step near magnitude 1 is 2^-3; 1e-1 covers it. NaN/Inf checked
# separately. These are the same tolerances the kernel's own harness uses.
_TOL = {torch.bfloat16: (2e-2, 2e-2), torch.float8_e4m3fn: (1e-1, 1e-1)}


def _golden(q_input, freqs_real, positions, tdt):
    """Pure-torch fp32 reference, matching the kernel's two-round behavior:
    normalize + round to DType (incl. rope tile), then rotate the rope tail and
    round again. RMSNorm-self over 512 dims, NO weight vector."""
    x = q_input.float()
    ss = (x * x).sum(-1, keepdim=True)
    norm = torch.rsqrt(ss / HEAD_DIM + EPS)
    xn = (x * norm).to(tdt).float()  # round every element (rope tile included)

    pos = positions.long()
    freqs = freqs_real.index_select(0, pos)  # [B,64] interleaved (re,im,...)
    cos = freqs[:, 0::2][:, None, :]
    sin = freqs[:, 1::2][:, None, :]
    tail = xn[:, :, NOPE_DIM:]
    re, im = tail[:, :, 0::2], tail[:, :, 1::2]
    nr = re * cos - im * sin
    ni = re * sin + im * cos
    ntail = torch.stack([nr, ni], dim=-1).flatten(-2)
    y = torch.cat([xn[:, :, :NOPE_DIM], ntail], dim=-1)
    return y.to(tdt)


def _run_case(num_tokens, num_q_heads, dtype, pos_dtype):
    torch.manual_seed(0)
    dev = "cuda"
    q_input = torch.randn(
        num_tokens, num_q_heads, HEAD_DIM, dtype=torch.float32, device=dev
    ).to(dtype)
    q_output = torch.empty_like(q_input)

    angles = torch.rand(MAX_POS, ROPE_DIM // 2, device=dev) * 6.2831853
    freqs_cis = torch.polar(torch.ones_like(angles), angles)  # complex64
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2).contiguous()

    positions = torch.randint(0, MAX_POS, (num_tokens,), dtype=pos_dtype, device=dev)

    fused_q_norm_rope(q_input, q_output, EPS, freqs_cis, positions)
    torch.cuda.synchronize()

    got = q_output.float()
    exp = _golden(q_input, freqs_real, positions, dtype).float()

    n_nan = torch.isnan(got).sum().item()
    n_inf = torch.isinf(got).sum().item()
    assert n_nan == 0 and n_inf == 0, f"NaN={n_nan} Inf={n_inf}"

    rtol, atol = _TOL[dtype]
    torch.testing.assert_close(got, exp, rtol=rtol, atol=atol)


# (num_tokens, num_q_heads) chosen to cover both dispatch paths + boundaries.
# total_works = N*H; fp8 block-per-token when total_works >= 4096.
_SHAPES = [
    (17, 17),  # 289  : warp-per-work, total_works%4=1 tail warp
    (256, 64),  # 16384: block-per-token
    (1024, 64),  # 65536: block-per-token (perf sweet spot)
    (4096, 64),  # big  : block-per-token
    (63, 64),  # 4032 : just below threshold -> warp-per-work
    (64, 64),  # 4096 : exactly at threshold -> block-per-token
    (
        1024,
        17,
    ),  # H%8!=0: block-per-token remainder block (ceil(17/8)=3, last covers 1 head)
    (256, 32),  # TP=2 tier
    (512, 16),  # TP=4 tier
    (8, 64),  # 512  : small-N decode, warp-per-work
]


class TestDSv4QNormRope(CustomTestCase):
    def test_fp8_correctness(self):
        for n, h in _SHAPES:
            for pos_dtype in (torch.int32, torch.int64):
                with self.subTest(N=n, H=h, pos=str(pos_dtype)):
                    _run_case(n, h, torch.float8_e4m3fn, pos_dtype)

    def test_bf16_correctness(self):
        for n, h in _SHAPES:
            for pos_dtype in (torch.int32, torch.int64):
                with self.subTest(N=n, H=h, pos=str(pos_dtype)):
                    _run_case(n, h, torch.bfloat16, pos_dtype)


if __name__ == "__main__":
    import unittest

    unittest.main()
