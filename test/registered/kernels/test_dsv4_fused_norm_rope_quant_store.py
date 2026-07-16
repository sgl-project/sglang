"""Correctness of the fused RMSNorm + RoPE + e4m3 quant + scatter kernel.

Reference is the unfused pipeline it replaces: fused_norm_rope_inplace_triton
followed by an e4m3 cast + index_put -- the two must produce bit-identical
cache contents (same math, same rounding points).
"""

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import (
    fused_norm_rope_inplace_triton,
    fused_norm_rope_quant_store_triton,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2, suite="nightly-1-gpu", nightly=True)

HEAD_DIM = 512
ROPE_DIM = 64


def _make_freqs(max_pos: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, device="cuda").float() / ROPE_DIM)
    )
    t = torch.arange(max_pos, device="cuda").float()
    angles = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)  # complex64 [max_pos, 32]


@pytest.mark.parametrize("num_rows", [1, 7, 128, 4096])
@pytest.mark.parametrize("has_weight", [True, False])
@pytest.mark.parametrize("scatter", ["identity", "random"])
def test_fused_norm_rope_quant_store(num_rows, has_weight, scatter):
    torch.manual_seed(num_rows)
    num_slots = max(num_rows * 4, 64)

    kv = torch.randn(num_rows, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    weight = (
        (torch.rand(HEAD_DIM, dtype=torch.bfloat16, device="cuda") + 0.5)
        if has_weight
        else None
    )
    eps = 1e-6
    freqs = _make_freqs(max_pos=8192)
    positions = torch.randint(0, 8192, (num_rows,), device="cuda", dtype=torch.int64)
    if scatter == "identity":
        loc = torch.arange(num_rows, device="cuda", dtype=torch.int64)
    else:
        loc = torch.randperm(num_slots, device="cuda", dtype=torch.int64)[:num_rows]

    # Reference: the unfused two-pass store this kernel replaces.
    ref_kv = kv.clone()
    fused_norm_rope_inplace_triton(ref_kv, weight, eps, freqs, positions=positions)
    ref_cache = torch.zeros(
        num_slots, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"
    )
    ref_cache.view(torch.uint8)[loc] = (
        ref_kv.to(torch.float8_e4m3fn).view(torch.uint8)
    )

    out_cache = torch.zeros(
        num_slots, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda"
    )
    kv_before = kv.clone()
    fused_norm_rope_quant_store_triton(
        kv, weight, eps, freqs, out_cache=out_cache, loc=loc, positions=positions
    )

    # Input must be untouched (unlike the in-place reference path).
    assert torch.equal(kv, kv_before)
    # The kernel rounds through the input dtype before the e4m3 cast, so it
    # is bit-identical to the unfused reference.
    assert torch.equal(ref_cache.view(torch.uint8), out_cache.view(torch.uint8))
    # And the untouched slots stay zero.
    touched = torch.zeros(num_slots, dtype=torch.bool, device="cuda")
    touched[loc] = True
    assert out_cache.view(torch.uint8)[~touched].abs().sum().item() == 0


@pytest.mark.parametrize("num_rows", [0])
def test_fused_norm_rope_quant_store_empty(num_rows):
    kv = torch.empty(0, HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    freqs = _make_freqs(16)
    out_cache = torch.zeros(8, HEAD_DIM, dtype=torch.float8_e4m3fn, device="cuda")
    loc = torch.empty(0, dtype=torch.int64, device="cuda")
    positions = torch.empty(0, dtype=torch.int64, device="cuda")
    fused_norm_rope_quant_store_triton(
        kv, None, 1e-6, freqs, out_cache=out_cache, loc=loc, positions=positions
    )
    assert out_cache.view(torch.uint8).abs().sum().item() == 0
