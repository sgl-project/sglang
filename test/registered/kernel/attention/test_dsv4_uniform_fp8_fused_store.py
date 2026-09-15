"""Correctness of the uniform-FP8 fused store (CUDA FusedNormRopeKernel).

The kernel fuses RMSNorm + RoPE + plain e4m3 cast (per-tensor scale 1.0) +
paged scatter into the 512-byte-per-token uniform pool, reading positions,
destinations, decode window boundaries and prefill-row validity from the
compress plan. Reference is the unfused pipeline it replaces
(fused_norm_rope_inplace_triton, then an e4m3 cast + index_put) with the
plan semantics emulated in Python.

The CUDA block reduction sums the RMSNorm squares in a different order than
the Triton reference, so a ~1e-6 fraction of elements can land one e4m3 ulp
apart; the assertions allow exactly that and nothing more.
"""

import pytest
import torch

from sglang.kernels.ops.attention.deepseek_v4_rope import fused_norm_rope_inplace_triton
from sglang.kernels.ops.attention.dsv4.compress import (
    CompressorDecodePlan,
    CompressorPrefillPlan,
    compress_norm_rope_store,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

HEAD_DIM = 512
ROPE_DIM = 64
RATIO = 4
PAGE_SIZE = 64


def _make_freqs(max_pos: int) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, device="cuda").float() / ROPE_DIM)
    )
    t = torch.arange(max_pos, device="cuda").float()
    angles = torch.outer(t, inv_freq)
    return torch.polar(torch.ones_like(angles), angles)  # complex64 [max_pos, 32]


def _reference_rows(kv, weight, eps, freqs, positions):
    x = kv.clone()
    fused_norm_rope_inplace_triton(x, weight, eps, freqs, positions=positions)
    return x.to(torch.float8_e4m3fn)


def _assert_rows_match(got_fp8, ref_fp8):
    got, ref = got_fp8.float(), ref_fp8.float()
    mismatch = (got_fp8.view(torch.uint8) != ref_fp8.view(torch.uint8)).float().mean()
    assert mismatch.item() <= 1e-4, f"{mismatch.item()=}"
    # any mismatches must be within one e4m3 ulp (<= 2^-3 relative)
    denom = ref.abs().clamp(min=2**-9)
    assert ((got - ref).abs() / denom).max().item() <= 0.13


@pytest.mark.parametrize("num_rows", [1, 7, 128, 2048])
def test_uniform_fp8_store_decode_plan(num_rows):
    torch.manual_seed(num_rows)
    kv = torch.randn(num_rows, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 2
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda").abs() + 0.5
    eps = 1e-6
    seq = torch.randint(1, 8192, (num_rows,), device="cuda", dtype=torch.int32)
    # half the rows sit on a window boundary (stored), half not (skipped)
    seq[::2] = (seq[::2] // RATIO).clamp(min=1) * RATIO
    seq[1::2] = (seq[1::2] // RATIO).clamp(min=1) * RATIO + 1
    plan_i32 = torch.zeros(num_rows, 4, dtype=torch.int32, device="cuda")
    plan_i32[:, 0] = seq
    plan = CompressorDecodePlan(RATIO, plan_i32.view(torch.uint8))
    freqs = _make_freqs(8192 + RATIO)
    out_loc = torch.randperm(num_rows * 2, device="cuda")[:num_rows].to(torch.int64)

    num_pages = (num_rows * 2 + PAGE_SIZE - 1) // PAGE_SIZE + 1
    cache = torch.zeros(
        num_pages, PAGE_SIZE * HEAD_DIM, dtype=torch.uint8, device="cuda"
    )
    compress_norm_rope_store(
        kv,
        plan,
        norm_weight=weight,
        norm_eps=eps,
        freq_cis=freqs,
        out_loc=out_loc,
        kvcache=cache,
        page_size=PAGE_SIZE,
        uniform_fp8_store=True,
    )

    boundary = (seq % RATIO) == 0
    positions = (seq - RATIO).to(torch.int32)
    ref = _reference_rows(kv, weight, eps, freqs, positions)
    rows = cache.view(-1, HEAD_DIM)
    got = rows[out_loc[boundary]].view(torch.float8_e4m3fn)
    _assert_rows_match(got, ref[boundary])
    # non-boundary rows must not be written (their slots stay zero)
    skipped = rows[out_loc[~boundary]]
    assert (skipped == 0).all(), "non-boundary decode rows must be skipped"


@pytest.mark.parametrize("num_rows", [8, 300, 1024])
def test_uniform_fp8_store_prefill_plan_with_invalid_rows(num_rows):
    torch.manual_seed(num_rows)
    kv = torch.randn(num_rows, HEAD_DIM, dtype=torch.bfloat16, device="cuda") * 2
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device="cuda").abs() + 0.5
    eps = 1e-6
    seq = (torch.randint(1, 2048, (num_rows,), device="cuda") * RATIO).to(torch.int32)
    ragged = torch.randperm(num_rows, device="cuda").to(torch.int32)
    valid = torch.rand(num_rows, device="cuda") > 0.25
    plan_i32 = torch.zeros(num_rows, 4, dtype=torch.int32, device="cuda")
    plan_i32[:, 0] = torch.where(valid, seq, torch.full_like(seq, -1))
    plan_i32[:, 1] = torch.where(valid, ragged, torch.full_like(ragged, 0xFFFF))
    plan_c = plan_i32.view(torch.uint8)
    plan_w = torch.zeros(num_rows, 8, dtype=torch.uint8, device="cuda")
    plan = CompressorPrefillPlan(RATIO, plan_c, plan_w)
    freqs = _make_freqs(2048 * RATIO + RATIO)
    out_loc = torch.randperm(num_rows * 2, device="cuda")[:num_rows].to(torch.int64)

    num_pages = (num_rows * 2 + PAGE_SIZE - 1) // PAGE_SIZE + 1
    cache = torch.zeros(
        num_pages, PAGE_SIZE * HEAD_DIM, dtype=torch.uint8, device="cuda"
    )
    compress_norm_rope_store(
        kv,
        plan,
        norm_weight=weight,
        norm_eps=eps,
        freq_cis=freqs,
        out_loc=out_loc,
        kvcache=cache,
        page_size=PAGE_SIZE,
        uniform_fp8_store=True,
    )

    positions = (seq - RATIO).clamp(min=0).to(torch.int32)
    ref = _reference_rows(kv, weight, eps, freqs, positions)
    rows = cache.view(-1, HEAD_DIM)
    got = rows[out_loc[ragged[valid].long()]].view(torch.float8_e4m3fn)
    _assert_rows_match(got, ref[valid])
    # invalid rows are skipped entirely: every slot not mapped by a valid row
    # stays zero
    written = torch.zeros(rows.shape[0], dtype=torch.bool, device="cuda")
    written[out_loc[ragged[valid].long()]] = True
    assert (rows[~written] == 0).all(), "invalid prefill rows must be skipped"
