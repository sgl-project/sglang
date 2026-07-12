"""Correctness tests for the trait-driven per_token_group_quant_v3 JIT kernel.

The reference is computed in pure PyTorch (the quantization math itself), NOT by
calling the v2 / minimax kernels -- those are being deprecated, so the tests
must outlive them.

Two guard strengths, chosen by what the kernel's numerics can actually pin:
  - UE8M0 paths: the quant multiplier is an exact power of two (a bit shift, no
    division), so codes and packed exponent bytes are compared BIT-EXACT
    against the torch reference. These are the production paths (DeepGEMM dense,
    EP-MoE), so this is where bit-exactness matters.
  - fp32 / int8 scale paths: the kernel divides under ``--use_fast_math`` (fast
    reciprocal), so codes are not bit-reproducible from an exact torch divide.
    Those tests pin the exactly-reproducible parts -- the stored scale (a single
    multiply) -- and the dequant round-trip error, which is what downstream
    actually consumes.
"""

import itertools

import pytest
import torch

from sglang.jit_kernel.per_token_group_quant_v3 import per_token_group_quant_v3
from sglang.jit_kernel.utils import get_ci_test_range
from sglang.srt.layers.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_dtype,
    fp8_max,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

G = 128
FMAX = float(fp8_max)  # 448 for e4m3
I8_MAX, I8_MIN = 127.0, -128.0
EPS = 1e-10


# --------------------------------------------------------------------------- #
# Pure-torch references (match the kernel's expression order).
# --------------------------------------------------------------------------- #
def _group_amax(x: torch.Tensor, gs: int) -> torch.Tensor:
    """Per-group absmax over the last dim, floored at EPS. Returns [..., ng]."""
    xf = x.float().unflatten(-1, (-1, gs))
    return xf.abs().amax(-1).clamp_min(EPS)


def _quantize(x: torch.Tensor, gs: int, quant_scale: torch.Tensor, out_dtype, lo, hi):
    xf = x.float().unflatten(-1, (-1, gs))
    q = (xf * quant_scale.unsqueeze(-1)).clamp(lo, hi).to(out_dtype)
    return q.flatten(-2)


def ref_fp8_fp32_scale(x, gs):
    """fp8 codes + fp32 stored scale (scale = amax / FMAX, a single multiply)."""
    amax = _group_amax(x, gs)
    scale_inv = amax * (1.0 / FMAX)
    q = _quantize(x, gs, FMAX / amax, fp8_dtype, -FMAX, FMAX)
    return q, scale_inv


def ref_int8(x, gs):
    amax = _group_amax(x, gs)
    scale_inv = amax * (1.0 / I8_MAX)
    q = _quantize(x, gs, I8_MAX / amax, torch.int8, I8_MIN, I8_MAX)
    return q, scale_inv


def ref_fp8_ue8m0(x, gs):
    """fp8 codes + UE8M0 exponent bytes [..., ng]. The multiplier 2^-e is exact
    in fp32, so codes are bit-reproducible (unlike the fp32-scale path)."""
    amax = _group_amax(x, gs)
    raw = (amax / FMAX).contiguous()
    bits = raw.view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(
        torch.int32
    )  # ceil to ue8m0
    quant_scale = ((127 + 127 - exp) << 23).view(torch.float32)  # 2^(127 - (exp-127))
    q = _quantize(x, gs, quant_scale, fp8_dtype, -FMAX, FMAX)
    return q, exp.to(torch.uint8)


def _decode_packed_exp(s_int32: torch.Tensor, ng: int) -> torch.Tensor:
    """Decode an int32 packed-UE8M0 scale (logical [..., ceil(ng/4)]) to the
    [..., ng] exponent grid, independent of the physical (row/col-major) layout:
    exponent[..., g] = byte (g % 4) of int32[..., g // 4]."""
    g = torch.arange(ng, device=s_int32.device)
    col = s_int32.index_select(-1, g // 4)
    return ((col >> (8 * (g % 4))) & 0xFF).to(torch.uint8)


def _dequant_rel_err(q, scale_inv, x, gs) -> float:
    deq = (q.float().unflatten(-1, (-1, gs)) * scale_inv.unsqueeze(-1)).flatten(-2)
    return ((x.float() - deq).abs() / (x.float().abs() + 1e-6)).mean().item()


def _packed_exp_to_dequant_scale(x_s, ng) -> torch.Tensor:
    """Decode a packed-UE8M0 scale buffer to the fp32 dequant scale 2^(e-127)."""
    exp = _decode_packed_exp(x_s, ng).to(torch.int32)
    return torch.exp2(exp.float() - 127.0)


def _alloc_scale(x_shape, *, column_major, scale_ue8m0):
    s = create_per_token_group_quant_fp8_output_scale(
        x_shape=x_shape,
        device="cuda",
        group_size=G,
        column_major_scales=column_major,
        scale_tma_aligned=column_major,
        scale_ue8m0=scale_ue8m0,
    )
    s.zero_()
    return s


# --------------------------------------------------------------------------- #
# UE8M0 paths: bit-exact vs the torch reference.
# --------------------------------------------------------------------------- #
# hidden 768 (Qwen3-30B-A3B moe_intermediate: 6 groups) exercises the non-4
# aligned col-packed tail; 128 is a single group.
UE8M0_CASES = get_ci_test_range(
    list(
        itertools.product(
            [torch.bfloat16, torch.float16],
            [1, 7, 38, 333],
            [128, 768, 2048, 7168],
        )
    ),
    [
        (torch.bfloat16, 1, 128),
        (torch.bfloat16, 38, 768),
        (torch.bfloat16, 333, 7168),
        (torch.float16, 7, 2048),
    ],
)


@pytest.mark.parametrize("dtype,num_tokens,hidden", UE8M0_CASES)
def test_v3_ue8m0_bitexact(dtype, num_tokens, hidden):
    """Col-major packed UE8M0: fp8 codes and decoded exponent bytes are
    bit-exact with the torch reference (exact pow-2 multiplier). Covers the
    aligned and non-4-aligned (hidden=768) pack-tail layouts."""
    torch.manual_seed(hidden * 10 + num_tokens)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype)
    q_ref, exp_ref = ref_fp8_ue8m0(x, G)

    x_q = torch.zeros_like(x, dtype=fp8_dtype)
    x_s = _alloc_scale((num_tokens, hidden), column_major=True, scale_ue8m0=True)
    per_token_group_quant_v3(x, x_q, x_s, G, scale_ue8m0=True)
    torch.cuda.synchronize()

    assert torch.equal(x_q.view(torch.int8), q_ref.view(torch.int8)), "codes differ"
    exp = _decode_packed_exp(x_s, hidden // G)
    assert torch.equal(exp, exp_ref), "exponent bytes differ"


@pytest.mark.parametrize("group_size", get_ci_test_range([16, 32, 64, 128], [16, 64]))
def test_v3_ue8m0_group_sizes(group_size):
    """Group size is a template axis (v2 dispatched a runtime switch). Each size
    maps a group onto a different subwarp lane count; codes/exponents must stay
    bit-exact -- a wrong lane span would fold the wrong elements into absmax."""
    torch.manual_seed(group_size)
    num_tokens, hidden = 9, 4096
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    q_ref, exp_ref = ref_fp8_ue8m0(x, group_size)

    x_q = torch.zeros_like(x, dtype=fp8_dtype)
    x_s = create_per_token_group_quant_fp8_output_scale(
        x_shape=(num_tokens, hidden),
        device="cuda",
        group_size=group_size,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    x_s.zero_()
    per_token_group_quant_v3(x, x_q, x_s, group_size, scale_ue8m0=True)
    torch.cuda.synchronize()

    assert torch.equal(x_q.view(torch.int8), q_ref.view(torch.int8)), "codes differ"
    exp = _decode_packed_exp(x_s, hidden // group_size)
    assert torch.equal(exp, exp_ref), "exponent bytes differ"


# hidden 4096 -> 32 groups (aligned); 768 -> 6 groups (6 % 4 = 2, unaligned:
# the last int32 holds 2 real exponent bytes + 2 zero-padded tail bytes).
@pytest.mark.parametrize("hidden", [4096, 768])
def test_v3_ue8m0_row_packed_bitexact(hidden):
    """Row-major packed UE8M0 (int32 [T, ceil(G/4)] contiguous, the minimax
    layout): bit-exact vs the torch reference. The unaligned hidden exercises
    the row-major pack-tail zeroing (fill_unaligned)."""
    torch.manual_seed(hidden)
    num_tokens = 17
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    q_ref, exp_ref = ref_fp8_ue8m0(x, G)

    x_q = torch.zeros_like(x, dtype=fp8_dtype)
    x_s = torch.zeros(
        num_tokens, (hidden // G + 3) // 4, device="cuda", dtype=torch.int32
    )
    per_token_group_quant_v3(x, x_q, x_s, G, scale_ue8m0=True)
    torch.cuda.synchronize()

    assert torch.equal(x_q.view(torch.int8), q_ref.view(torch.int8)), "codes differ"
    exp = _decode_packed_exp(x_s, hidden // G)
    assert torch.equal(exp, exp_ref), "exponent bytes differ"
    # unaligned tail bytes of the last int32 must be zero-padded, not garbage.
    ng = hidden // G
    if ng % 4:
        last_bytes = x_s[:, -1].contiguous().view(torch.uint8).view(num_tokens, 4)
        assert torch.all(last_bytes[:, ng % 4 :] == 0), "pack-tail bytes not zeroed"


# --------------------------------------------------------------------------- #
# fp32 / int8 scale paths: exact stored scale + dequant round-trip (the codes
# are not bit-reproducible under fast-math division).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("hidden", [4096, 768])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_v3_fp32_scale(dtype, column_major, hidden):
    """fp32 scale (row-major contiguous / col-major TMA view): the stored scale
    is amax/FMAX (a single multiply, bit-exact) and dequant round-trips within
    fp8 error.

    hidden=768 (6 groups, ng % 4 != 0) is a bug regression: the host check
    used to apply the ue8m0 pack-tail alignment requirement to fp32 scales,
    which have no packing, and rejected this shape outright."""
    torch.manual_seed(int(column_major) + 2 * (dtype == torch.float16))
    num_tokens = 128
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype)
    _, scale_ref = ref_fp8_fp32_scale(x, G)

    x_q = torch.zeros_like(x, dtype=fp8_dtype)
    x_s = _alloc_scale(
        (num_tokens, hidden), column_major=column_major, scale_ue8m0=False
    )
    per_token_group_quant_v3(x, x_q, x_s, G)
    torch.cuda.synchronize()

    torch.testing.assert_close(x_s, scale_ref, rtol=0, atol=0)
    assert _dequant_rel_err(x_q, x_s, x, G) < 0.05


@pytest.mark.parametrize("column_major", [False, True])
def test_v3_int8_scale(column_major):
    """int8 output (row-major / col-major fp32 scale): exact stored scale
    (amax/127) + dequant round-trip. Pins the multiply-by-inverse family (v1
    divided by the scale and differed by a ULP) without depending on v2."""
    torch.manual_seed(2 + int(column_major))
    num_tokens, hidden = 33, 4096
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    _, scale_ref = ref_int8(x, G)

    x_q = torch.zeros(num_tokens, hidden, device="cuda", dtype=torch.int8)
    x_s = _alloc_scale(
        (num_tokens, hidden), column_major=column_major, scale_ue8m0=False
    )
    per_token_group_quant_v3(x, x_q, x_s, G)
    torch.cuda.synchronize()

    torch.testing.assert_close(x_s, scale_ref, rtol=0, atol=0)
    # int8 group-quant is coarser than fp8 (127 vs 448 levels), so its mean
    # relative round-trip error on randn data sits a little above the fp8 0.05.
    assert _dequant_rel_err(x_q, x_s, x, G) < 0.08


def test_v3_group_size_256_roundtrip():
    """Group 256 (32 lanes on H100, 16 on Blackwell) is above v2's old cap of
    128. Pin the derived property: the fp32 scale equals absmax/FMAX and dequant
    round-trips. A mis-mapped wide subwarp would fold the wrong elements into
    the group absmax and move the scale."""
    torch.manual_seed(256)
    num_tokens, hidden, gs = 9, 4096, 256
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    _, scale_ref = ref_fp8_fp32_scale(x, gs)

    x_q = torch.zeros_like(x, dtype=fp8_dtype)
    x_s = torch.zeros(num_tokens, hidden // gs, device="cuda", dtype=torch.float32)
    per_token_group_quant_v3(x, x_q, x_s, gs)
    torch.cuda.synchronize()

    torch.testing.assert_close(x_s, scale_ref, rtol=0, atol=0)
    assert _dequant_rel_err(x_q, x_s, x, gs) < 0.05


# --------------------------------------------------------------------------- #
# Fused silu+mul.
# --------------------------------------------------------------------------- #
def _ref_silu_mul(x, hidden):
    """silu in fp32, round to the input dtype, multiply in the input dtype --
    matching the kernel's fused path exactly."""
    gate, up = x[..., :hidden], x[..., hidden:]
    return torch.nn.functional.silu(gate.float()).to(x.dtype) * up


@pytest.mark.parametrize("column_major", [True, False])
@pytest.mark.parametrize("scale_ue8m0", [True, False])
def test_v3_fused_silu(scale_ue8m0, column_major):
    """fuse_silu_and_mul quantizes ``silu(x[..., :h]) * x[..., h:]`` (SGLang's
    SiluAndMul: first half is the gated half). Covered across all four scale
    layouts so the fused [gate | up] input layout is pinned everywhere.

    The kernel's silu uses the fast ``__tanhf`` intrinsic on Blackwell, which
    is not bit-reproducible from torch's sigmoid-based silu, so this is a
    property test: dequant the kernel output through its own scale and check it
    round-trips to the torch activation within fp8 error. (The quant math is
    pinned bit-exact by the non-fused ue8m0 tests; a wrong gate/up split or
    offset would move the round-trip well past tolerance.)"""
    torch.manual_seed(int(scale_ue8m0) * 2 + int(column_major))
    num_tokens, hidden = 37, 4096
    x = torch.randn(num_tokens, hidden * 2, device="cuda", dtype=torch.bfloat16)
    act = _ref_silu_mul(x, hidden)

    x_q = torch.zeros(num_tokens, hidden, device="cuda", dtype=fp8_dtype)
    if scale_ue8m0 and not column_major:
        x_s = torch.zeros(
            num_tokens, hidden // G // 4, device="cuda", dtype=torch.int32
        )
    else:
        x_s = _alloc_scale(
            (num_tokens, hidden), column_major=column_major, scale_ue8m0=scale_ue8m0
        )
    per_token_group_quant_v3(
        x, x_q, x_s, G, scale_ue8m0=scale_ue8m0, fuse_silu_and_mul=True
    )
    torch.cuda.synchronize()

    deq_scale = _packed_exp_to_dequant_scale(x_s, hidden // G) if scale_ue8m0 else x_s
    assert _dequant_rel_err(x_q, deq_scale, act, G) < 0.05


# --------------------------------------------------------------------------- #
# Masked EP-MoE schedule.
# --------------------------------------------------------------------------- #
MASKED_CASES = get_ci_test_range(
    list(itertools.product([2, 5], [2048, 4096], [128, 384])),
    [(2, 2048, 128), (5, 4096, 384)],
)


@pytest.mark.parametrize("masked_m_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("expected_m", [None, 4])
@pytest.mark.parametrize("num_experts,hidden,tokens_pad", MASKED_CASES)
def test_v3_masked(num_experts, hidden, tokens_pad, expected_m, masked_m_dtype):
    """Masked EP-MoE schedule (col-packed ue8m0, plain quant -- no silu, so the
    quant is bit-reproducible): rows < masked_m[e] are bit-exact vs the torch
    reference; rows >= masked_m[e] stay zero (untouched). Fusion numerics are
    covered by test_v3_fused_silu; here the schedule is what's under test.

    masked_m is accepted as int32 or int64 (the latter read as its low word),
    so both dtypes are exercised.

    expected_m=4 shrinks the grid's token axis far below masked_m, so the
    grid-stride token loop must still cover every valid token -- guards the
    host-hint-only contract (a wrong hint can never drop tokens)."""
    torch.manual_seed(num_experts * 1000 + hidden + tokens_pad)
    x = torch.randn(
        num_experts, tokens_pad, hidden, device="cuda", dtype=torch.bfloat16
    )
    masked_m = torch.randint(
        0, tokens_pad + 1, (num_experts,), device="cuda", dtype=masked_m_dtype
    )
    out_shape = (num_experts, tokens_pad, hidden)

    x_q = torch.zeros(out_shape, device="cuda", dtype=fp8_dtype)
    x_s = _alloc_scale(out_shape, column_major=True, scale_ue8m0=True)
    per_token_group_quant_v3(
        x, x_q, x_s, G, scale_ue8m0=True, masked_m=masked_m, expected_m=expected_m
    )
    torch.cuda.synchronize()

    q_ref, exp_ref = ref_fp8_ue8m0(x, G)
    exp = _decode_packed_exp(x_s, hidden // G)
    for e in range(num_experts):
        m = int(masked_m[e])
        assert torch.equal(
            x_q[e, :m].view(torch.int8), q_ref[e, :m].view(torch.int8)
        ), "written codes differ"
        assert torch.equal(exp[e, :m], exp_ref[e, :m]), "written exponents differ"
        assert torch.all(x_q[e, m:].view(torch.int8) == 0), "padding codes touched"


def _as_int32(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int32) if t.dtype == torch.int32 else t


# (out_dtype, column_major_scales, scale_ue8m0); ue8m0 implies fp8 output.
AUTO_ALLOC_CASES = [
    (torch.float8_e4m3fn, True, True),
    (torch.float8_e4m3fn, False, True),
    (torch.float8_e4m3fn, True, False),
    (torch.float8_e4m3fn, False, False),
    (torch.int8, True, False),
    (torch.int8, False, False),
]


def test_v3_masked_fused():
    """The production EP-MoE path: masked schedule + fuse_silu_and_mul +
    col-packed ue8m0. silu is not bit-reproducible, so check the written rows
    round-trip to the torch activation and padding rows stay zero."""
    torch.manual_seed(7)
    num_experts, tokens_pad, hidden = 3, 256, 2048
    x = torch.randn(
        num_experts, tokens_pad, hidden * 2, device="cuda", dtype=torch.bfloat16
    )
    masked_m = torch.randint(
        0, tokens_pad + 1, (num_experts,), device="cuda", dtype=torch.int32
    )
    out_shape = (num_experts, tokens_pad, hidden)

    x_q = torch.zeros(out_shape, device="cuda", dtype=fp8_dtype)
    x_s = _alloc_scale(out_shape, column_major=True, scale_ue8m0=True)
    per_token_group_quant_v3(
        x, x_q, x_s, G, scale_ue8m0=True, fuse_silu_and_mul=True, masked_m=masked_m
    )
    torch.cuda.synchronize()

    act = _ref_silu_mul(x, hidden)
    deq_scale = _packed_exp_to_dequant_scale(x_s, hidden // G)
    for e in range(num_experts):
        m = int(masked_m[e])
        if m > 0:
            assert _dequant_rel_err(x_q[e, :m], deq_scale[e, :m], act[e, :m], G) < 0.05
        assert torch.all(x_q[e, m:].view(torch.int8) == 0), "padding touched"


@pytest.mark.parametrize("out_dtype,column_major_scales,scale_ue8m0", AUTO_ALLOC_CASES)
def test_v3_auto_allocation(out_dtype, column_major_scales, scale_ue8m0):
    """Omitting output_q/output_s allocates them per out_dtype / major mode /
    scale format and returns (q, s). The auto-allocated run must be bit-
    identical to quantizing into caller-supplied buffers of the same layout --
    guards that _allocate_outputs picks the layout the kernel decodes."""
    torch.manual_seed(int(column_major_scales) * 2 + int(scale_ue8m0))
    num_tokens, hidden = 38, 2048  # 16 groups, %4 == 0 for row-packed ue8m0
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)

    q_buf = torch.zeros(num_tokens, hidden, device="cuda", dtype=out_dtype)
    if scale_ue8m0 and not column_major_scales:
        s_buf = torch.zeros(
            num_tokens, hidden // G // 4, device="cuda", dtype=torch.int32
        )
    else:
        s_buf = _alloc_scale(
            (num_tokens, hidden),
            column_major=column_major_scales,
            scale_ue8m0=scale_ue8m0,
        )
    per_token_group_quant_v3(x, q_buf, s_buf, G, scale_ue8m0=scale_ue8m0)

    q_auto, s_auto = per_token_group_quant_v3(
        x,
        group_size=G,
        scale_ue8m0=scale_ue8m0,
        column_major_scales=column_major_scales,
        out_dtype=out_dtype,
    )
    torch.cuda.synchronize()

    assert q_auto.dtype == out_dtype and q_auto.shape == x.shape
    assert s_auto.dtype == s_buf.dtype and s_auto.shape == s_buf.shape
    assert torch.equal(q_auto.view(torch.int8), q_buf.view(torch.int8)), "codes differ"
    assert torch.equal(_as_int32(s_auto), _as_int32(s_buf)), "scales differ"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
