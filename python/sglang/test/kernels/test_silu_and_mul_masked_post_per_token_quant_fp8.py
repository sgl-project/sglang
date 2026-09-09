"""Tests for ``silu_and_mul_masked_post_per_token_quant_fp8`` (EP, 3D per-token FP8)."""

import sys

import pytest
import torch

from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="PPU-only kernel")


_EP_SHAPES = [
    # --- (E, T, two_N)  E=experts, T=max_tokens, two_N=2*moe_intermediate_size ---
    # T=4096: large prefill batch
    (8, 4096, 6144),  # DS-Pro E=8
    (4, 4096, 4096),  # DS-Flash E=4
    (16, 4096, 6144),  # Minimax-M3 E=16 (H=3072)
    # T=2048: medium prefill
    (16, 2048, 6144),  # DS-Pro E=16
    (8, 2048, 4096),  # DS-Flash E=8
    # T=1024: small prefill / large decode
    (32, 1024, 6144),  # DS-Pro E=32
    (8, 1024, 2048),  # Qwen E=8
    # T=512: decode batch
    (8, 512, 4096),  # DS-Flash E=8
    (4, 512, 3072),  # Minimax E=4 (H=1536)
    (16, 512, 6144),  # Minimax-M3 E=16 (H=3072)
    # T=256: small decode
    (4, 256, 2048),  # Qwen E=4
    # T=128: tiny decode
    (4, 128, 4096),  # DS-Flash E=4
    # T=64: minimal
    (4, 64, 2048),  # Qwen E=4
    (16, 64, 6144),  # Minimax-M3 E=16 (H=3072)
    # Edge case
    (1, 1, 4096),  # E=1 T=1
]

# Activation modes: plain silu / DeepSeek V4 swiglu_limit / oai-swiglu
# (MiniMax-M3 / gpt-oss) gemm1_alpha + gemm1_clamp_limit.
_ACT_CASES = [
    dict(swiglu_limit=None, gemm1_alpha=None, gemm1_clamp_limit=None),
    dict(swiglu_limit=7.0, gemm1_alpha=None, gemm1_clamp_limit=None),
    dict(swiglu_limit=None, gemm1_alpha=1.702, gemm1_clamp_limit=7.0),
]
_ACT_IDS = ["plain", "swiglu", "oai"]

# (gemm1_alpha, gemm1_clamp_limit) pairs for the oai-swiglu numerical tests.
_OAI_CASES = [
    (1.702, 7.0),  # gpt-oss style
    (1.0, 4.0),
    (2.5, 12.0),
]

_EXPECTED_M_CASES = [
    # (E, T, two_N, expected_m_values)
    (8, 8192, 6144, [31, 64, 128, 256]),  # Pro EP32, T_padded=8192
    (13, 8192, 6144, [31, 64, 128]),  # Pro EP20
    (16, 8192, 4096, [31, 64, 128, 256]),  # Flash EP16
    (8, 4096, 6144, [16, 32, 64]),  # Pro smaller T
    (4, 1024, 4096, [8, 16, 31]),  # Flash small T
]


def _run(inp, masked_m, act, **kwargs):
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_masked_post_per_token_quant_fp8,
    )

    return silu_and_mul_masked_post_per_token_quant_fp8(inp, masked_m, **act, **kwargs)


def _ref_per_token_fp8(inp, act, eps=1e-10):
    """Pure-torch reference mirroring the triton fallback / CUDA math."""
    two_H = inp.shape[-1]
    H = two_H // 2
    gate = inp[..., :H]
    up = inp[..., H:]

    if act["gemm1_alpha"] is not None:
        alpha = act["gemm1_alpha"]
        clamp_limit = act["gemm1_clamp_limit"]
        # oai-swiglu is computed in fp32: gate * sigmoid(alpha*gate) * (up+1)
        gate = gate.float().clamp(max=clamp_limit)
        up = up.float().clamp(min=-clamp_limit, max=clamp_limit)
        prod = gate * torch.sigmoid(gate * alpha) * (up + 1.0)
    else:
        if act["swiglu_limit"] is not None:
            # clamp in bf16 to match the kernel (__hmin2 / __hmax2)
            lim = torch.tensor(
                act["swiglu_limit"], dtype=torch.bfloat16, device=inp.device
            )
            gate = gate.clamp(max=lim)
            up = up.clamp(min=-lim, max=lim)
        # silu computed in fp32, rounded to bf16, product in bf16
        silu = (gate.float() * torch.sigmoid(gate.float())).to(torch.bfloat16)
        prod = up * silu

    # per-token absmax (bf16 domain for the non-alpha path, like the kernel)
    absmax = prod.abs().amax(dim=-1, keepdim=True).float().clamp_min(eps)
    scale = absmax / 448.0
    q = (prod.float() * (1.0 / scale)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return q, scale


def _assert_close_quant(out_q, out_s, ref_q, ref_s, ctx):
    """Compare fp8 codes via dequantization (1-ulp tolerance) + exact scales."""
    torch.testing.assert_close(
        out_s,
        ref_s,
        rtol=1e-6,
        atol=1e-12,
        msg=f"scale mismatch: {ctx}",
    )
    # fp8 codes may differ by 1 ulp near rounding boundaries (PPU sigmoid
    # hardware function vs torch.sigmoid, multiply-by-reciprocal vs divide),
    # so compare the dequantized values with a loose relative tolerance.
    deq = out_q.float() * out_s
    ref_deq = ref_q.float() * ref_s
    torch.testing.assert_close(
        deq,
        ref_deq,
        rtol=0.2,
        atol=1e-3,
        msg=f"quant mismatch: {ctx}",
    )


@pytest.mark.parametrize("act", _ACT_CASES, ids=_ACT_IDS)
@pytest.mark.parametrize("E,T,two_N", _EP_SHAPES)
def test_shapes(E: int, T: int, two_N: int, act: dict) -> None:
    torch.manual_seed(E * T)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(
        low=1, high=T + 1, size=(E,), dtype=torch.int32, device="cuda"
    )

    output, output_scale = _run(inp, masked_m, act)

    H = two_N // 2
    assert output.dtype == torch.float8_e4m3fn
    assert output.shape == (E, T, H)
    assert output_scale.dtype == torch.float32
    assert output_scale.shape == (E, T, 1)


def test_masked_rows_consistency() -> None:
    """Verify that valid rows (within masked_m) are consistent across runs."""
    E, T, two_N = 4, 2048, 4096
    torch.manual_seed(0)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")

    valid = T // 2
    masked_m = torch.full((E,), valid, dtype=torch.int32, device="cuda")

    for act in _ACT_CASES:
        out_a, scale_a = _run(inp, masked_m, act)
        out_b, scale_b = _run(inp, masked_m, act)

        for e in range(E):
            assert torch.equal(
                out_a[e, :valid], out_b[e, :valid]
            ), f"quant mismatch E={e} act={act}"
            assert torch.equal(
                scale_a[e, :valid], scale_b[e, :valid]
            ), f"scale mismatch E={e} act={act}"


def test_determinism() -> None:
    E, T, two_N = 4, 2048, 4096
    torch.manual_seed(11)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(1, T + 1, (E,), dtype=torch.int32, device="cuda")

    for act in _ACT_CASES:
        out_a, scale_a = _run(inp, masked_m, act)
        out_b, scale_b = _run(inp, masked_m, act)

        masked_cpu = masked_m.cpu().tolist()
        for e, m in enumerate(masked_cpu):
            assert torch.equal(
                out_a[e, :m], out_b[e, :m]
            ), f"quant determinism fail E={e} act={act}"
            assert torch.equal(
                scale_a[e, :m], scale_b[e, :m]
            ), f"scale determinism fail E={e} act={act}"


@pytest.mark.parametrize("E,T,two_N,em_list", _EXPECTED_M_CASES)
@pytest.mark.parametrize("act", _ACT_CASES, ids=_ACT_IDS)
def test_expected_m_bitexact(E, T, two_N, em_list, act) -> None:
    """Verify expected_m grid optimization produces bit-exact results."""
    torch.manual_seed(E * T + two_N)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(1, min(T, 300) + 1, (E,), dtype=torch.int32, device="cuda")

    # Baseline: no expected_m
    out_base, scale_base = _run(inp, masked_m, act, expected_m=None)

    masked_cpu = masked_m.cpu().tolist()
    for em in em_list:
        out_opt, scale_opt = _run(inp, masked_m, act, expected_m=em)
        for e, m in enumerate(masked_cpu):
            assert torch.equal(
                out_base[e, :m], out_opt[e, :m]
            ), f"quant mismatch: E={E} T={T} em={em} expert={e} act={act}"
            assert torch.equal(
                scale_base[e, :m], scale_opt[e, :m]
            ), f"scale mismatch: E={E} T={T} em={em} expert={e} act={act}"


def test_silu_numerical_correctness() -> None:
    """silu / swiglu_limit paths match the pure-torch reference."""
    E, T, two_N = 4, 512, 4096
    torch.manual_seed(2024)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    # scale up so the swiglu clamp path is exercised
    inp = inp * 3.0
    masked_m = torch.randint(1, T + 1, (E,), dtype=torch.int32, device="cuda")
    masked_cpu = masked_m.cpu().tolist()

    for act in _ACT_CASES[:2]:
        out_q, out_s = _run(inp, masked_m, act)
        ref_q, ref_s = _ref_per_token_fp8(inp, act)
        for e, m in enumerate(masked_cpu):
            _assert_close_quant(
                out_q[e, :m],
                out_s[e, :m],
                ref_q[e, :m],
                ref_s[e, :m],
                ctx=f"act={act} expert={e}",
            )


@pytest.mark.parametrize("alpha,clamp_limit", _OAI_CASES)
def test_oai_swiglu_numerical_correctness(alpha: float, clamp_limit: float) -> None:
    """oai-swiglu (gemm1_alpha + gemm1_clamp_limit) matches the torch reference."""
    E, T, two_N = 4, 512, 4096
    torch.manual_seed(2024)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    # scale up so the clamp paths are exercised
    inp = inp * 3.0
    masked_m = torch.randint(1, T + 1, (E,), dtype=torch.int32, device="cuda")
    masked_cpu = masked_m.cpu().tolist()

    act = dict(swiglu_limit=None, gemm1_alpha=alpha, gemm1_clamp_limit=clamp_limit)
    out_q, out_s = _run(inp, masked_m, act)
    ref_q, ref_s = _ref_per_token_fp8(inp, act)
    for e, m in enumerate(masked_cpu):
        _assert_close_quant(
            out_q[e, :m],
            out_s[e, :m],
            ref_q[e, :m],
            ref_s[e, :m],
            ctx=f"alpha={alpha} clamp={clamp_limit} expert={e}",
        )


def test_scale_ue8m0_rounds_up_to_pow2() -> None:
    """Verify scale_ue8m0 rounds the per-token scale up to a power of two."""
    E, T, two_N = 4, 256, 4096
    torch.manual_seed(7)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(1, T + 1, (E,), dtype=torch.int32, device="cuda")
    masked_cpu = masked_m.cpu().tolist()

    for act in _ACT_CASES:
        out_q, out_s = _run(inp, masked_m, act, scale_ue8m0=True)
        _, ref_s = _run(inp, masked_m, act)

        for e, m in enumerate(masked_cpu):
            s = out_s[e, :m, 0]
            assert (s > 0).all(), f"scale must be positive: expert={e}"
            log2s = torch.log2(s)
            assert torch.allclose(
                log2s, log2s.round(), atol=1e-5
            ), f"scale not a power of two: expert={e}"
            # round-up must never shrink the scale
            assert (s >= ref_s[e, :m, 0]).all(), f"ue8m0 scale shrunk: expert={e}"


def test_invalid_args() -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_masked_post_per_token_quant_fp8,
    )

    E, T, two_N = 2, 64, 2048
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.full((E,), T, dtype=torch.int32, device="cuda")

    # swiglu_limit and gemm1_alpha are mutually exclusive
    with pytest.raises(AssertionError, match="mutually exclusive"):
        silu_and_mul_masked_post_per_token_quant_fp8(
            inp,
            masked_m,
            swiglu_limit=7.0,
            gemm1_alpha=1.702,
            gemm1_clamp_limit=7.0,
        )
    # gemm1_alpha requires gemm1_clamp_limit
    with pytest.raises(AssertionError, match="requires gemm1_clamp_limit"):
        silu_and_mul_masked_post_per_token_quant_fp8(inp, masked_m, gemm1_alpha=1.702)
    # H must be a multiple of 8
    bad = torch.randn((E, T, 4100), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(AssertionError, match="multiple of 8"):
        silu_and_mul_masked_post_per_token_quant_fp8(bad, masked_m)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
