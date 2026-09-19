"""Tests for ``silu_and_mul_masked_post_quant_mxfp4`` (EP, 3D MXFP4)."""

import sys

import pytest
import torch

from sglang.srt.utils import is_ppu

pytestmark = pytest.mark.skipif(not is_ppu(), reason="PPU-only kernel")


_EP_SHAPES = [
    # --- (E, T, two_N)  E=experts, T=max_tokens, two_N=2*moe_intermediate_size ---
    # T=4096: large prefill batch
    (8, 4096, 6144),  # DS-Pro EP=8
    (4, 4096, 4096),  # DS-Flash EP=4
    # T=2048: medium prefill
    (16, 2048, 6144),  # DS-Pro EP=16 (non-aligned H=3072)
    (8, 2048, 4096),  # DS-Flash EP=8
    # T=1024: small prefill / large decode
    (32, 1024, 6144),  # DS-Pro EP=32 (non-aligned H=3072)
    (8, 1024, 2048),  # Qwen EP=8
    # T=512: decode batch
    (8, 512, 4096),  # DS-Flash EP=8
    (4, 512, 3072),  # Minimax EP=4 (non-aligned H=1536)
    # T=256: small decode
    (4, 256, 2048),  # Qwen EP=4
    # T=128: tiny decode
    (4, 128, 4096),  # DS-Flash EP=4
    # T=64: minimal
    (4, 64, 2048),  # Qwen EP=4
    # Edge case
    (1, 1, 4096),  # E=1 T=1
]


@pytest.mark.parametrize("E,T,two_N", _EP_SHAPES)
@pytest.mark.parametrize("swiglu_limit", [None, 4.0, 6.0, 8.0])
def test_shapes(E: int, T: int, two_N: int, swiglu_limit: float | None) -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_masked_post_quant_mxfp4,
    )

    torch.manual_seed(E * T)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(
        low=1, high=T + 1, size=(E,), dtype=torch.int32, device="cuda"
    )

    output, output_scale = silu_and_mul_masked_post_quant_mxfp4(
        inp, masked_m, swiglu_limit=swiglu_limit
    )

    H = two_N // 2
    assert output.dtype == torch.uint8
    assert output.shape == (E, T, H // 2)
    assert output_scale.dtype == torch.uint16
    S_valid = (H + 63) // 64
    assert output_scale.shape == (E, S_valid, T)


def test_masked_rows_consistency() -> None:
    """Verify that valid rows (within masked_m) are consistent across runs."""
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_masked_post_quant_mxfp4,
    )

    E, T, two_N = 4, 2048, 4096
    torch.manual_seed(0)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")

    valid = T // 2
    masked_m = torch.full((E,), valid, dtype=torch.int32, device="cuda")

    for swiglu in [None, 4.0, 8.0, 12.0]:
        out_a, scale_a = silu_and_mul_masked_post_quant_mxfp4(
            inp, masked_m, swiglu_limit=swiglu
        )
        out_b, scale_b = silu_and_mul_masked_post_quant_mxfp4(
            inp, masked_m, swiglu_limit=swiglu
        )

        for e in range(E):
            assert torch.equal(out_a[e, :valid], out_b[e, :valid]), (
                f"quant mismatch E={e} swiglu={swiglu}"
            )
            assert torch.equal(scale_a[e, :, :valid], scale_b[e, :, :valid]), (
                f"scale mismatch E={e} swiglu={swiglu}"
            )


def test_determinism() -> None:
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_masked_post_quant_mxfp4,
    )

    E, T, two_N = 4, 2048, 4096
    torch.manual_seed(11)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(1, T + 1, (E,), dtype=torch.int32, device="cuda")

    for swiglu in [6.0, 8.0, 12.0]:
        out_a, scale_a = silu_and_mul_masked_post_quant_mxfp4(
            inp, masked_m, swiglu_limit=swiglu
        )
        out_b, scale_b = silu_and_mul_masked_post_quant_mxfp4(
            inp, masked_m, swiglu_limit=swiglu
        )

        masked_cpu = masked_m.cpu().tolist()
        for e, m in enumerate(masked_cpu):
            assert torch.equal(out_a[e, :m], out_b[e, :m]), (
                f"quant determinism fail E={e} swiglu={swiglu}"
            )
            assert torch.equal(scale_a[e, :, :m], scale_b[e, :, :m]), (
                f"scale determinism fail E={e} swiglu={swiglu}"
            )


def test_ep_scale_layout_for_deepgemm() -> None:
    """Verify EP scale layout matches DeepGemm masked GEMM expectations.

    EP returns scale as (E, S_valid, T). DeepGemm expects (E, T, S_valid),
    so the caller does .permute(0, 2, 1). Verify this works correctly.
    """
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_masked_post_quant_mxfp4,
    )

    # DeepSeek V3 EP shape: E=4, T=1024, H=2048
    E, T, twoH = 4, 1024, 4096
    H = twoH // 2  # = 2048
    torch.manual_seed(42)
    inp = torch.randn((E, T, twoH), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(1, T + 1, (E,), dtype=torch.int32, device="cuda")

    out_quant, out_scale = silu_and_mul_masked_post_quant_mxfp4(inp, masked_m)

    S_valid = (H + 63) // 64  # = 32
    assert out_scale.shape == (
        E,
        S_valid,
        T,
    ), f"Expected ({E}, {S_valid}, {T}), got {out_scale.shape}"

    # DeepGemm permute: (E, S_valid, T) -> (E, T, S_valid)
    dg_scale = out_scale.permute(0, 2, 1)
    assert dg_scale.shape == (E, T, S_valid)
    # Verify the permuted view is correct (contiguous check)
    dg_contig = dg_scale.contiguous()
    assert dg_contig.shape == (E, T, S_valid)
    assert torch.equal(dg_scale, dg_contig)


def test_ep_scale_layout_nonaligned() -> None:
    """Verify EP scale layout for non-aligned H (DS-Pro: H=3072)."""
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_masked_post_quant_mxfp4,
    )

    E, T, twoH = 8, 1024, 6144  # H=3072, 3072%64!=0
    H = twoH // 2
    torch.manual_seed(77)
    inp = torch.randn((E, T, twoH), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(1, T + 1, (E,), dtype=torch.int32, device="cuda")

    out_quant, out_scale = silu_and_mul_masked_post_quant_mxfp4(inp, masked_m)

    S_valid = (H + 63) // 64  # = 48
    assert out_scale.shape == (E, S_valid, T)
    assert out_quant.shape == (E, T, H // 2)


_EXPECTED_M_CASES = [
    # (E, T, two_N, expected_m_values)
    (8, 8192, 6144, [31, 64, 128, 256]),  # Pro EP32, T_padded=8192
    (13, 8192, 6144, [31, 64, 128]),  # Pro EP20
    (16, 8192, 4096, [31, 64, 128, 256]),  # Flash EP16
    (8, 4096, 6144, [16, 32, 64]),  # Pro smaller T
    (4, 1024, 4096, [8, 16, 31]),  # Flash small T
]


@pytest.mark.parametrize("E,T,two_N,em_list", _EXPECTED_M_CASES)
@pytest.mark.parametrize("swiglu_limit", [None, 8.0, 10.0])
def test_expected_m_bitexact(E, T, two_N, em_list, swiglu_limit) -> None:
    """Verify expected_m grid optimization produces bit-exact results."""
    from sglang.kernels.ops.elementwise.silu_mul_quant import (
        silu_and_mul_masked_post_quant_mxfp4,
    )

    torch.manual_seed(E * T + two_N)
    inp = torch.randn((E, T, two_N), dtype=torch.bfloat16, device="cuda")
    masked_m = torch.randint(1, min(T, 300) + 1, (E,), dtype=torch.int32, device="cuda")

    # Baseline: no expected_m
    out_base, scale_base = silu_and_mul_masked_post_quant_mxfp4(
        inp, masked_m, swiglu_limit=swiglu_limit, expected_m=None
    )

    masked_cpu = masked_m.cpu().tolist()
    for em in em_list:
        out_opt, scale_opt = silu_and_mul_masked_post_quant_mxfp4(
            inp, masked_m, swiglu_limit=swiglu_limit, expected_m=em
        )
        for e, m in enumerate(masked_cpu):
            assert torch.equal(out_base[e, :m], out_opt[e, :m]), (
                f"quant mismatch: E={E} T={T} em={em} expert={e}"
            )
            assert torch.equal(scale_base[e, :, :m], scale_opt[e, :, :m]), (
                f"scale mismatch: E={E} T={T} em={em} expert={e}"
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
