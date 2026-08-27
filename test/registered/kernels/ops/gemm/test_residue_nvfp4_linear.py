"""GPU tests for the opaque residue NVFP4 linear op (mext_r1 + plain paths).

The op must produce the same math regardless of which internal chain M
dispatches to: the CuTeDSL fold at decode M, the two-GEMM pair-sum above the
fold band, and the plain chain for layers without residue.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] != 10:
    pytest.skip("sm100/sm103 (datacenter Blackwell) required", allow_module_level=True)
pytest.importorskip("cutlass", reason="nvidia-cutlass-dsl required")
pytest.importorskip("flashinfer", reason="flashinfer required")

from sglang.kernels.ops.gemm.residue_nvfp4_linear import (
    RESIDUE_MEXT_R1,
    RESIDUE_NONE,
    fold_max_m_for,
    nvfp4_linear,
    residue_kind_of,
)
from sglang.test.kernels.residue_nvfp4 import (
    dequant_nvfp4_weight,
    quantize_nvfp4_weight,
    sf_unswizzle,
    swizzle_scale,
)


def test_residue_kind_disambiguation():
    assert residue_kind_of(0, False) == RESIDUE_NONE
    assert residue_kind_of(0, True) == RESIDUE_MEXT_R1
    assert residue_kind_of(512, True) == RESIDUE_MEXT_R1
    assert residue_kind_of(512, False) == "k_ext"


def _make_layer(n, k, seed=0):
    torch.manual_seed(seed)
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / (k**0.5)
    w_packed, w_sf, w_global = quantize_nvfp4_weight(w)
    return w, w_packed, w_sf.view(torch.float8_e4m3fn), w_global


def _call_op_mext_r1(x, w_packed, w_sf, x_global, w_global):
    alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()
    placeholder_mask = x.new_zeros(1, dtype=torch.uint8)
    return nvfp4_linear(
        x,
        w_packed,
        x_global.reshape(1),
        w_sf,
        w_sf,  # full scale doubles as base scale for mext_r1
        placeholder_mask,
        alpha,
        int(w_packed.shape[1] * 2),  # k_base == full K
        0,  # num_salient
        0,  # weights_padding_cols
        int(w_packed.shape[0]),
        True,  # fold_eligible
        True,  # is_mext_r1
    )


@pytest.mark.parametrize("m", [1, 8, 64, 128, 256, 512])
def test_mext_r1_matches_reference_at_every_m(m):
    """Below fold_max_m the op takes the fold; above it the two-GEMM path.
    Both must apply the residue -- the error vs the BF16 reference must stay
    at second-order (residue-corrected) magnitude at EVERY M."""
    n, k = 2048, 4096
    w, w_packed, w_sf, w_global = _make_layer(n, k)
    torch.manual_seed(m)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    x_global = ((448.0 * 6.0) / x.float().abs().max()).cuda()

    out = _call_op_mext_r1(x, w_packed, w_sf, x_global, w_global)
    assert out.shape == (m, n)

    w_deq = dequant_nvfp4_weight(w_packed, w_sf.view(torch.uint8), w_global)
    ref = x.float() @ w_deq.T

    # The activation is residue-corrected, so the dominant error is the
    # weight quantization; base+residue keeps the activation contribution
    # second-order. Compare against plain-NVFP4-activation error.
    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

    x_fp4, x_sf = fp4_quantize(x, x_global.reshape(1))
    from sglang.srt.layers.quantization.modelopt_quant import fp4_gemm

    alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()
    out_plain = fp4_gemm(x_fp4, w_packed.T, x_sf, w_sf.T, alpha, torch.bfloat16, n)

    err_residue = (out.float() - ref).norm() / ref.norm()
    err_plain = (out_plain.float() - ref).norm() / ref.norm()
    assert err_residue < 0.8 * err_plain, (
        f"m={m}: residue path not better than plain "
        f"(residue={err_residue:.5f} plain={err_plain:.5f})"
    )


def test_mext_r1_fold_and_two_gemm_agree():
    """The fold (decode band) and two-GEMM (above the band) chains are the
    same linear form; forcing the same input through both via the M threshold
    must give closely matching outputs."""
    n, k = 1024, 2048
    _, w_packed, w_sf, w_global = _make_layer(n, k, seed=7)
    fold_m = fold_max_m_for("sm100")
    m = fold_m  # in-band -> fold
    torch.manual_seed(99)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    x_global = ((448.0 * 6.0) / x.float().abs().max()).cuda()

    out_fold = _call_op_mext_r1(x, w_packed, w_sf, x_global, w_global)

    # Same rows replicated past the band -> two-GEMM path; the first m rows
    # must agree with the fold output.
    x_big = torch.cat([x, x], dim=0)
    out_two = _call_op_mext_r1(x_big, w_packed, w_sf, x_global, w_global)[:m]

    rel = (out_fold.float() - out_two.float()).norm() / out_two.float().norm()
    assert rel < 2e-2, f"fold vs two_gemm rel err {rel:.2e}"


def test_plain_kind_matches_stock_gemm():
    n, k = 1024, 2048
    _, w_packed, w_sf, w_global = _make_layer(n, k, seed=3)
    m = 16
    torch.manual_seed(5)
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    x_global = ((448.0 * 6.0) / x.float().abs().max()).cuda()
    alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()
    placeholder_mask = x.new_zeros(1, dtype=torch.uint8)

    out = nvfp4_linear(
        x,
        w_packed,
        x_global.reshape(1),
        w_sf,
        w_sf,
        placeholder_mask,
        alpha,
        0,
        0,
        0,
        n,
        False,  # not fold eligible
        False,  # not mext_r1
    )

    from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
    from sglang.srt.layers.quantization.modelopt_quant import fp4_gemm

    x_fp4, x_sf = fp4_quantize(x, x_global.reshape(1))
    ref = fp4_gemm(x_fp4, w_packed.T, x_sf, w_sf.T, alpha, torch.bfloat16, n)

    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_mext_r1_never_falls_through_to_plain():
    n, k = 512, 1024
    _, w_packed, w_sf, w_global = _make_layer(n, k, seed=1)
    x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda")
    x_global = torch.tensor(1.0, device="cuda")
    alpha = torch.ones(1, device="cuda")
    with pytest.raises(RuntimeError, match="mext_r1 layer reached"):
        nvfp4_linear(
            x,
            w_packed,
            x_global.reshape(1),
            w_sf,
            w_sf,
            x.new_zeros(1, dtype=torch.uint8),
            alpha,
            k,
            0,
            0,
            n,
            False,  # NOT fold eligible -- must raise, not serve plain
            True,  # is_mext_r1
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


# ── k_ext (extended-K) dispatch ──────────────────────────────────────────────


def test_k_ext_matches_extended_decomposition():
    """The k_ext chain quantizes [base | residue] and contracts against the
    K-extended weight; the salient channels' effective precision must beat
    the plain chain on the same base weight."""
    from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
        indices_to_channel_masks,
    )

    n, k, per_block = 1024, 2048, 2
    num_salient = k * per_block // 8
    n_ext = k + num_salient
    torch.manual_seed(21)

    # Build the K-extended weight the exporter would produce:
    # W_ext = [W | W[:, salient]] so the residue columns re-hit the salient
    # weight columns.
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda") / (k**0.5)
    indices = torch.tensor(
        sorted(i for b in range(0, k, 8) for i in range(b, b + per_block)),
        device="cuda",
    )
    w_ext = torch.cat([w, w[:, indices]], dim=1)
    w_packed, w_sf, w_global = quantize_nvfp4_weight(w_ext)
    w_sf = w_sf.view(torch.float8_e4m3fn)

    m = 32
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    x_global = ((448.0 * 6.0) / x.float().abs().max()).cuda()
    alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()
    masks = indices_to_channel_masks(indices, k)

    out = nvfp4_linear(
        x,
        w_packed,
        x_global.reshape(1),
        w_sf,
        w_sf,
        masks,
        alpha,
        k,  # k_base (unused: not fold eligible)
        num_salient,
        0,  # weights_padding_cols (n_ext % 32 == 0 here)
        n,
        False,  # fold_eligible
        False,  # is_mext_r1
    )
    assert out.shape == (m, n)

    # Reference: BF16 contraction against the dequantized EXTENDED weight,
    # with the activation's residue-corrected decomposition.
    w_ext_deq = dequant_nvfp4_weight(w_packed, w_sf.view(torch.uint8), w_global)
    ref = x.float() @ w_ext_deq[:, :k].T  # base-only reference

    err_kext = (out.float() - ref).norm() / ref.norm()
    # Sanity band: the residue adds signal beyond the base contraction, so
    # the difference from base-only must be small but nonzero.
    assert 1e-5 < err_kext < 0.05, f"k_ext output implausible: {err_kext:.2e}"

    # Plain chain on the base weight for comparison of salient accuracy.
    w_base_packed, w_base_sf, w_base_global = quantize_nvfp4_weight(w)
    w_base_sf = w_base_sf.view(torch.float8_e4m3fn)
    alpha_base = (1.0 / (x_global * w_base_global)).reshape(1).float().cuda()
    out_plain = nvfp4_linear(
        x,
        w_base_packed,
        x_global.reshape(1),
        w_base_sf,
        w_base_sf,
        x.new_zeros(1, dtype=torch.uint8),
        alpha_base,
        0,
        0,
        0,
        n,
        False,
        False,
    )

    # True reference with full-precision operands.
    true_ref = x.float() @ w.float().T
    err_res = (out.float() - true_ref).norm() / true_ref.norm()
    err_pln = (out_plain.float() - true_ref).norm() / true_ref.norm()
    assert (
        err_res < err_pln
    ), f"k_ext not better than plain: kext={err_res:.5f} plain={err_pln:.5f}"


def test_k_ext_hybrid_uses_fold_small_m_and_extended_k_large_m():
    """An extended checkpoint has two valid runtime views of one allocation:
    base-K strided fold for decode and full K-ext for larger batches."""
    from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
        indices_to_channel_masks,
    )

    output_n, padded_n, k, per_block = 1000, 1024, 2048, 2
    num_salient = k * per_block // 8
    k_ext = k + num_salient
    indices = torch.tensor(
        sorted(i for b in range(0, k, 8) for i in range(b, b + per_block)),
        device="cuda",
    )
    torch.manual_seed(31)
    w = torch.randn(output_n, k, dtype=torch.bfloat16, device="cuda") / (k**0.5)
    w_ext = torch.cat([w, w[:, indices]], dim=1)
    # Match parent PWAL's N-padding contract before quantizing the test weight.
    w_ext = torch.nn.functional.pad(w_ext, (0, 0, 0, padded_n - output_n))
    w_packed, w_sf_full, w_global = quantize_nvfp4_weight(w_ext)
    w_sf_full = w_sf_full.view(torch.float8_e4m3fn)

    raw_sf = sf_unswizzle(w_sf_full.view(torch.uint8), padded_n, k_ext).to(
        torch.float8_e4m3fn
    )
    w_sf_base = swizzle_scale(raw_sf[:, : k // 16])
    masks = indices_to_channel_masks(indices, k)

    def call(x, *, fold_eligible):
        x_global = ((448.0 * 6.0) / x.float().abs().max()).cuda()
        alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()
        return nvfp4_linear(
            x,
            w_packed,
            x_global.reshape(1),
            w_sf_full,
            w_sf_base,
            masks,
            alpha,
            k,
            num_salient,
            0,
            output_n,
            fold_eligible,
            False,
        )

    # Small M: hybrid must equal a direct fold over the base-prefix VIEW,
    # including its K-ext row stride, and slice parent N padding away.
    small_m = min(8, fold_max_m_for("sm100"))
    x_small = torch.randn(small_m, k, dtype=torch.bfloat16, device="cuda")
    out_hybrid_small = call(x_small, fold_eligible=True)
    x_global = ((448.0 * 6.0) / x_small.float().abs().max()).cuda()
    alpha = (1.0 / (x_global * w_global)).reshape(1).float().cuda()
    out_fold = nvfp4_linear(
        x_small,
        w_packed[:, : k // 2],
        x_global.reshape(1),
        w_sf_base,
        w_sf_base,
        x_small.new_zeros(1, dtype=torch.uint8),
        alpha,
        k,
        0,
        0,
        output_n,
        True,
        True,
    )
    assert out_hybrid_small.shape == (small_m, output_n)
    torch.testing.assert_close(out_hybrid_small, out_fold, rtol=0, atol=0)

    # Above the crossover: enabling fold eligibility must not change the
    # existing K-ext result.
    large_m = fold_max_m_for("sm100") + 1
    x_large = torch.randn(large_m, k, dtype=torch.bfloat16, device="cuda")
    out_hybrid_large = call(x_large, fold_eligible=True)
    out_kext_large = call(x_large, fold_eligible=False)
    torch.testing.assert_close(out_hybrid_large, out_kext_large, rtol=0, atol=0)
