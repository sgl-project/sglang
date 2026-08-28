"""GPU parity tests for the residue NVFP4 mext_r1 activation quantization.

Everything is checked against a pure-torch oracle: FP4 e2m1 nibbles are
decoded through a LUT, the swizzled fp8-e4m3 scale bytes are de-swizzled with
the tiled-layout gather formula, and the base+residue reconstruction is
compared to the original activation. The residue must reduce the
reconstruction error by a large factor -- that property (not bit-exactness of
a re-derived oracle) is the contract the fold GEMM relies on.
"""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)
if torch.cuda.get_device_capability()[0] not in (10, 12):
    pytest.skip("Blackwell (SM100/SM103/SM120) required", allow_module_level=True)

from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
    MEXT_R1_LAYOUT_CONCAT,
    MEXT_R1_LAYOUT_CONCAT_K,
    MEXT_R1_LAYOUT_ROW_PAIR,
    scaled_fp4_quant_mext_r1,
)
from sglang.test.kernels.residue_nvfp4 import (
    base_row,
    decode_fp4,
    residue_row,
    sf_unswizzle,
)


def dequant_rows(
    data: torch.Tensor, sf: torch.Tensor, rows: torch.Tensor, scale_val: float
) -> torch.Tensor:
    """Dequantize the selected rows: code * SF / SFScale."""
    vals = decode_fp4(data[rows])
    sfs = sf[rows].repeat_interleave(16, dim=1)
    return vals * sfs / scale_val


def reconstruct(
    x: torch.Tensor,
    layout_mode: int,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (base_only, base_plus_residue) reconstructions of x."""
    m, k = x.shape
    data, sf_bytes = scaled_fp4_quant_mext_r1(x, scale, layout_mode=layout_mode)
    out_m = (
        data.shape[0] // 2 if layout_mode != MEXT_R1_LAYOUT_CONCAT_K else data.shape[0]
    )
    r = torch.arange(m, device=x.device)
    s_base = float(scale[0])

    if layout_mode == MEXT_R1_LAYOUT_CONCAT_K:
        sf = sf_unswizzle(sf_bytes, out_m, 2 * k)
        vals = decode_fp4(data)  # [out_m, 2K]
        sfs = sf.repeat_interleave(16, dim=1)
        base = vals[:m, :k] * sfs[:m, :k] / s_base
        residue = vals[:m, k:] * sfs[:m, k:] / s_base
        return base, base + residue

    sf = sf_unswizzle(sf_bytes, 2 * out_m, k)
    base = dequant_rows(data, sf, base_row(r, out_m, layout_mode), s_base)
    residue = dequant_rows(data, sf, residue_row(r, out_m, layout_mode), s_base)
    return base, base + residue


LAYOUTS = [
    ("concat", MEXT_R1_LAYOUT_CONCAT),
    ("row_pair", MEXT_R1_LAYOUT_ROW_PAIR),
    ("concat_k", MEXT_R1_LAYOUT_CONCAT_K),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("m", [1, 8, 64, 129])
@pytest.mark.parametrize("k", [512, 4096])
@pytest.mark.parametrize("layout_name,layout_mode", LAYOUTS)
def test_reconstruction_accuracy(dtype, m, k, layout_name, layout_mode):
    torch.manual_seed(m * k + layout_mode)
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    xf = x.float()
    amax = xf.abs().max()
    scale = ((448.0 * 6.0) / amax).reshape(1).float().cuda()

    base, full = reconstruct(x, layout_mode, scale)

    err_base = (base - xf).norm() / xf.norm()
    err_full = (full - xf).norm() / xf.norm()

    # Plain NVFP4 relative error is a few percent; the residue must cut it by
    # a large factor (second-order quantization error).
    assert err_base < 0.10, f"base-only error too large: {err_base:.4f}"
    assert err_full < 0.35 * err_base, (
        f"residue does not improve reconstruction: "
        f"base={err_base:.5f} full={err_full:.5f} ({layout_name})"
    )


@pytest.mark.parametrize("elts_mode", [8, 16])
def test_pack8_pack16_equivalent_reconstruction(elts_mode):
    """pack8 vs pack16 differ in SF granularity only at the thread level; both
    must reconstruct within the same tolerance and produce identical shapes."""
    torch.manual_seed(0)
    m, k = 32, 2048
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([448.0 * 6.0 / x.float().abs().max()], device="cuda")

    data, sf = scaled_fp4_quant_mext_r1(
        x, scale, layout_mode="row_pair", elts_mode=elts_mode
    )
    assert data.shape == (2 * m, k // 2)

    r = torch.arange(m, device="cuda")
    sf_grid = sf_unswizzle(sf, 2 * m, k)
    s = float(scale[0])
    base = dequant_rows(data, sf_grid, base_row(r, m, MEXT_R1_LAYOUT_ROW_PAIR), s)
    res = dequant_rows(data, sf_grid, residue_row(r, m, MEXT_R1_LAYOUT_ROW_PAIR), s)
    err = ((base + res) - x.float()).norm() / x.float().norm()
    assert err < 0.02, f"elts_mode={elts_mode} reconstruction error {err:.5f}"


def test_layouts_hold_identical_row_data():
    """All layouts hold the same base/residue data in different positions."""
    torch.manual_seed(1)
    m, k = 16, 1024
    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([448.0 * 6.0 / x.float().abs().max()], device="cuda")

    r = torch.arange(m, device="cuda")
    rows = {}
    for name, mode in LAYOUTS:
        data, _ = scaled_fp4_quant_mext_r1(x, scale, layout_mode=mode)
        if mode == MEXT_R1_LAYOUT_CONCAT_K:
            rows[name] = (data[:, : k // 2], data[:, k // 2 :])
        else:
            rows[name] = (data[base_row(r, m, mode)], data[residue_row(r, m, mode)])

    ref_base, ref_res = rows["concat"]
    for name in ("row_pair", "concat_k"):
        got_base, got_res = rows[name]
        assert torch.equal(ref_base, got_base), f"{name}: base rows differ"
        assert torch.equal(ref_res, got_res), f"{name}: residue rows differ"


def test_scalar_input_scale_matches_single_element_vector():
    """ModelOpt per-tensor scales are 0-D parameters in real checkpoints."""
    torch.manual_seed(4)
    x = torch.randn(7, 512, dtype=torch.bfloat16, device="cuda")
    scalar_scale = (448.0 * 6.0 / x.float().abs().max()).float()

    scalar_data, scalar_sf = scaled_fp4_quant_mext_r1(
        x, scalar_scale, layout_mode="row_pair"
    )
    vector_data, vector_sf = scaled_fp4_quant_mext_r1(
        x, scalar_scale.reshape(1), layout_mode="row_pair"
    )

    assert torch.equal(scalar_data, vector_data)
    # The swizzled allocation includes padding bytes outside the logical
    # [2M, K/16] scale grid; only compare the initialized logical entries.
    scalar_sf_grid = sf_unswizzle(scalar_sf, 2 * x.shape[0], x.shape[1])
    vector_sf_grid = sf_unswizzle(vector_sf, 2 * x.shape[0], x.shape[1])
    assert torch.equal(scalar_sf_grid, vector_sf_grid)


def test_rejects_bad_k():
    x = torch.randn(4, 24, dtype=torch.bfloat16, device="cuda")
    scale = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="K % 16"):
        scaled_fp4_quant_mext_r1(x, scale)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


# ── k_ext (selective residue) parity ─────────────────────────────────────────

from sglang.kernels.ops.quantization.residue_nvfp4_quant import (  # noqa: E402
    indices_to_channel_masks,
    scaled_fp4_quant_with_mask,
)


def salient_indices_for(k: int, per_block: int) -> torch.Tensor:
    """Top-`per_block` channels of every 8-channel block (exporter shape)."""
    idx = [i for b in range(0, k, 8) for i in range(b, b + per_block)]
    return torch.tensor(sorted(idx), dtype=torch.int64, device="cuda")


def kext_reconstruct(x, num_salient, indices, elts_mode):
    """(base_only, base_plus_residue) reconstructions via the k_ext op."""
    m, k = x.shape
    n_ext = k + num_salient
    scale = torch.tensor([448.0 * 6.0 / x.float().abs().max()], device="cuda")
    masks = indices_to_channel_masks(indices, k)
    data, sf_bytes = scaled_fp4_quant_with_mask(
        x, scale, masks, num_salient, elts_mode=elts_mode
    )
    assert data.shape == (m, n_ext // 2)

    s = float(scale[0])
    sf = sf_unswizzle(sf_bytes, (m + 127) // 128 * 128, n_ext)[:m]
    vals = decode_fp4(data)  # [m, n_ext]
    deq = vals * sf.repeat_interleave(16, dim=1) / s

    base = deq[:, :k]
    residue = deq[:, k:]
    full = base.clone()
    # The g-th salient channel (global sorted order) lands at extension
    # position g -- per-block-uniform selection makes this ordering exact.
    full[:, indices] += residue
    return base, full


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("per_block,ratio", [(1, 0.125), (2, 0.25), (4, 0.5)])
@pytest.mark.parametrize("elts_mode", ["auto", 8, 16])
@pytest.mark.parametrize("m", [1, 33, 256])
def test_kext_reconstruction(dtype, per_block, ratio, elts_mode, m):
    k = 2048
    num_salient = int(k * ratio)
    torch.manual_seed(per_block * m)
    x = torch.randn(m, k, dtype=dtype, device="cuda")
    xf = x.float()
    indices = salient_indices_for(k, per_block)

    base, full = kext_reconstruct(x, num_salient, indices, elts_mode)

    err_base = (base - xf).norm() / xf.norm()
    err_sal_base = (base[:, indices] - xf[:, indices]).norm() / xf[:, indices].norm()
    err_sal_full = (full[:, indices] - xf[:, indices]).norm() / xf[:, indices].norm()

    assert err_base < 0.10, f"base reconstruction too lossy: {err_base:.4f}"
    # The residue must sharply improve the SALIENT channels.
    assert err_sal_full < 0.35 * err_sal_base, (
        f"residue does not improve salient channels: "
        f"base={err_sal_base:.5f} full={err_sal_full:.5f} "
        f"(ratio={ratio}, elts={elts_mode}, m={m})"
    )


def test_kext_pack8_pack16_bitwise_identical():
    """pack16 is a thread-mapping change only; output must be bit-identical
    to pack8 (same data bytes, same SF placement)."""
    k, per_block = 2048, 2
    num_salient = k * per_block // 8
    torch.manual_seed(11)
    x = torch.randn(64, k, dtype=torch.bfloat16, device="cuda")
    scale = torch.tensor([448.0 * 6.0 / x.float().abs().max()], device="cuda")
    masks = indices_to_channel_masks(salient_indices_for(k, per_block), k)

    d8, s8 = scaled_fp4_quant_with_mask(x, scale, masks, num_salient, elts_mode=8)
    d16, s16 = scaled_fp4_quant_with_mask(x, scale, masks, num_salient, elts_mode=16)
    assert torch.equal(d8, d16), "pack8 vs pack16 data bytes differ"
    # SF buffers cover (M_pad, n_ext); rows beyond M are uninitialized in
    # both, so compare the decoded in-range grid.
    m, n_ext = x.shape[0], k + num_salient
    g8 = sf_unswizzle(s8, (m + 127) // 128 * 128, n_ext)[:m]
    g16 = sf_unswizzle(s16, (m + 127) // 128 * 128, n_ext)[:m]
    assert torch.equal(g8, g16), "pack8 vs pack16 SF grids differ"


def test_kext_scalar_input_scale_matches_single_element_vector():
    """Real ModelOpt checkpoints store per-tensor scales as 0-D parameters."""
    k, per_block = 2048, 1
    num_salient = k * per_block // 8
    torch.manual_seed(12)
    x = torch.randn(33, k, dtype=torch.bfloat16, device="cuda")
    scalar_scale = (448.0 * 6.0 / x.float().abs().max()).float()
    masks = indices_to_channel_masks(salient_indices_for(k, per_block), k)

    scalar_data, scalar_sf = scaled_fp4_quant_with_mask(
        x, scalar_scale, masks, num_salient
    )
    vector_data, vector_sf = scaled_fp4_quant_with_mask(
        x, scalar_scale.reshape(1), masks, num_salient
    )

    assert torch.equal(scalar_data, vector_data)
    n_ext = k + num_salient
    scalar_sf_grid = sf_unswizzle(scalar_sf, 128, n_ext)[: x.shape[0]]
    vector_sf_grid = sf_unswizzle(vector_sf, 128, n_ext)[: x.shape[0]]
    assert torch.equal(scalar_sf_grid, vector_sf_grid)


def test_kext_rejects_unsupported_ratio():
    k = 2048
    x = torch.randn(4, k, dtype=torch.bfloat16, device="cuda")
    scale = torch.ones(1, device="cuda")
    masks = torch.zeros(k // 8, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="ratio 1.0 is mext_r1"):
        scaled_fp4_quant_with_mask(x, scale, masks, k)  # ratio 1.0
    with pytest.raises(ValueError, match="Unsupported salient ratio"):
        scaled_fp4_quant_with_mask(x, scale, masks, k * 3 // 8)
