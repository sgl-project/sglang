"""Block-scaled CuTeDSL GEMMs against independent dequantized references.
The scale atoms follow tcgen05 128x4 order."""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.quant_ref_utils import (
    quantize_mx_ref,
    quantize_nvfp4_per_token_ref,
    quantize_nvfp4_static_ref,
)

# Block-scaled MMA requires SM100. These tests use one GPU.
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _skip_unless_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("block-scaled tcgen05 MMA needs SM100")


FORMATS = {
    "mxfp8": dict(elem="e4m3", vec=32),
    "mxfp4": dict(elem="e2m1", vec=32),
    "nvfp4": dict(elem="e2m1", vec=16),
}


def _quantize(fmt: str, x: torch.Tensor, global_scale=None):
    if fmt == "nvfp4":
        return quantize_nvfp4_static_ref(x, global_scale)
    return quantize_mx_ref(x, FORMATS[fmt]["vec"], FORMATS[fmt]["elem"])


# Plain per-token NVFP4 uses D = 6 * 448.
_PER_TOKEN_DENOMINATOR = 6.0 * 448.0


# Row indices of the special cases that _per_token_rows places first.
_ROW_ZERO = 0
_ROWS_TINY = (1, 2, 3, 4)  # amax 2^-133, 2^-126, 2^-120, 2^-100
_ROW_BF16_MAX = 5
_ROW_ZERO_SCALE_BLOCK = 6  # block 1 at 2^-20 under a row amax of 1
_ROW_SUBNORMAL_SCALE_BLOCK = 7  # block 1 at 2^-18: E4M3 scale 2^-9
_ROW_FLAT = 8  # every element equal: every block scale saturates at 448


def _per_token_rows(k: int, generator: torch.Generator) -> torch.Tensor:
    """BF16 edge cases followed by normal rows with log-uniform scales."""
    rows = torch.randn((96, k), generator=generator)
    rows *= torch.pow(10.0, torch.rand((96, 1), generator=generator) * 6 - 3)
    rows = rows.to(torch.bfloat16)
    rows[_ROW_ZERO].zero_()
    for row, amax in zip(_ROWS_TINY, (2.0**-133, 2.0**-126, 2.0**-120, 2.0**-100)):
        rows[row] = (torch.randn((k,), generator=generator) * amax).to(torch.bfloat16)
        rows[row, 0] = amax
    rows[_ROW_BF16_MAX] = torch.finfo(torch.bfloat16).max
    rows[_ROW_BF16_MAX, 1::2] *= -1
    for row, small in (
        (_ROW_ZERO_SCALE_BLOCK, 2.0**-20),
        (_ROW_SUBNORMAL_SCALE_BLOCK, 2.0**-18),
    ):
        rows[row].zero_()
        rows[row, 0] = 1.0
        rows[row, 16:32] = small
        rows[row, 17] = -small
    rows[_ROW_FLAT] = 0.75
    return rows


def _unpack_scale_atoms(atoms, rows: int, k: int, vec: int, rows_per_slab: int):
    """[rows, k // vec] scale bytes read back from the tcgen05 atom order."""
    groups = k // vec
    row_atoms, k_atoms = -(-rows_per_slab // 128), -(-groups // 4)
    r = torch.arange(rows, device=atoms.device)
    j = torch.arange(groups, device=atoms.device)
    slab, slab_row = r // rows_per_slab, r % rows_per_slab
    row_block = (slab * row_atoms + slab_row // 128)[:, None]
    row_byte = ((slab_row % 32) * 16 + (slab_row // 32 % 4) * 4)[:, None]
    offs = (row_block * k_atoms + (j // 4)[None, :]) * 512 + row_byte + (j % 4)[None, :]
    return atoms[offs.long()]


@pytest.mark.parametrize(
    "fmt,k", [("mxfp8", 640), ("mxfp4", 5120), ("nvfp4", 640), ("nvfp4_pt", 5120)]
)
def test_blockscaled_row_movers_match_row_quantizer(fmt, k):
    """Check the quantizer and both fills, including invalid pairs."""
    spec = FORMATS.get(fmt, FORMATS["nvfp4"])
    per_token = fmt == "nvfp4_pt"
    if spec["elem"] == "e2m1":
        _skip_unless_sm100()  # the E2M1 conversion needs the sm_100 cvt
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.dispatch_contiguous import (
        contiguous_m_pad_ceiling,
        dispatch_fill_rows_contiguous_blockscaled,
        dispatch_layout_contiguous,
    )
    from sglang.kernels.ops.lora.moe.dispatch_masked import (
        dispatch_fill_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    vec, elem = spec["vec"], spec["elem"]
    device = torch.device("cuda")
    # Force live rows on both sides of a masked slab's 128-row scale-atom boundary.
    crosses_atom = fmt == "mxfp4"
    num_tokens, top_k, num_experts = (132 if crosses_atom else 24), 4, 8
    generator = torch.Generator().manual_seed(0x5B1F)
    scores = torch.rand((num_tokens, num_experts), generator=generator)
    if crosses_atom:
        scores[:, 0] = 2.0
    topk_ids = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)
    topk_ids[0] = -1  # a token without experts
    topk_ids[1, 2] = -1  # one dropped pair
    topk_ids = topk_ids.to(device)
    x = (torch.randn((num_tokens, k), generator=generator) * 0.3).to(torch.bfloat16)
    x[2].zero_()
    x[3, ::128] = 32
    if crosses_atom:
        x[4, :32] = 0
        x[4, :8] = (
            torch.tensor([0.75, -0.25, 1.25, -1.75, 2.5, -3.5, 5.0, 6.0]) * 2.0**-3
        )
    x = x.to(device)
    global_scale = 0.3 if fmt == "nvfp4" else None
    if not per_token:
        ref_global = (
            torch.tensor([global_scale], device=device)
            if global_scale is not None
            else None
        )
        ref_bytes, ref_scales, _ = _quantize(fmt, x, ref_global)
        ref_bytes = ref_bytes.view(torch.uint8)
        if crosses_atom:
            assert ref_bytes[4, :4].tolist() == [0x82, 0xC2, 0xE4, 0x76]
    pair_to_row = torch.empty(num_tokens * top_k, dtype=torch.int32, device=device)
    valid = topk_ids.view(-1) >= 0
    tokens = torch.arange(num_tokens, device=device).repeat_interleave(top_k)[valid]

    def rows(num_rows, rows_per_slab):
        data = torch.full(
            (num_rows, k // 2 if elem == "e2m1" else k), 0x55, dtype=torch.uint8
        ).to(device)
        if elem != "e2m1":
            data = data.view(torch.float8_e4m3fn)
        scales = torch.full(
            (scale_atom_bytes(rows_per_slab, k, vec, num_rows // rows_per_slab),),
            0x55,
            dtype=torch.uint8,
        ).to(device)
        return BlockScaledRows(
            data=data,
            scales=scales,
            sf_vec_size=vec,
            rows_per_slab=rows_per_slab,
            global_scale=global_scale,
            row_scales=(
                torch.full((num_rows,), -1.0, device=device) if per_token else None
            ),
            row_denominator=_PER_TOKEN_DENOMINATOR if per_token else 0.0,
        )

    for domain in ("masked", "contiguous"):
        if domain == "masked":
            m_max = 160 if crosses_atom else 32
            masked_m = torch.empty(num_experts, dtype=torch.int32, device=device)
            target = rows(num_experts * m_max, m_max)
            dispatch_fill_masked_blockscaled(
                x,
                topk_ids,
                top_k,
                masked_m_out=masked_m,
                pair_to_row_out=pair_to_row,
                rows_out=target,
            )
            assert torch.equal(
                masked_m,
                torch.bincount(
                    topk_ids.view(-1)[valid].long(), minlength=num_experts
                ).to(torch.int32),
            )
            if crosses_atom:
                expert_zero_rows = pair_to_row[topk_ids.view(-1) == 0]
                assert bool((expert_zero_rows == 127).any())
                assert bool((expert_zero_rows == 128).any())
        else:
            counts = torch.empty(num_experts, dtype=torch.int32, device=device)
            offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
            m_pad = contiguous_m_pad_ceiling(pair_to_row.numel(), num_experts, 128)
            dispatch_layout_contiguous(
                x,
                topk_ids,
                num_experts,
                top_k,
                128,
                seg_counts_out=counts,
                seg_offsets_out=offsets,
                pair_to_row_out=pair_to_row,
            )
            target = rows(m_pad, m_pad)
            dispatch_fill_rows_contiguous_blockscaled(
                x,
                topk_ids,
                pair_to_row,
                rows_out=target,
                token_scales=(
                    torch.empty(num_tokens, device=device) if per_token else None
                ),
            )
        assert bool(torch.all(pair_to_row[~valid] == -1))
        # The row quantizer over each token's row at its destination.
        reference = rows(target.data.shape[0], target.rows_per_slab)
        source = torch.zeros(
            (target.data.shape[0], k), dtype=torch.bfloat16, device=device
        )
        dst = pair_to_row[valid].long()
        source[dst] = x[tokens]
        quantize_rows(source, reference, pair_to_row)
        if not per_token:
            actual_scales = _unpack_scale_atoms(
                reference.scales, source.shape[0], k, vec, target.rows_per_slab
            )
            assert torch.equal(reference.data.view(torch.uint8)[dst], ref_bytes[tokens])
            assert torch.equal(actual_scales[dst], ref_scales[tokens])
        assert torch.equal(
            target.data.view(torch.uint8), reference.data.view(torch.uint8)
        ), domain
        assert torch.equal(target.scales, reference.scales), domain
        if per_token:
            # Live rows carry the token's stored scale; dead rows keep -1.
            assert torch.equal(target.row_scales, reference.row_scales), domain
            live_rows = pair_to_row[valid].long()
            assert bool((target.row_scales[live_rows] >= 0).all()), domain


def test_triton_row_quantizer_addresses_rows_past_2_31_elements():
    """Offsets past 2**31 elements must address the intended expert rows."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    device = torch.device("cuda")
    k, vec = 128, 32
    rows = 2**31 // k + 2048  # the last rows' offsets pass 2**31 elements
    if torch.cuda.mem_get_info(device)[0] < 8 * 2**30:
        pytest.skip("needs 8 GiB of free device memory")
    x = torch.zeros((rows, k), dtype=torch.bfloat16, device=device)
    generator = torch.Generator(device="cpu").manual_seed(3)
    pattern = (torch.randn(k, generator=generator) * 0.3).to(torch.bfloat16)
    probes = torch.tensor(
        [0, 2**31 // k - 1, 2**31 // k, rows - 1], dtype=torch.int32, device=device
    )
    x[probes.long()] = pattern.to(device)
    out = torch.empty((rows, k), dtype=torch.float8_e4m3fn, device=device)
    row_atoms = -(-rows // 128)  # k // vec = 4 groups: one atom column
    sf = torch.zeros(row_atoms * 512, dtype=torch.uint8, device=device)
    target = BlockScaledRows(data=out, scales=sf, sf_vec_size=vec, rows_per_slab=rows)
    quantize_rows(x, target, probes)
    torch.cuda.synchronize()

    codes = out.view(torch.uint8)[probes.long()]
    assert codes[0].any()
    assert torch.equal(codes, codes[:1].expand_as(codes))

    def scale_bytes(row: int) -> torch.Tensor:
        base = (row // 128) * 512 + (row % 32) * 16 + (row // 32 % 4) * 4
        return sf[base : base + 4]

    reference = scale_bytes(0)
    assert reference.any()
    for row in probes.tolist()[1:]:
        assert torch.equal(scale_bytes(row), reference), row


@pytest.mark.parametrize("use_4over6", [False, True], ids=["plain", "4over6"])
def test_per_token_row_quantizer_matches_reference(use_4over6):
    """Check edge values and the exact 4over6 scale choice."""
    _skip_unless_sm100()
    from types import SimpleNamespace

    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0x9E7)
    recipe = None
    row_denominator = _PER_TOKEN_DENOMINATOR
    if use_4over6:
        recipe = SimpleNamespace(e4m3_max=256, err_mode="MSE", err_use_fast_math=True)
        row_denominator = 6 * recipe.e4m3_max
        k, rows_per_slab = 128, 3
        x = torch.zeros((rows_per_slab, k), device=device, dtype=torch.bfloat16)
        x[0, :3] = torch.tensor([1536.0, 576.0, 1152.0], device=device)
        x[1, 0] = 1.0
        x[1, 16:19] = torch.tensor([3.0, -3.0, 1.5], device=device) * 2**-20
    else:
        k, rows_per_slab = 640, 32
        x = _per_token_rows(k, g).to(device)
    num_rows = x.shape[0]
    pair_to_row = torch.randperm(num_rows, generator=g).to(torch.int32)
    pair_to_row[pair_to_row == num_rows - 1] = -1  # Keep every crafted row live.
    pair_to_row = pair_to_row.to(device)
    target = BlockScaledRows(
        data=torch.full((num_rows, k // 2), 0x55, dtype=torch.uint8, device=device),
        scales=torch.full(
            (scale_atom_bytes(rows_per_slab, k, 16, num_rows // rows_per_slab),),
            0x55,
            dtype=torch.uint8,
            device=device,
        ),
        sf_vec_size=16,
        rows_per_slab=rows_per_slab,
        row_scales=torch.full((num_rows,), -1.0, device=device),
        row_denominator=row_denominator,
        use_4over6=use_4over6,
        err_mode="MAE" if recipe is None else recipe.err_mode,
        err_use_fast_math=False if recipe is None else recipe.err_use_fast_math,
    )
    quantize_rows(x, target, pair_to_row)
    torch.cuda.synchronize()
    data_ref, sf_ref, s_ref, _, _ = quantize_nvfp4_per_token_ref(x, recipe)
    live = torch.zeros(num_rows, dtype=torch.bool, device=device)
    live[pair_to_row[pair_to_row >= 0].long()] = True
    dead = ~live
    assert dead.any()

    assert torch.equal(target.data[live], data_ref[live])
    unpacked = _unpack_scale_atoms(target.scales, num_rows, k, 16, rows_per_slab)
    assert torch.equal(unpacked[live], sf_ref[live])
    assert torch.equal(target.row_scales[live], s_ref[live])
    assert bool(torch.isfinite(target.row_scales).all())
    assert bool((target.data[dead] == 0x55).all())
    assert bool((unpacked[dead] == 0x55).all())
    assert bool((target.row_scales[dead] == -1).all())

    if use_4over6:
        assert unpacked[0, 0] == 0x7C  # The 4-based scale of 384 lowers the error.
        assert unpacked[1, 1] == 0x01  # The 6-based candidate rounds to zero.
        return

    assert s_ref[_ROW_ZERO] == 0
    assert bool((data_ref[_ROW_ZERO] == 0).all()) and bool(
        (sf_ref[_ROW_ZERO] == 0).all()
    )
    tiny = torch.finfo(torch.float32).tiny
    for row in _ROWS_TINY[:3]:
        assert 0 < s_ref[row] < tiny, row  # a subnormal stored scale
        assert bool((data_ref[row] != 0).any()), row
    assert s_ref[_ROWS_TINY[3]] >= tiny
    assert sf_ref[_ROW_ZERO_SCALE_BLOCK, 1] == 0
    assert bool((data_ref[_ROW_ZERO_SCALE_BLOCK, 8:16] == 0).all())
    assert (
        sf_ref[_ROW_ZERO_SCALE_BLOCK, 0] > 0 and data_ref[_ROW_ZERO_SCALE_BLOCK, 0] != 0
    )
    assert 0 < int(sf_ref[_ROW_SUBNORMAL_SCALE_BLOCK, 1]) < 8  # a subnormal E4M3
    assert bool((data_ref[_ROW_SUBNORMAL_SCALE_BLOCK, 8:16] != 0).any())
    assert bool((sf_ref[_ROW_FLAT] == 0x7E).all())  # 448 everywhere


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
