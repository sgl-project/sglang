"""Block-scaled CuTeDSL grouped GEMMs against exact dequantized references.

MXFP8 (E4M3 x E8M0/32), MXFP4 (E2M1 x E8M0/32) and NVFP4 (E2M1 x E4M3/16)
operands are quantized on the host with the same rules as the Triton row
quantizer; the scale atoms follow the tcgen05 128x4 block order.
"""

from __future__ import annotations

import time

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
# Every test but four (the MXFP8 row movers, the alpha-split alignment check
# and the row quantizer's 64-bit addressing) needs SM100 (block-scaled tcgen05
# MMA) and skips on the H100 lane, so the file is also registered on the SM100
# lane. It runs on one GPU: 4-gpu-b200 is the kernel lane's only SM100 pool, so
# its four GPUs are allocation granularity, not multi-GPU coverage.
register_cuda_ci(est_time=250, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

_E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _e2m1_codes(magnitude: torch.Tensor) -> torch.Tensor:
    """Nearest E2M1 code; ties go to the even code, as the hardware cvt.rn does."""
    mid = _E2M1_MID.to(magnitude.device)
    idx = torch.bucketize(magnitude, mid)  # a tie lands on the lower code
    tie = (idx < 7) & (magnitude == mid[idx.clamp(max=6)]) & (idx % 2 == 1)
    return idx + tie.long()


def _skip_unless_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("block-scaled tcgen05 MMA needs SM100")


def _quant_mx_ref(x: torch.Tensor, vec: int, elem: str):
    """[..., K] -> (data, scale bytes [..., K/vec] uint8, exact dequant fp32)."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // vec, vec)
    amax = groups.abs().amax(dim=-1)
    elem_max, elem_max_log2 = (6.0, 2) if elem == "e2m1" else (448.0, 8)
    # Smallest power of two with amax <= elem_max * 2**e, as the row quantizer.
    floor_log2 = torch.frexp(amax)[1].float() - 1
    exponent = floor_log2 - elem_max_log2
    exponent = exponent + (amax > elem_max * torch.exp2(exponent)).float()
    exponent = torch.where(amax > 0, exponent, torch.zeros_like(amax))
    exponent = exponent.clamp(-126.0, 127.0)
    scale = torch.exp2(exponent)
    scaled = groups / scale[..., None]
    if elem == "e2m1":
        idx = _e2m1_codes(scaled.abs())
        sign = scaled < 0
        table = _E2M1.to(x.device)
        values = torch.where(sign, -table[idx], table[idx])
        codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
        data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    else:
        q = scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        values = q.float()
        data = q.reshape(*lead, k)
    dequant = (values * scale[..., None]).reshape(*lead, k)
    return data, (exponent + 127).to(torch.uint8), dequant


def _quant_nvfp4_ref(
    x: torch.Tensor, global_scale: torch.Tensor, dequant_dtype=torch.float32
):
    """NVFP4 with E4M3 group-16 scales under a per-slab global scale. The
    dequantized values are exact in float64 (the fp32 form rounds the block
    scale times the global scale)."""
    *lead, k = x.shape
    groups = x.float().reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    gs = global_scale.view(-1, *([1] * (len(lead) - 1)), 1).float()
    block_scale = (
        (amax / (6.0 * gs)).clamp(min=2.0**-6, max=448.0).to(torch.float8_e4m3fn)
    )
    denom = block_scale.float() * gs
    scaled = (groups * (1.0 / denom)[..., None]).clamp(-6.0, 6.0)
    idx = _e2m1_codes(scaled.abs())
    sign = scaled < 0
    table = _E2M1.to(x.device)
    values = torch.where(sign, -table[idx], table[idx])
    codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    denom = block_scale.to(dequant_dtype) * gs.to(dequant_dtype)
    dequant = (values.to(dequant_dtype) * denom[..., None]).reshape(*lead, k)
    return data, block_scale.view(torch.uint8), dequant


FORMATS = {
    "mxfp8": dict(elem="e4m3", vec=32),
    "mxfp4": dict(elem="e2m1", vec=32),
    "nvfp4": dict(elem="e2m1", vec=16),
}


def _quantize(fmt: str, x: torch.Tensor, global_scale=None):
    if fmt == "nvfp4":
        return _quant_nvfp4_ref(x, global_scale)
    return _quant_mx_ref(x, FORMATS[fmt]["vec"], FORMATS[fmt]["elem"])


# The default per-token recipe: D = 6 * e4m3_max with the 448 reference.
_PER_TOKEN_DENOMINATOR = 6.0 * 448.0


def _quant_nvfp4_per_token_ref(
    x: torch.Tensor, denominator: float = _PER_TOKEN_DENOMINATOR
):
    """Per-token NVFP4 as the kernels evaluate it: the stored row scale is
    fl(row_amax / D); block scales fl(amax / (6 s)) saturate at 448 and cast
    to E4M3; codes are e2m1((x / s) / dec(bs)). A zero row, and a block whose
    E4M3 scale rounds to zero, give scale 0 and codes 0. Returns the packed
    codes, the scale bytes [.., k // 16], the row scales and the dequantized
    values in float64, exact (zero rows dequantize to zero)."""
    *lead, k = x.shape
    xf = x.float()
    # A tensor divisor: dividing by a Python float multiplies by its
    # reciprocal on CUDA, which is not the correctly rounded division.
    s = xf.abs().amax(dim=-1) / torch.tensor(denominator, device=x.device)
    s_safe = torch.where(s > 0, s, torch.ones_like(s))
    groups = xf.reshape(*lead, k // 16, 16)
    amax = groups.abs().amax(dim=-1)
    block_scale = (amax / (6.0 * s_safe)[..., None]).clamp(max=448.0)
    block_scale = block_scale.to(torch.float8_e4m3fn)
    decoded = block_scale.float()
    live = decoded > 0
    denom = torch.where(live, decoded, torch.ones_like(decoded))
    scaled = (groups / s_safe[..., None, None]) / denom[..., None]
    scaled = torch.where(live[..., None], scaled, torch.zeros_like(scaled))
    scaled = scaled.clamp(-6.0, 6.0)
    idx = _e2m1_codes(scaled.abs())
    # The hardware conversion keeps the sign of a negative zero (a BF16 -0.0
    # input, or a value that underflows), so the code's sign bit follows it.
    sign = torch.signbit(scaled)
    table = _E2M1.to(x.device)
    values = torch.where(sign, -table[idx], table[idx])
    codes = (idx | (sign.long() << 3)).to(torch.uint8).reshape(*lead, k)
    data = codes[..., 0::2] | (codes[..., 1::2] << 4)
    dequant = (
        values.double() * denom.double()[..., None] * s.double()[..., None, None]
    ).reshape(*lead, k)
    return data, block_scale.view(torch.uint8), s, dequant


# Row indices of the special cases that _per_token_rows places first.
_ROW_ZERO = 0
_ROWS_TINY = (1, 2, 3, 4)  # amax 2^-133, 2^-126, 2^-120, 2^-100
_ROW_BF16_MAX = 5
_ROW_ZERO_SCALE_BLOCK = 6  # block 1 at 2^-20 under a row amax of 1
_ROW_SUBNORMAL_SCALE_BLOCK = 7  # block 1 at 2^-18: E4M3 scale 2^-9
_ROW_FLAT = 8  # every element equal: every block scale saturates at 448


def _per_token_rows(k: int, generator: torch.Generator) -> torch.Tensor:
    """BF16 rows for the per-token cases: the special rows above, then rows
    with an amax drawn log-uniformly over 1e-3..1e3."""
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


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_triton_row_quantizer_matches_reference(fmt):
    """The Triton quantizer reproduces the host rule and atom order exactly."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        pack_scale_atoms,
        quantize_rows,
    )

    vec, elem = FORMATS[fmt]["vec"], FORMATS[fmt]["elem"]
    device = torch.device("cuda")
    slabs, rows, k = 3, 200, 512
    x = (torch.randn(slabs, rows, k, device=device) * 0.3).to(torch.bfloat16)
    x[0, 5] = 0  # an all-zero row keeps a finite scale
    # One group of exact E2M1 ties (amax = 6 * 2**-3) rounds to the even codes.
    x[1, 7, :32] = 0
    x[1, 7, :8] = (
        torch.tensor([0.75, -0.25, 1.25, -1.75, 2.5, -3.5, 5.0, 6.0]) * 2.0**-3
    )
    pair_to_row = torch.arange(slabs * rows, dtype=torch.int32, device=device)
    global_scale = 0.3
    data_ref, sf_ref, _ = _quantize(
        fmt, x, torch.full((slabs,), global_scale, device=device)
    )
    out = torch.empty(
        (slabs * rows, k // 2 if elem == "e2m1" else k),
        dtype=torch.uint8 if elem == "e2m1" else torch.float8_e4m3fn,
        device=device,
    )
    sf_out = torch.zeros(
        scale_atom_bytes(rows, k, vec, slabs), dtype=torch.uint8, device=device
    )
    target = BlockScaledRows(
        data=out,
        scales=sf_out,
        sf_vec_size=vec,
        rows_per_slab=rows,
        global_scale=global_scale if fmt == "nvfp4" else None,
    )
    quantize_rows(x.view(slabs * rows, k), target, pair_to_row)
    assert torch.equal(
        out.view(torch.uint8), data_ref.view(torch.uint8).view(slabs * rows, -1)
    )
    assert torch.equal(sf_out, pack_scale_atoms(sf_ref))


@pytest.mark.parametrize("fmt", sorted(FORMATS) + ["nvfp4_pt"])
@pytest.mark.parametrize("k", [640, 5120])
def test_blockscaled_row_movers_match_row_quantizer(fmt, k):
    """Both dispatch fills (expert slabs and contiguous segments) write the row
    quantizer's codes and scale atoms bitwise over whole column chunks (K=5120
    is five chunks of 1024, K=640 five of 128); other rows stay untouched.
    ``nvfp4_pt`` is the per-token mode: the stored row scales match too."""
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
    num_tokens, top_k, num_experts = 24, 4, 8
    generator = torch.Generator().manual_seed(0x5B1F)
    scores = torch.rand((num_tokens, num_experts), generator=generator)
    topk_ids = torch.topk(scores, top_k, dim=1).indices.to(torch.int32)
    topk_ids[0] = -1  # a token without experts
    topk_ids[1, 2] = -1  # one dropped pair
    topk_ids = topk_ids.to(device)
    x = (torch.randn((num_tokens, k), generator=generator) * 0.3).to(torch.bfloat16)
    x[2].zero_()
    x[3, ::128] = 32
    x = x.to(device)
    global_scale = 0.3 if fmt == "nvfp4" else None
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
            m_max = 32
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
        source[pair_to_row[valid].long()] = x[tokens]
        quantize_rows(source, reference, pair_to_row)
        assert torch.equal(
            target.data.view(torch.uint8), reference.data.view(torch.uint8)
        ), domain
        assert torch.equal(target.scales, reference.scales), domain
        if per_token:
            # Live rows carry the token's stored scale; dead rows keep -1.
            assert torch.equal(target.row_scales, reference.row_scales), domain
            live_rows = pair_to_row[valid].long()
            assert bool((target.row_scales[live_rows] >= 0).all()), domain


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("rows_mode", ["masked", "contiguous"])
def test_blockscaled_grouped_gemm_matches_dequantized_reference(fmt, rows_mode):
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_contiguous_blockscaled,
        prepare_masked_blockscaled,
        scale_atom_bytes,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_contiguous,
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    vec = FORMATS[fmt]["vec"]
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(3)
    experts, n, k, m_max = 4, 256, 512, 256
    weights = (torch.randn(experts, n, k, generator=g) * 0.05).to(device)
    tokens = (torch.randn(experts, m_max, k, generator=g) * 0.5).to(device)
    w_gs = torch.full((experts,), 0.01, device=device)
    t_gs = torch.full((experts,), 0.1, device=device)
    w_data, w_sf, w_deq = _quantize(fmt, weights, w_gs)
    t_data, t_sf, t_deq = _quantize(fmt, tokens, t_gs)
    alpha, alpha_rows_per_half = None, 1
    if fmt == "nvfp4":
        # Global scales reach the output through the per-expert alpha epilogue;
        # the second half of the weight rows doubles its alpha to prove the
        # gate/up selection.
        alpha = torch.stack([w_gs * t_gs, 2 * w_gs * t_gs], dim=1).contiguous()
        alpha_rows_per_half = n // 2
        w_deq = w_deq.clone()
        w_deq[:, n // 2 :] *= 2
    sf_weights = pack_scale_atoms(w_sf)
    assert sf_weights.numel() == scale_atom_bytes(n, k, vec, experts)
    config = GroupedGemmConfig(mma_tiler_mn=(128, 128), persistent_clusters=8)
    masked_m = torch.tensor([200, 0, 128, 7], dtype=torch.int32, device=device)

    if rows_mode == "masked":
        sf_tokens = pack_scale_atoms(t_sf)
        c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
        schedule, tiles, _, _ = build_dual_stage_schedules_masked(
            masked_m,
            m_max=m_max,
            token_width=128,
            n_gemm1=n,
            n_gemm2=n,
            output_width=128,
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
        )
        prepared = prepare_masked_blockscaled(
            t_data,
            w_data,
            sf_tokens,
            sf_weights,
            c,
            masked_m,
            config=config,
            sf_vec_size=vec,
            alpha=alpha,
            alpha_rows_per_half=alpha_rows_per_half,
            direct_schedule=schedule,
            schedule_tiles=tiles,
        )
        prepared.launch()
        torch.cuda.synchronize()
        for e in range(experts):
            rows = int(masked_m[e])
            if rows == 0:
                continue
            ref = t_deq[e, :rows] @ w_deq[e].T
            assert _rel_l2(c[e, :rows].float(), ref) < 1.5e-2, (fmt, e)
    else:
        seg_counts = masked_m
        seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
        seg_offsets[1:] = torch.cumsum(((seg_counts + 127) // 128) * 128, 0)
        m_ceil = int(seg_offsets[-1])
        flat = torch.zeros((m_ceil, k), device=device)
        for e in range(experts):
            start, rows = int(seg_offsets[e]), int(seg_counts[e])
            flat[start : start + rows] = tokens[e, :rows]
        f_data, f_sf, f_deq = _quantize(fmt, flat.unsqueeze(0), t_gs[:1])
        f_data = f_data[0]
        sf_tokens = pack_scale_atoms(f_sf)
        assert sf_tokens.numel() == scale_atom_bytes(m_ceil, k, vec, 1)
        c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
        schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
            seg_counts,
            m_pad_ceiling=m_ceil,
            max_expert_rows=m_max,
            m_alignment=128,
            token_width=128,
            n_gemm1=n,
            n_gemm2=n,
            output_width=128,
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
        )
        prepared = prepare_contiguous_blockscaled(
            f_data,
            w_data,
            sf_tokens,
            sf_weights,
            c,
            seg_offsets,
            config=config,
            sf_vec_size=vec,
            alpha=alpha,
            alpha_rows_per_half=alpha_rows_per_half,
            direct_schedule=schedule,
            schedule_tiles=tiles,
        )
        prepared.launch()
        torch.cuda.synchronize()
        for e in range(experts):
            start, rows = int(seg_offsets[e]), int(seg_counts[e])
            if rows == 0:
                continue
            ref = f_deq[0, start : start + rows] @ w_deq[e].T
            assert _rel_l2(c[start : start + rows].float(), ref) < 1.5e-2, (fmt, e)


def test_blockscaled_alpha_split_must_align_with_output_tiles():
    """The epilogue applies one alpha per output tile, so a gate/up boundary
    inside a tile, or more halves than alpha columns, is refused before compiling."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_contiguous_blockscaled,
        scale_atom_bytes,
    )

    device = torch.device("cuda")
    experts, n, k, m_ceil = 1, 384, 256, 128
    a = torch.zeros((m_ceil, k // 2), dtype=torch.uint8, device=device)
    b = torch.zeros((experts, n, k // 2), dtype=torch.uint8, device=device)
    sf_tokens = torch.zeros(
        scale_atom_bytes(m_ceil, k, 16), dtype=torch.uint8, device=device
    )
    sf_weights = torch.zeros(
        scale_atom_bytes(n, k, 16, experts), dtype=torch.uint8, device=device
    )
    c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
    seg_offsets = torch.zeros((experts + 1,), dtype=torch.int32, device=device)
    alpha = torch.ones((experts, 2), dtype=torch.float32, device=device)
    # A boundary at row 192 falls inside the tile covering rows 128-255; a
    # boundary every 128 rows needs three columns.
    for rows_per_half in (192, 128):
        with pytest.raises(ValueError, match="alpha changes every"):
            prepare_contiguous_blockscaled(
                a,
                b,
                sf_tokens,
                sf_weights,
                c,
                seg_offsets,
                config=GroupedGemmConfig(mma_tiler_mn=(128, 128)),
                sf_vec_size=16,
                alpha=alpha,
                alpha_rows_per_half=rows_per_half,
            )


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_fused_activation_quant_matches_row_quantizer(fmt):
    """act_delta's block-scaled output equals the row quantizer on its BF16 rows."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.activation_delta import act_delta
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    num_tokens, top_k, num_experts, inter = 7, 3, 8, 384
    num_pairs = num_tokens * top_k
    vec, elem = FORMATS[fmt]["vec"], FORMATS[fmt]["elem"]
    gateup = torch.randn(num_pairs, 2 * inter, dtype=torch.bfloat16, device=device)
    delta = 0.1 * torch.randn(
        num_tokens, top_k, 2 * inter, dtype=torch.bfloat16, device=device
    )
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    topk_ids[0, 0] = -1
    pair_to_row = torch.arange(num_pairs, dtype=torch.int32, device=device)
    pair_to_row[0] = -1

    def rows_spec():
        data = torch.zeros(
            (num_pairs, inter // 2 if elem == "e2m1" else inter),
            dtype=torch.uint8 if elem == "e2m1" else torch.float8_e4m3fn,
            device=device,
        )
        scales = torch.zeros(
            scale_atom_bytes(num_pairs, inter, vec), dtype=torch.uint8, device=device
        )
        return BlockScaledRows(
            data=data,
            scales=scales,
            sf_vec_size=vec,
            rows_per_slab=num_pairs,
            global_scale=0.3 if fmt == "nvfp4" else None,
        )

    def run(act_quant):
        act = torch.zeros(num_pairs, inter, dtype=torch.bfloat16, device=device)
        pairs = torch.zeros(
            num_tokens, top_k, inter, dtype=torch.bfloat16, device=device
        )
        act_delta(
            gateup,
            delta,
            act,
            pairs,
            pair_to_row,
            topk_ids,
            num_local_experts=num_experts,
            act_quant=act_quant,
        )
        return act

    ref_act = run(None)
    fused = rows_spec()
    assert torch.equal(run(fused), ref_act)
    separate = rows_spec()
    quantize_rows(ref_act, separate, pair_to_row)
    assert torch.equal(fused.data, separate.data)
    assert torch.equal(fused.scales, separate.scales)


def test_triton_row_quantizer_addresses_rows_past_2_31_elements():
    """Row offsets are int64. With int32 ones, a Beam TP4 decode batch of 16k
    tokens (256 slabs x 16384 rows x 640) quantized experts 205-255 wrong."""
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("rows_mode", ["masked", "contiguous"])
def test_blockscaled_epilogue_lora_matches_references(fmt, rows_mode):
    """The down GEMM with the epilogue LoRA: every row of a tile gets its own
    slot's ``bridge @ B_slot[expert]^T`` added in FP32 before the one BF16
    rounding (NVFP4 adds it after alpha, from the second accumulator).
    Against an FP64 target it and the separate path (BF16 base plus BF16
    delta) are each within 1.5e-2 relative L2, and it is at most 5% farther
    than the separate path; zero planes or zero slots reproduce the plain
    kernel bitwise, and rows without an adapter stay base rows."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_contiguous_blockscaled,
        prepare_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_contiguous,
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    vec = FORMATS[fmt]["vec"]
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(11)
    experts, n, k, m_max, slots, rank = 4, 256, 512, 256, 2, 16
    weights = (torch.randn(experts, n, k, generator=g) * 0.05).to(device)
    tokens = (torch.randn(experts, m_max, k, generator=g) * 0.5).to(device)
    w_gs = torch.full((experts,), 0.01, device=device)
    t_gs = torch.full((experts,), 0.1, device=device)
    w_data, w_sf, w_deq = _quantize(fmt, weights, w_gs)
    sf_weights = pack_scale_atoms(w_sf)
    alpha, lora_acc = None, "shared"
    if fmt == "nvfp4":
        # One alpha per expert over every output row, as the down GEMM uses
        # it; the LoRA MMAs fill the second accumulator.
        alpha = (w_gs * t_gs).reshape(experts, 1).contiguous()
        lora_acc = "separate"
    config = GroupedGemmConfig(mma_tiler_mn=(128, 128), persistent_clusters=8)
    masked_m = torch.tensor([200, 0, 128, 7], dtype=torch.int32, device=device)
    # LoRA operands: per (slot, expert) down-B weights, a bridge per row. The
    # bridge is the A kernel's FP32 accumulator, rounded to BF16 once.
    lora_w = (torch.randn(slots, experts, n, rank, generator=g) * 0.2).to(
        device=device, dtype=torch.bfloat16
    )
    bridge = (torch.randn(experts, m_max, rank, generator=g) * 0.5).to(device)
    # Rows alternate slots 0, 1 and no adapter (-1) so tiles mix all three.
    row_slot = (torch.arange(m_max) % 3 - 1).to(device)
    slots_t = torch.tensor([slots], dtype=torch.int32, device=device)

    def planes_for(expert_rows, bridge_rows, slot_rows):
        """Zeroed planes with each row's BF16 bridge on its slot's plane, as
        the down A kernel stores it."""
        planes = torch.zeros(
            (slots, expert_rows, rank), dtype=torch.bfloat16, device=device
        )
        for s in range(slots):
            mask = slot_rows == s
            planes[s, mask] = bridge_rows[mask].to(torch.bfloat16)
        return planes

    def target_and_separate(rows_deq, expert_of_row, slot_rows, bridge_rows, c_base):
        """FP64 target A @ W^T + bridge @ B_slot[e]^T per row (the dequantized
        operands carry the global scales the kernel applies as alpha), and the
        separate path's composition BF16(base) + BF16(BF16(bridge) @ B^T)."""
        base = torch.einsum(
            "rk,rnk->rn", rows_deq.double(), w_deq[expert_of_row].double()
        )
        adapted = (slot_rows >= 0).double()[:, None]
        w_rows = lora_w[slot_rows.clamp(min=0), expert_of_row].double()
        delta = torch.einsum("rq,rnq->rn", bridge_rows.double(), w_rows) * adapted
        delta_bf16_bridge = (
            torch.einsum("rq,rnq->rn", bridge_rows.to(torch.bfloat16).double(), w_rows)
            * adapted
        )
        separate = c_base.double() + delta_bf16_bridge.to(torch.bfloat16).double()
        return base + delta, separate

    if rows_mode == "masked":
        t_data, t_sf, t_deq = _quantize(fmt, tokens, t_gs)
        sf_tokens = pack_scale_atoms(t_sf)
        schedule, tiles, _, _ = build_dual_stage_schedules_masked(
            masked_m,
            m_max=m_max,
            token_width=128,
            n_gemm1=n,
            n_gemm2=n,
            output_width=128,
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
        )
        expert_of_row = torch.arange(experts, device=device).repeat_interleave(m_max)
        slot_rows = row_slot.repeat(experts)
        bridge_rows = bridge.reshape(-1, rank)
        planes = planes_for(experts * m_max, bridge_rows, slot_rows)
        planes = (
            planes.view(slots, experts, m_max, rank)
            .transpose(0, 1)
            .reshape(slots * experts, m_max, rank)
            .contiguous()
        )
        # The kernel indexes planes by slot * experts + expert.
        planes = (
            planes.view(experts, slots, m_max, rank)
            .transpose(0, 1)
            .reshape(slots * experts, m_max, rank)
            .contiguous()
        )
        live = torch.cat(
            [
                torch.arange(e * m_max, e * m_max + int(masked_m[e]), device=device)
                for e in range(experts)
            ]
        )

        def run(lora):
            c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
            prepared = prepare_masked_blockscaled(
                t_data,
                w_data,
                sf_tokens,
                sf_weights,
                c,
                masked_m,
                config=config,
                sf_vec_size=vec,
                alpha=alpha,
                alpha_rows_per_half=1 << 30,
                direct_schedule=schedule,
                schedule_tiles=tiles,
                **(lora or {}),
            )
            prepared.launch()
            torch.cuda.synchronize()
            return c.reshape(-1, n)

        rows_deq = t_deq.reshape(-1, k)
        lora_args = dict(
            lora_w=lora_w.reshape(slots * experts, n, rank),
            lora_planes=planes,
            lora_slots=slots_t,
            lora_acc=lora_acc,
        )
    else:
        seg_counts = masked_m
        seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
        seg_offsets[1:] = torch.cumsum(((seg_counts + 127) // 128) * 128, 0)
        m_ceil = int(seg_offsets[-1])
        flat = torch.zeros((m_ceil, k), device=device)
        expert_of_row = torch.zeros(m_ceil, dtype=torch.long, device=device)
        slot_rows = torch.full((m_ceil,), -1, dtype=torch.long, device=device)
        bridge_rows = torch.zeros((m_ceil, rank), device=device)
        live = []
        for e in range(experts):
            start, rows = int(seg_offsets[e]), int(seg_counts[e])
            flat[start : start + rows] = tokens[e, :rows]
            expert_of_row[start : start + rows] = e
            slot_rows[start : start + rows] = row_slot[:rows]
            bridge_rows[start : start + rows] = bridge[e, :rows]
            live.append(torch.arange(start, start + rows, device=device))
        live = torch.cat(live)
        f_data, f_sf, f_deq = _quantize(fmt, flat.unsqueeze(0), t_gs[:1])
        f_data, rows_deq = f_data[0], f_deq[0]
        sf_tokens = pack_scale_atoms(f_sf)
        schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
            seg_counts,
            m_pad_ceiling=m_ceil,
            max_expert_rows=m_max,
            m_alignment=128,
            token_width=128,
            n_gemm1=n,
            n_gemm2=n,
            output_width=128,
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
        )
        planes = planes_for(m_ceil, bridge_rows, slot_rows)

        def run(lora):
            c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
            prepared = prepare_contiguous_blockscaled(
                f_data,
                w_data,
                sf_tokens,
                sf_weights,
                c,
                seg_offsets,
                config=config,
                sf_vec_size=vec,
                alpha=alpha,
                alpha_rows_per_half=1 << 30,
                direct_schedule=schedule,
                schedule_tiles=tiles,
                **(lora or {}),
            )
            prepared.launch()
            torch.cuda.synchronize()
            return c

        lora_args = dict(
            lora_w=lora_w.reshape(slots * experts, n, rank),
            lora_planes=planes,
            lora_slots=slots_t,
            lora_acc=lora_acc,
        )

    c_base = run(None)
    c_fused = run(lora_args)
    target, separate = target_and_separate(
        rows_deq, expert_of_row, slot_rows, bridge_rows, c_base
    )
    err_fused = _rel_l2(c_fused.double()[live], target[live])
    err_separate = _rel_l2(separate[live], target[live])
    assert err_fused < 1.5e-2 and err_separate < 1.5e-2, (
        fmt,
        rows_mode,
        err_fused,
        err_separate,
    )
    # One rounding after the FP32 sum is never worse than two.
    assert err_fused <= err_separate * 1.05, (fmt, rows_mode, err_fused, err_separate)
    # Rows without an adapter are base rows, bitwise.
    base_rows = live[slot_rows[live] < 0]
    assert torch.equal(c_fused[base_rows], c_base[base_rows])
    # Adapted rows actually moved.
    adapted_rows = live[slot_rows[live] >= 0]
    assert not torch.equal(c_fused[adapted_rows], c_base[adapted_rows])
    # Zero planes, or no slots, add exact zeros.
    zero = dict(lora_args, lora_planes=torch.zeros_like(lora_args["lora_planes"]))
    assert torch.equal(run(zero), c_base)
    none = dict(lora_args, lora_slots=torch.zeros_like(slots_t))
    assert torch.equal(run(none), c_base)
    # Only the first slot: slot-1 rows become base rows.
    first = dict(lora_args, lora_slots=torch.ones_like(slots_t))
    c_first = run(first)
    slot1 = live[slot_rows[live] == 1]
    slot0 = live[slot_rows[live] == 0]
    assert torch.equal(c_first[slot1], c_base[slot1])
    assert torch.equal(c_first[slot0], c_fused[slot0])


def _wait_bounded(seconds: float) -> None:
    """Wait for the current stream with a deadline: a deadlock between the
    kernel's warps would otherwise hang the process."""
    done = torch.cuda.Event()
    done.record()
    deadline = time.monotonic() + seconds
    while not done.query():
        if time.monotonic() > deadline:
            pytest.fail(
                f"the block-scaled GEMM did not finish within {seconds:.0f} s "
                "(LoRA pipeline deadlock?)"
            )
        time.sleep(0.005)


class _EpilogueDownProblem:
    """A down GEMM over four experts (one empty, one partial tile) on two
    persistent CTAs, every row carrying one of eight adapter slots or none,
    with the operands the down A kernel would place for any set of resident
    slots. ``w_gs`` / ``t_gs`` are the NVFP4 global scales (per expert / of
    the layer), ``row_slot`` and ``row_gain`` override each row's slot and
    multiply its bridge."""

    capacity = 8

    def __init__(
        self,
        fmt: str,
        rows_mode: str,
        *,
        width: int = 128,
        masked_m: tuple = (200, 0, 128, 7),
        w_gs: tuple = (0.01,) * 4,
        t_gs: float = 0.1,
        row_slot: torch.Tensor | None = None,
        row_gain: torch.Tensor | None = None,
    ) -> None:
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            GroupedGemmConfig,
            prepare_contiguous_blockscaled,
            prepare_masked_blockscaled,
        )
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            build_dual_stage_schedules_masked,
        )
        from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

        vec = FORMATS[fmt]["vec"]
        device = torch.device("cuda")
        g = torch.Generator(device="cpu").manual_seed(13)
        experts, n, k, m_max, rank = len(masked_m), 256, 512, 256, 16
        self.experts, self.n, self.rank = experts, n, rank
        weights = (torch.randn(experts, n, k, generator=g) * 0.05).to(device)
        tokens = (torch.randn(experts, m_max, k, generator=g) * 0.5).to(device)
        w_gs = torch.tensor(w_gs, dtype=torch.float32, device=device)
        t_gs = torch.full((experts,), t_gs, device=device)
        # The codes sit at the same magnitudes under every weight global scale.
        weights = weights * (w_gs / 0.01)[:, None, None]
        w_data, w_sf, self.w_deq = _quantize(fmt, weights, w_gs)
        sf_weights = pack_scale_atoms(w_sf)
        alpha, self.lora_acc = None, "shared"
        if fmt == "nvfp4":
            alpha = (w_gs * t_gs).reshape(experts, 1).contiguous()
            self.lora_acc = "separate"
        self.alpha = alpha
        # Eight output tiles over two CTAs: each reuses its LoRA stages across tiles.
        config = GroupedGemmConfig(mma_tiler_mn=(128, width), persistent_clusters=2)
        masked_m = torch.tensor(masked_m, dtype=torch.int32, device=device)
        self.lora_w = (
            torch.randn(self.capacity, experts, n, rank, generator=g) * 0.2
        ).to(device=device, dtype=torch.bfloat16)
        bridge = (torch.randn(experts, m_max, rank, generator=g) * 0.5).to(device)
        if row_gain is not None:
            bridge = bridge * row_gain.to(device)[None, :, None]
        # Rows cycle through the eight slots and the base path.
        if row_slot is None:
            row_slot = torch.arange(m_max) % (self.capacity + 1) - 1
        row_slot = row_slot.to(device)
        common = dict(
            config=config, sf_vec_size=vec, alpha=alpha, alpha_rows_per_half=1 << 30
        )
        if rows_mode == "masked":
            t_data, t_sf, t_deq = _quantize(fmt, tokens, t_gs)
            sf_tokens = pack_scale_atoms(t_sf)
            schedule, tiles, _, _ = build_dual_stage_schedules_masked(
                masked_m,
                m_max=m_max,
                token_width=width,
                n_gemm1=n,
                n_gemm2=n,
                output_width=128,
                cluster_shape_mn=(1, 1),
                use_2cta_instrs=False,
            )
            self.expert_of_row = torch.arange(experts, device=device).repeat_interleave(
                m_max
            )
            self.slot_rows = row_slot.repeat(experts)
            self.row_in_slab = torch.arange(m_max, device=device).repeat(experts)
            self.bridge_rows = bridge.reshape(-1, rank)
            self.rows_deq = t_deq.reshape(-1, k)
            self.live = torch.cat(
                [
                    torch.arange(e * m_max, e * m_max + int(masked_m[e]), device=device)
                    for e in range(experts)
                ]
            )
            # One plane per (slot, expert) slab of m_max rows.
            self.planes_per_slot, self.rows_per_plane = experts, m_max

            def prepare(lora):
                c = torch.zeros(
                    (experts, m_max, n), dtype=torch.bfloat16, device=device
                )
                prepared = prepare_masked_blockscaled(
                    t_data,
                    w_data,
                    sf_tokens,
                    sf_weights,
                    c,
                    masked_m,
                    direct_schedule=schedule,
                    schedule_tiles=tiles,
                    **common,
                    **lora,
                )
                return c.reshape(-1, n), prepared

        else:
            seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
            seg_offsets[1:] = torch.cumsum(((masked_m + 127) // 128) * 128, 0)
            m_ceil = int(seg_offsets[-1])
            flat = torch.zeros((m_ceil, k), device=device)
            self.expert_of_row = torch.zeros(m_ceil, dtype=torch.long, device=device)
            self.slot_rows = torch.full((m_ceil,), -1, dtype=torch.long, device=device)
            self.bridge_rows = torch.zeros((m_ceil, rank), device=device)
            self.row_in_slab = torch.zeros(m_ceil, dtype=torch.long, device=device)
            live = []
            for e in range(experts):
                start, rows = int(seg_offsets[e]), int(masked_m[e])
                flat[start : start + rows] = tokens[e, :rows]
                self.expert_of_row[start : start + rows] = e
                self.slot_rows[start : start + rows] = row_slot[:rows]
                self.bridge_rows[start : start + rows] = bridge[e, :rows]
                self.row_in_slab[start : start + rows] = torch.arange(
                    rows, device=device
                )
                live.append(torch.arange(start, start + rows, device=device))
            self.live = torch.cat(live)
            f_data, f_sf, f_deq = _quantize(fmt, flat.unsqueeze(0), t_gs[:1])
            f_data, self.rows_deq = f_data[0], f_deq[0]
            sf_tokens = pack_scale_atoms(f_sf)
            schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
                masked_m,
                m_pad_ceiling=m_ceil,
                max_expert_rows=m_max,
                m_alignment=128,
                token_width=width,
                n_gemm1=n,
                n_gemm2=n,
                output_width=128,
                cluster_shape_mn=(1, 1),
                use_2cta_instrs=False,
            )
            # One plane per slot over the flat rows.
            self.planes_per_slot, self.rows_per_plane = 1, m_ceil

            def prepare(lora):
                c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
                prepared = prepare_contiguous_blockscaled(
                    f_data,
                    w_data,
                    sf_tokens,
                    sf_weights,
                    c,
                    seg_offsets,
                    direct_schedule=schedule,
                    schedule_tiles=tiles,
                    **common,
                    **lora,
                )
                return c, prepared

        self._prepare = prepare
        self.device = device

    def operands(self, placement: list, bound: int | None = None) -> dict:
        """Operands for ``len(placement)`` resident slots: position ``p``
        carries slot ``placement[p]``'s rows (the bridge, rounded once) and
        weights, or slot ``p``'s weights and no rows for ``None``; ``bound``
        overrides the slot bound the kernel applies."""
        resident = len(placement)
        weights = torch.empty(
            (resident, self.experts, self.n, self.rank),
            dtype=torch.bfloat16,
            device=self.device,
        )
        planes = torch.zeros(
            (resident, self.planes_per_slot * self.rows_per_plane, self.rank),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for p, s in enumerate(placement):
            weights[p] = self.lora_w[p if s is None else s]
            if s is None:
                continue
            mask = self.slot_rows == s
            planes[p, mask] = self.bridge_rows[mask].to(torch.bfloat16)
        return dict(
            lora_w=weights.reshape(-1, self.n, self.rank),
            lora_planes=planes.reshape(-1, self.rows_per_plane, self.rank),
            lora_slots=torch.tensor(
                [resident if bound is None else bound],
                dtype=torch.int32,
                device=self.device,
            ),
            lora_acc=self.lora_acc,
        )

    def run(self, lora: dict | None = None) -> torch.Tensor:
        c, prepared = self._prepare(lora or {})
        prepared.launch()
        _wait_bounded(60.0)
        return c

    def adapted(self, placement: list) -> torch.Tensor:
        placed = torch.zeros_like(self.slot_rows, dtype=torch.bool)
        for s in placement:
            if s is not None:
                placed |= self.slot_rows == s
        return placed

    def base64(self) -> torch.Tensor:
        """FP64 ``A @ W^T`` from the dequantized operands."""
        return torch.einsum(
            "rk,rnk->rn",
            self.rows_deq.double(),
            self.w_deq[self.expert_of_row].double(),
        )

    def delta64(self) -> torch.Tensor:
        """FP64 ``bridge @ B_slot[e]^T`` of each row's own slot."""
        w_rows = self.lora_w[self.slot_rows.clamp(min=0), self.expert_of_row].double()
        return torch.einsum("rq,rnq->rn", self.bridge_rows.double(), w_rows)

    def target(self, placement: list) -> torch.Tensor:
        """FP64 ``A @ W^T`` plus each placed row's ``bridge @ B_slot[e]^T``."""
        placed = self.adapted(placement).double()[:, None]
        return self.base64() + self.delta64() * placed


# Resident slots as the pool presents them (position -> slot of the rows it
# carries, None = a resident adapter without rows): bounds below, at and
# above the kernel's four TMA stages, one adapter alone in slot 4 or 7,
# mixed slots, all eight, unused trailing slots, and residents without rows.
_SLOT_PLACEMENTS = {
    "one": [0],
    "three": [0, 1, 2],
    "four": [0, 1, 2, 3],
    "five": [0, 1, 2, 3, 4],
    "eight": [0, 1, 2, 3, 4, 5, 6, 7],
    "slot4_alone": [None, None, None, None, 4],
    "slot7_alone": [None, None, None, None, None, None, None, 7],
    "mixed": [None, 1, None, None, 4, None, None, 7],
    "trailing": [0, 1, None, None, None, None, None, None],
    "no_rows": [None] * 8,
}


@pytest.mark.parametrize("fmt", sorted(FORMATS))
@pytest.mark.parametrize("rows_mode", ["masked", "contiguous"])
def test_blockscaled_epilogue_lora_slot_bounds(fmt, rows_mode):
    """The epilogue LoRA over every resident-slot bound of an eight-slot pool
    (the kernel streams the slots through four TMA stages, so bounds past
    four reuse a stage within the tile). A row's result depends only on its
    own slot: bitwise the result of the same adapter served alone from slot
    0, base rows and rows of residents without rows stay base rows, and every
    bound is within 1.5e-2 relative L2 of the FP64 target."""
    _skip_unless_sm100()
    problem = _EpilogueDownProblem(fmt, rows_mode)
    live, slot_rows = problem.live, problem.slot_rows
    c_base = problem.run()
    alone = {}
    for s in range(problem.capacity):
        alone[s] = problem.run(problem.operands([s]))
        assert _rel_l2(alone[s].double()[live], problem.target([s])[live]) < 1.5e-2, (
            fmt,
            rows_mode,
            s,
        )
        rows = live[slot_rows[live] == s]
        assert not torch.equal(alone[s][rows], c_base[rows]), (fmt, rows_mode, s)
    for name, placement in _SLOT_PLACEMENTS.items():
        c = problem.run(problem.operands(placement))
        expected = c_base.clone()
        for s in placement:
            if s is not None:
                rows = slot_rows == s
                expected[rows] = alone[s][rows]
        assert torch.equal(c[live], expected[live]), (fmt, rows_mode, name)
        assert _rel_l2(c.double()[live], problem.target(placement)[live]) < 1.5e-2, (
            fmt,
            rows_mode,
            name,
        )


# Static NVFP4 scales of down projections in nvidia/Qwen3.5-122B-A10B-NVFP4
# (snapshot 98915d83): the epilogue's alpha is the fp32 product of an expert's
# weight global scale and its layer's activation scale. Layer 6's pair for
# expert 188 as stored, then weight scales that give the alphas of two other
# near-dead experts, the checkpoint's median and its largest.
_STATIC_T_GS = 0.008603050373494625
_STATIC_W_GS = (
    2.0711090409231365e-37,  # alpha 1.78e-39: a subnormal fp32, 1 / alpha = inf
    1.75e-36,  # alpha 1.5e-38 (layer 4 expert 239): BF16(z / alpha) = inf past |z| = 5
    1.47e-35,  # alpha 1.27e-37 (layer 10 expert 6): the same past |z| = 43
    2.07e-5,  # alpha 1.78e-7, the median
    2.07e-5,  # (an expert without rows)
    4.04e-3,  # alpha 3.47e-5, the largest
)
_STATIC_MASKED_M = (200, 128, 96, 71, 0, 64)


@pytest.mark.parametrize(
    "rows_mode,width",
    [("masked", 64), ("masked", 128), ("contiguous", 64), ("contiguous", 128)],
)
def test_static_epilogue_lora_at_the_checkpoint_scale_extremes(rows_mode, width):
    """The static NVFP4 down GEMM with the down-B fused through the second
    accumulator, ``C * alpha + D``, at the alphas a served checkpoint carries:
    one a subnormal fp32 (its reciprocal is infinite), two whose reciprocal
    is finite but overflows a BF16 bridge, the median and the largest. An
    eight-slot pool with ordinary bridges, zero bridges and bridges 64 times
    larger, base rows inside adapter tiles and whole tiles of base rows
    after them. Every output is finite and within 1.5e-2 relative L2 of the
    FP64 target per expert and bridge class (a gate on each group of rows,
    not an elementwise bound); rows without an adapter, rows with a zero
    bridge, zero weights, zero planes and a bound of zero are bitwise the
    plain build; each row depends only on its own slot; and beside the
    subnormal alpha, where the base is below 1e-20 of the delta, the output
    is within the same distance of the delta alone."""
    _skip_unless_sm100()
    m_max = 256
    row = torch.arange(m_max)
    row_slot = row % (_EpilogueDownProblem.capacity + 1) - 1
    row_slot[128:] = -1
    row_gain = torch.ones(m_max)
    row_gain[row % 16 == 3] = 0.0
    row_gain[row % 16 == 5] = 64.0
    problem = _EpilogueDownProblem(
        "nvfp4",
        rows_mode,
        width=width,
        masked_m=_STATIC_MASKED_M,
        w_gs=_STATIC_W_GS,
        t_gs=_STATIC_T_GS,
        row_slot=row_slot,
        row_gain=row_gain,
    )
    live, slot_rows = problem.live, problem.slot_rows
    expert_of_row = problem.expert_of_row
    gain_rows = row_gain.to(problem.device)[problem.row_in_slab]
    tag = (rows_mode, width)
    # What an inverse-alpha bridge would carry on these operands: no finite
    # reciprocal for the first expert, a BF16 overflow for the next two.
    inverse = 1.0 / problem.alpha.reshape(-1)
    assert not bool(torch.isfinite(inverse[0])) and bool(
        torch.isfinite(inverse[1:]).all()
    )
    compensated = (problem.bridge_rows * inverse[expert_of_row][:, None]).to(
        torch.bfloat16
    )
    overflows = ~torch.isfinite(compensated).all(dim=1) & (slot_rows >= 0)
    assert [
        bool(overflows[live[expert_of_row[live] == e]].any())
        for e in range(problem.experts)
    ] == [True, True, True, False, False, False], tag

    def rel(c, target, rows):
        diff = (c.double()[rows] - target[rows]).norm()
        return float(diff / target[rows].norm().clamp(min=1e-300))

    def check(c, placement, name):
        assert bool(torch.isfinite(c[live]).all()), (*tag, name)
        target = problem.target(placement)
        for e in range(problem.experts):
            for gain in (0.0, 1.0, 64.0):
                rows = live[(expert_of_row[live] == e) & (gain_rows[live] == gain)]
                if rows.numel():
                    assert rel(c, target, rows) < 1.5e-2, (*tag, name, e, gain)

    plain = problem.run()
    check(plain, [], "plain")
    every = list(range(problem.capacity))
    full = problem.run(problem.operands(every))
    check(full, every, "full")
    adapted = slot_rows >= 0
    quiet = live[~adapted[live] | (gain_rows[live] == 0.0)]
    assert torch.equal(full[quiet], plain[quiet]), tag
    for e in range(problem.experts):
        rows = live[adapted[live] & (gain_rows[live] > 0) & (expert_of_row[live] == e)]
        assert rows.numel() == 0 or not torch.equal(full[rows], plain[rows]), (*tag, e)
    # Beside the subnormal alpha the base is below 1e-20 of the delta: the
    # output is within 1.5e-2 relative L2 of the delta alone.
    rows = live[adapted[live] & (gain_rows[live] > 0) & (expert_of_row[live] == 0)]
    base64, delta64 = problem.base64(), problem.delta64()
    assert float(base64[rows].norm()) < 1e-20 * float(delta64[rows].norm())
    assert rel(full, delta64, rows) < 1.5e-2, tag
    # No LoRA MMAs, zero weights or zero bridges: exact zeros beside the base.
    assert torch.equal(problem.run(problem.operands(every, bound=0)), plain), tag
    operands = problem.operands(every)
    zero_w = dict(operands, lora_w=torch.zeros_like(operands["lora_w"]))
    assert torch.equal(problem.run(zero_w), plain), tag
    zero_p = dict(operands, lora_planes=torch.zeros_like(operands["lora_planes"]))
    assert torch.equal(problem.run(zero_p), plain), tag
    # Sparse and mixed pools: a row's result depends only on its own slot.
    for name in ("five", "slot7_alone", "mixed", "no_rows"):
        placement = _SLOT_PLACEMENTS[name]
        c = problem.run(problem.operands(placement))
        expected = plain.clone()
        for s in placement:
            if s is not None:
                expected[slot_rows == s] = full[slot_rows == s]
        assert torch.equal(c[live], expected[live]), (*tag, name)
        check(c, placement, name)


# Bytes K of an fp32 alpha = (K * 2^18 + 1) / 3 * 2^-33: with one product
# C = 3 * 2^-6 in the accumulator, C * alpha = (K * 2^18 + 1) * 2^-39 rounds
# to the fp32 (and BF16) value K * 2^-21.
_STATIC_ROUNDING_K = (128, 149, 170, 191)


def test_static_epilogue_lora_scales_the_base_alone_and_rounds_it_before_d():
    """The static NVFP4 epilogue is ``fl(fl(C * alpha) + D)``: alpha reaches
    the base accumulator only and the product rounds to fp32 before D is
    added. Three rows per alpha, one product C = 3 * 2^-6 or none, one
    weight of -1 on the first output: a zero row with bridge 3 stores -3 (D
    is not scaled); the product row with a zero bridge stores
    BF16(fl(C * alpha)) (alpha once); and with the bridge at that value the
    output is an exact zero, where a fused multiply-add would keep the
    residual 2^-39."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    device = torch.device("cuda")
    n, k, m_max, rank = 256, 512, 64, 16
    delta_row, base_row, residual_row = 0, 1, 2
    alphas = torch.tensor(
        [((byte << 18) + 1) // 3 * 2.0**-33 for byte in _STATIC_ROUNDING_K],
        dtype=torch.float32,
        device=device,
    )
    experts = alphas.numel()
    c_one = 3.0 * 2.0**-6
    rounded = []
    for byte, alpha in zip(_STATIC_ROUNDING_K, alphas.tolist()):
        product = torch.tensor(c_one, dtype=torch.float32) * torch.tensor(
            alpha, dtype=torch.float32
        )
        exact = torch.tensor(c_one * alpha, dtype=torch.float64)
        assert float(product) == float(exact.to(torch.bfloat16)) == byte * 2.0**-21
        assert float(exact) - float(product) == 2.0**-39
        rounded.append(float(product))
    tokens = torch.zeros((experts, m_max, k), device=device)
    tokens[:, [base_row, residual_row], 17] = 6.0
    t_data, t_sf, _ = _quant_nvfp4_ref(tokens, torch.ones(experts, device=device))
    assert int(t_sf[0, base_row, 1]) == 0x38 and int(t_data[0, base_row, 8] >> 4) == 7
    weights = torch.zeros((experts, n, k), device=device)
    weights[:, 0, 17] = 0.5 * 2.0**-6 * alphas
    w_data, w_sf, _ = _quant_nvfp4_ref(weights, alphas)
    assert int(w_sf[0, 0, 1]) == 0x08 and int(w_data[0, 0, 8] >> 4) == 1
    lora_w = torch.zeros((experts, n, rank), dtype=torch.bfloat16, device=device)
    lora_w[:, 0, 0] = -1.0
    planes = torch.zeros((experts, m_max, rank), dtype=torch.bfloat16, device=device)
    planes[:, delta_row, 0] = 3.0
    planes[:, residual_row, 0] = torch.tensor(rounded, device=device).to(torch.bfloat16)
    c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
    masked_m = torch.full((experts,), m_max, dtype=torch.int32, device=device)
    schedule, tiles, _, _ = build_dual_stage_schedules_masked(
        masked_m,
        m_max=m_max,
        token_width=64,
        n_gemm1=n,
        n_gemm2=n,
        output_width=128,
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
    )
    prepare_masked_blockscaled(
        t_data,
        w_data,
        pack_scale_atoms(t_sf),
        pack_scale_atoms(w_sf),
        c,
        masked_m,
        config=GroupedGemmConfig(mma_tiler_mn=(128, 64), persistent_clusters=8),
        sf_vec_size=16,
        alpha=alphas.view(-1, 1).contiguous(),
        alpha_rows_per_half=n,
        direct_schedule=schedule,
        schedule_tiles=tiles,
        lora_w=lora_w,
        lora_planes=planes,
        lora_slots=torch.ones(1, dtype=torch.int32, device=device),
        lora_acc="separate",
    ).launch()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(c).all())
    for e, value in enumerate(rounded):
        assert float(c[e, delta_row, 0]) == -3.0, (e, float(c[e, delta_row, 0]))
        assert float(c[e, base_row, 0]) == value, (e, float(c[e, base_row, 0]), value)
        assert float(c[e, residual_row, 0]) == 0.0, (e, float(c[e, residual_row, 0]))
    # Nothing else moved: the other outputs and rows are exact zeros.
    c[:, [delta_row, base_row, residual_row], 0] = 0.0
    assert not bool(c.any())


def test_per_token_row_quantizer_matches_reference():
    """The per-token row quantizer against the reference of the contract:
    codes, scale bytes and stored row scales bitwise over random rows and
    the special rows (zero, subnormal stored scales, the BF16 maximum, a
    block whose scale rounds to zero, a block with a subnormal E4M3 scale,
    a flat row); every stored value finite; dead rows untouched."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0x9E7)
    k, rows_per_slab = 640, 32
    x = _per_token_rows(k, g).to(device)
    num_rows = x.shape[0]
    pair_to_row = torch.randperm(num_rows, generator=g).to(torch.int32)
    pair_to_row[9] = -1  # a pair without a row
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
        row_denominator=_PER_TOKEN_DENOMINATOR,
    )
    quantize_rows(x, target, pair_to_row)
    torch.cuda.synchronize()
    data_ref, sf_ref, s_ref, deq_ref = _quant_nvfp4_per_token_ref(x)
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

    # The special rows, as the contract states them.
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
    assert bool(torch.isfinite(deq_ref).all())

    # Reconstruction against the FP64 input (the quantizer's own accuracy):
    # the rows that lose a block by construction are judged on the rest.
    x64 = x.double()
    for row in range(num_rows):
        if not live[row] or row == _ROW_ZERO:
            continue
        cols = slice(0, k)
        if row in (_ROW_ZERO_SCALE_BLOCK, _ROW_SUBNORMAL_SCALE_BLOCK):
            cols = slice(0, 16)
        err = _rel_l2(deq_ref[row, cols].double(), x64[row, cols])
        assert err < 0.25, (row, float(err))


def test_per_token_act_delta_matches_row_quantizer():
    """The decode activation's two-pass per-token quantization: codes, scale
    atoms and stored row scales bitwise with the row quantizer applied to the
    BF16 rows it stored; the amax comes from those BF16-rounded values (rows
    whose BF16 maximum exceeds the FP32 maximum exist and use it)."""
    _skip_unless_sm100()
    import torch.nn.functional as F

    from sglang.kernels.ops.lora.moe.activation_delta import act_delta
    from sglang.kernels.ops.lora.moe.cutedsl.api import scale_atom_bytes
    from sglang.kernels.ops.lora.moe.quant_blockscaled import (
        BlockScaledRows,
        quantize_rows,
    )

    device = torch.device("cuda")
    g = torch.Generator().manual_seed(0xAC7)
    experts, m_max, inter, num_tokens, top_k = 4, 64, 384, 40, 2
    rows = experts * m_max
    gateup = (torch.randn((rows, 2 * inter), generator=g) * 2).to(torch.bfloat16)
    delta = (torch.randn((num_tokens, top_k, 2 * inter), generator=g) * 0.5).to(
        torch.bfloat16
    )
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=g)[:top_k] for _ in range(num_tokens)]
    ).to(torch.int32)
    topk_ids[1, 0] = -1
    topk_ids[2] = -1
    pair_to_row = torch.full((num_tokens * top_k,), -1, dtype=torch.int32)
    slots = [0] * experts
    for pair, expert in enumerate(topk_ids.flatten().tolist()):
        if expert >= 0:
            pair_to_row[pair] = expert * m_max + slots[expert]
            slots[expert] += 1
    gateup, delta = gateup.to(device), delta.to(device)
    topk_ids, pair_to_row = topk_ids.to(device), pair_to_row.to(device)

    def rows_buffer():
        return BlockScaledRows(
            data=torch.full((rows, inter // 2), 0x55, dtype=torch.uint8, device=device),
            scales=torch.full(
                (scale_atom_bytes(m_max, inter, 16, experts),),
                0x55,
                dtype=torch.uint8,
                device=device,
            ),
            sf_vec_size=16,
            rows_per_slab=m_max,
            row_scales=torch.full((rows,), -1.0, device=device),
            row_denominator=_PER_TOKEN_DENOMINATOR,
        )

    act_out = torch.zeros((rows, inter), dtype=torch.bfloat16, device=device)
    lora_in = torch.zeros(
        (num_tokens, top_k, inter), dtype=torch.bfloat16, device=device
    )
    target = rows_buffer()
    act_delta(
        gateup,
        delta,
        act_out,
        lora_in,
        pair_to_row,
        topk_ids,
        num_local_experts=experts,
        act_quant=target,
    )
    torch.cuda.synchronize()
    reference = rows_buffer()
    quantize_rows(act_out, reference, pair_to_row)
    torch.cuda.synchronize()
    assert torch.equal(target.data, reference.data)
    assert torch.equal(target.scales, reference.scales)
    assert torch.equal(target.row_scales, reference.row_scales)
    live = pair_to_row[pair_to_row >= 0].long()
    assert bool((target.row_scales[live] >= 0).all())
    assert bool(torch.isfinite(target.row_scales).all())
    dead = torch.ones(rows, dtype=torch.bool, device=device)
    dead[live] = False
    assert bool((target.row_scales[dead] == -1).all())

    # The stored scale follows the BF16 maximum, which exceeds the FP32
    # maximum on the rows whose largest activation rounded up.
    pairs = torch.nonzero(pair_to_row >= 0, as_tuple=True)[0]
    g32 = gateup.float()[live] + delta.view(-1, 2 * inter).float()[pairs]
    act32 = F.silu(g32[:, :inter]) * g32[:, inter:]
    bf16_amax = act_out.float()[live].abs().amax(dim=1)
    rounded_up = bf16_amax > act32.abs().amax(dim=1)
    assert bool(rounded_up.any())
    expected = bf16_amax / torch.tensor(_PER_TOKEN_DENOMINATOR, device=device)
    assert torch.equal(target.row_scales[live][rounded_up], expected[rounded_up])
    assert torch.equal(target.row_scales[live], expected)


def _per_token_gemm_operands(generator, device, experts, n, k, m_max):
    """Static-quantized weights under extreme per-expert global scales and
    per-token-quantized token slabs holding the special rows first."""
    w_gs = torch.tensor([1e-6, 1e-2, 1.0, 2.0**32], device=device)[:experts]
    weights = (torch.randn(experts, n, k, generator=generator) * 0.05).to(device)
    weights = weights * (w_gs * 10)[:, None, None]
    w_data, w_sf, w_deq = _quant_nvfp4_ref(weights, w_gs, dequant_dtype=torch.float64)
    tokens = torch.stack(
        [_per_token_rows(k, generator)[:m_max] for _ in range(experts)]
    ).to(device)
    # A BF16-maximum row's true output overflows BF16 itself unless the
    # weights are tiny, so only the 1e-6 expert keeps it; the others take a
    # random row there instead.
    for e in range(1, experts):
        tokens[e, _ROW_BF16_MAX] = tokens[e, _ROW_FLAT + 1]
    t_data, t_sf, s, t_deq = _quant_nvfp4_per_token_ref(tokens)
    # The second half of the weight rows doubles its alpha (the gate/up
    # selection); alpha carries the weight global scale only.
    alpha = torch.stack([w_gs, 2 * w_gs], dim=1).contiguous()
    w_deq = w_deq.clone()
    w_deq[:, n // 2 :] *= 2
    return w_data, w_sf, w_deq, t_data, t_sf, s, t_deq, alpha


def _check_per_token_rows(c, ref, s, tag):
    """Live rows against the float64 reference (exact dequantized operands,
    float64 products and sums): zero rows exactly zero; rows with a
    normal-range reference within tolerance; every value finite. Norms run in
    float64: the BF16-maximum rows' outputs square past fp32."""
    assert bool(torch.isfinite(c).all()), tag
    assert ref.dtype is torch.float64, tag
    c64, ref64 = c.double(), ref
    assert _rel_l2(c64, ref64) < 1.5e-2, (tag, float(_rel_l2(c64, ref64)))
    zero = s == 0
    assert bool((c[zero] == 0).all()), tag
    for row in range(c.shape[0]):
        if zero[row]:
            continue
        magnitude = ref64[row].abs().max()
        if magnitude > 2.0**-100:
            err = _rel_l2(c64[row], ref64[row])
            assert err < 2.5e-2, (tag, row, float(err), float(s[row]))
        else:
            assert bool((c64[row].abs() <= 2.0**-100).all()), (tag, row)


@pytest.mark.parametrize(
    "rows_mode,width", [("masked", 64), ("masked", 128), ("contiguous", 128)]
)
def test_blockscaled_grouped_gemm_per_token_matches_dequantized_reference(
    rows_mode, width
):
    """The NVFP4 GEMM with per-token scales applied in the epilogue, per
    output column and after alpha, against the float64 reference: the
    special rows (zero, subnormal stored scales, the BF16 maximum, blocks
    with zero or subnormal E4M3 scales) under alpha from 1e-6 to 2^32. A
    subnormal stored scale must survive the epilogue (no flush to zero): the
    2^32 expert lifts those rows' outputs into the BF16 normal range."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_contiguous_blockscaled,
        prepare_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_contiguous,
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(5)
    experts, n, k, m_max = 4, 256, 512, 96
    w_data, w_sf, w_deq, t_data, t_sf, s, t_deq, alpha = _per_token_gemm_operands(
        g, device, experts, n, k, m_max
    )
    sf_weights = pack_scale_atoms(w_sf)
    config = GroupedGemmConfig(mma_tiler_mn=(128, width), persistent_clusters=8)
    masked_m = torch.tensor([96, 0, 70, 9], dtype=torch.int32, device=device)
    geometry = dict(
        n_gemm1=n,
        n_gemm2=n,
        output_width=128,
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
    )
    if rows_mode == "masked":
        m_rows = -(-m_max // width) * width
        pad = m_rows - m_max
        t_data_p = torch.nn.functional.pad(t_data, (0, 0, 0, pad))
        t_sf_p = torch.nn.functional.pad(t_sf, (0, 0, 0, pad))
        s_p = torch.nn.functional.pad(s, (0, pad)).contiguous()
        c = torch.zeros((experts, m_rows, n), dtype=torch.bfloat16, device=device)
        schedule, tiles, _, _ = build_dual_stage_schedules_masked(
            masked_m, m_max=m_rows, token_width=width, **geometry
        )
        prepare_masked_blockscaled(
            t_data_p,
            w_data,
            pack_scale_atoms(t_sf_p),
            sf_weights,
            c,
            masked_m,
            config=config,
            sf_vec_size=16,
            alpha=alpha,
            alpha_rows_per_half=n // 2,
            tok_scale=s_p,
            direct_schedule=schedule,
            schedule_tiles=tiles,
        ).launch()
        torch.cuda.synchronize()
        for e in range(experts):
            rows = int(masked_m[e])
            if rows == 0:
                continue
            ref = t_deq[e, :rows] @ w_deq[e].T
            _check_per_token_rows(c[e, :rows], ref, s[e, :rows], ("masked", e))
    else:
        seg_counts = masked_m
        seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
        seg_offsets[1:] = torch.cumsum(((seg_counts + 127) // 128) * 128, 0)
        m_ceil = int(seg_offsets[-1])
        f_data = torch.zeros((m_ceil, k // 2), dtype=torch.uint8, device=device)
        f_sf = torch.zeros((m_ceil, k // 16), dtype=torch.uint8, device=device)
        f_s = torch.zeros((m_ceil,), device=device)
        f_deq = torch.zeros((m_ceil, k), device=device, dtype=torch.float64)
        for e in range(experts):
            start, rows = int(seg_offsets[e]), int(seg_counts[e])
            f_data[start : start + rows] = t_data[e, :rows]
            f_sf[start : start + rows] = t_sf[e, :rows]
            f_s[start : start + rows] = s[e, :rows]
            f_deq[start : start + rows] = t_deq[e, :rows]
        c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
        schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
            seg_counts,
            m_pad_ceiling=m_ceil,
            max_expert_rows=m_max,
            m_alignment=128,
            token_width=width,
            **geometry,
        )
        prepare_contiguous_blockscaled(
            f_data,
            w_data,
            pack_scale_atoms(f_sf.unsqueeze(0)),
            sf_weights,
            c,
            seg_offsets,
            config=config,
            sf_vec_size=16,
            alpha=alpha,
            alpha_rows_per_half=n // 2,
            tok_scale=f_s.view(1, -1).contiguous(),
            direct_schedule=schedule,
            schedule_tiles=tiles,
        ).launch()
        torch.cuda.synchronize()
        for e in range(experts):
            start, rows = int(seg_offsets[e]), int(seg_counts[e])
            if rows == 0:
                continue
            ref = f_deq[start : start + rows] @ w_deq[e].T
            _check_per_token_rows(
                c[start : start + rows], ref, f_s[start : start + rows], ("contig", e)
            )


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """One BF16 ulp at each magnitude (the subnormal grid below 2^-126)."""
    exponent = torch.floor(torch.log2(x.abs().clamp(min=2.0**-126)))
    return torch.pow(2.0, exponent - 7)


def test_per_token_epilogue_contract_at_the_smallest_alpha():
    """The per-token epilogue's accuracy contract (design 4.1) at the lower
    edge of its alpha domain, alpha = (1 + 2^-7) * 2^-126: on rows with an
    amax below 2^19 every element the output can hold is within half a BF16
    ulp of FP64 plus the accumulation term; a row at amax 2^30 whose
    accumulator for one output is a single minimum product (an activation
    code of 0.5 at a subnormal E4M3 scale times a weight code of 0.5 at the
    2^-6 floor, C = 2^-17) keeps that output nonzero, 2^-7 below FP64 and
    bitwise equal to the fp32 model (C * alpha) * s of the contract; the same
    accumulator on a row with s = 512 lands on the BF16 subnormal boundary and
    comes out zero where FP64 rounds to 2^-133: the documented tail, pinned.
    A third row (C = 3 * 2^-17, s = 192, amax 516,096 below 2^19) shows the
    first rounding exceeding 2^-9 relative (193.5 subnormal units to 194)
    while the output still equals FP64's rounding, 2^-133: the bound is
    absolute, not relative, below the BF16 normal range."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(8)
    n, k, m_max = 256, 512, 64
    tail_row, edge_row, bits_row = 1, 2, 3
    alpha = (1.0 + 2.0**-7) * 2.0**-126
    rows = torch.randn((m_max, k), generator=g)
    rows *= torch.pow(
        2.0, torch.rand((m_max, 1), generator=g) * 28 - 10
    )  # amax 2^-10..2^18
    rows[0].zero_()
    rows[tail_row].zero_()
    rows[tail_row, 0] = 2.0**30  # the row amax: s = fl(2^30 / 2688)
    rows[tail_row, 16] = 4700.0  # block 1's amax: its E4M3 scale rounds to 2^-9
    rows[tail_row, 17] = 388.0  # a code of 0.5 under that scale
    rows[edge_row].zero_()
    rows[edge_row, 0] = 512.0 * 2688.0  # s = 512 exactly
    rows[edge_row, 16] = 6.0  # block 1's E4M3 scale 2^-9
    rows[edge_row, 17] = 0.5  # a code of 0.5 under it
    rows[bits_row].zero_()
    rows[bits_row, 0] = 192.0 * 2688.0  # s = 192 exactly, amax 516,096 < 2^19
    rows[bits_row, 16] = 2.25  # block 1's E4M3 scale 2^-9
    rows[bits_row, 17] = 0.5625  # a code of 1.5 under it: C = 3 * 2^-17
    tokens = rows.to(torch.bfloat16).unsqueeze(0).to(device)
    t_data, t_sf, s, t_deq = _quant_nvfp4_per_token_ref(tokens)
    for row in (tail_row, edge_row):
        assert int(t_sf[0, row, 1]) == 0x01 and int(t_data[0, row, 8] >> 4) == 1
    assert int(t_sf[0, bits_row, 1]) == 0x01 and int(t_data[0, bits_row, 8] >> 4) == 3
    assert float(s[0, edge_row]) == 512.0 and float(s[0, bits_row]) == 192.0
    alphas = torch.tensor([alpha], dtype=torch.float32, device=device)
    # The weight codes come from a unit global scale (the reference quantizer's
    # reciprocal would overflow at alpha); alpha enters the kernel and the
    # oracle as the global scale of those codes.
    weights = torch.randn((1, n, k), generator=g) * 0.5
    weights[0, 0, :32] = 0.0
    weights[0, 0, 17] = 0.5 * 2.0**-6  # code 0.5 at the 2^-6 floor
    w_data, w_sf, w_deq = _quant_nvfp4_ref(
        weights.to(device), torch.ones(1, device=device), dequant_dtype=torch.float64
    )
    w_deq = w_deq * alpha
    assert int(w_sf[0, 0, 1]) == 0x08 and int(w_data[0, 0, 8] >> 4) == 1  # 2^-6, 0.5
    c = torch.zeros((1, m_max, n), dtype=torch.bfloat16, device=device)
    masked_m = torch.tensor([m_max], dtype=torch.int32, device=device)
    schedule, tiles, _, _ = build_dual_stage_schedules_masked(
        masked_m,
        m_max=m_max,
        token_width=64,
        n_gemm1=n,
        n_gemm2=n,
        output_width=128,
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
    )
    prepare_masked_blockscaled(
        t_data,
        w_data,
        pack_scale_atoms(t_sf),
        pack_scale_atoms(w_sf),
        c,
        masked_m,
        config=GroupedGemmConfig(mma_tiler_mn=(128, 64), persistent_clusters=8),
        sf_vec_size=16,
        alpha=alphas.view(-1, 1).contiguous(),
        alpha_rows_per_half=n,
        tok_scale=s.contiguous(),
        direct_schedule=schedule,
        schedule_tiles=tiles,
    ).launch()
    torch.cuda.synchronize()
    out = c[0].double()
    ref = t_deq[0] @ w_deq[0].T
    summands = t_deq[0].abs() @ w_deq[0].abs().T
    assert bool(torch.isfinite(c).all())
    # Rows below amax 2^19: half a BF16 ulp plus the fp32 accumulation over the sum of products.
    ordinary = torch.ones(m_max, dtype=torch.bool, device=device)
    ordinary[tail_row] = False
    ordinary[edge_row] = False
    ordinary[bits_row] = False
    budget = 0.5 * _bf16_ulp(ref) + 2.0**-14 * summands
    err = (out - ref).abs()
    assert bool((err[ordinary] <= budget[ordinary]).all()), float(
        (err[ordinary] / budget[ordinary].clamp(min=1e-300)).max()
    )
    # The tail row's accumulator for output 0 is exactly C = 2^-17 (one product).
    c_tail = 2.0**-17
    expected = c_tail * alpha * float(s[0, tail_row])
    assert abs(float(ref[tail_row, 0]) - expected) < 1e-6 * expected
    t32 = torch.tensor(c_tail, dtype=torch.float32, device=device) * torch.tensor(
        alpha, dtype=torch.float32, device=device
    )
    model = (t32 * s[0, tail_row]).to(torch.bfloat16)
    assert float(model) != 0.0 and torch.equal(c[0, tail_row, 0], model), (
        float(c[0, tail_row, 0]),
        float(model),
    )
    rel = float(err[tail_row, 0] / ref[tail_row, 0].abs())
    assert 2.0**-8 < rel <= 2.0**-4, rel  # the documented loss, 2^-7 at this C
    # The edge row: the fp32 value 2^-134 is half of BF16's smallest subnormal and
    # rounds to even (zero), while the exact product, 2^-7 above it, rounds to 2^-133.
    model_edge = (t32 * s[0, edge_row]).to(torch.bfloat16)
    assert float(model_edge) == 0.0 and float(c[0, edge_row, 0]) == 0.0, float(
        c[0, edge_row, 0]
    )
    assert float(ref[edge_row, 0].to(torch.bfloat16)) == 2.0**-133
    # The bits row: the first rounding takes 193.5 subnormal units to 194 (a
    # relative error of 1/387, above 2^-9), yet both roundings land on 2^-133.
    c_bits = 3.0 * 2.0**-17
    t32_bits = torch.tensor(c_bits, dtype=torch.float32, device=device) * torch.tensor(
        alpha, dtype=torch.float32, device=device
    )
    assert float(t32_bits) == 194.0 * 2.0**-149
    first_rounding = abs(float(t32_bits) - c_bits * alpha) / (c_bits * alpha)
    assert 2.0**-9 < first_rounding < 2.0**-8, first_rounding
    model_bits = (t32_bits * s[0, bits_row]).to(torch.bfloat16)
    assert float(model_bits) == 2.0**-133 and torch.equal(c[0, bits_row, 0], model_bits)
    assert float(ref[bits_row, 0].to(torch.bfloat16)) == 2.0**-133


def test_per_token_subnormal_block_scales_through_the_mma():
    """Whether the tcgen05 block-scaled MMA takes subnormal UE4M3 scales
    exactly: token rows whose blocks all carry the scale byte 0x01 (2^-9)
    except the row-maximum block, whose weight columns are zero, so the
    output comes from the subnormal-scaled blocks alone."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(6)
    experts, n, k, m_max = 1, 256, 512, 128
    weights = (torch.randn(experts, n, k, generator=g) * 0.05).to(device)
    weights[:, :, :16] = 0
    w_gs = torch.ones(experts, device=device)
    w_data, w_sf, w_deq = _quant_nvfp4_ref(weights, w_gs, dequant_dtype=torch.float64)
    tokens = (torch.rand((experts, m_max, k), generator=g) * 2 - 1).to(device)
    tokens = tokens * 2.0**-19
    tokens[..., 0] = 1.0
    tokens[..., 17::16] = 2.0**-18  # every other block's amax
    tokens = tokens.to(torch.bfloat16)
    t_data, t_sf, s, t_deq = _quant_nvfp4_per_token_ref(tokens)
    assert bool((t_sf[..., 1:] == 0x01).all())
    sf_tokens, sf_weights = pack_scale_atoms(t_sf), pack_scale_atoms(w_sf)
    c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
    masked_m = torch.tensor([m_max], dtype=torch.int32, device=device)
    schedule, tiles, _, _ = build_dual_stage_schedules_masked(
        masked_m,
        m_max=m_max,
        token_width=128,
        n_gemm1=n,
        n_gemm2=n,
        output_width=128,
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
    )
    prepare_masked_blockscaled(
        t_data,
        w_data,
        sf_tokens,
        sf_weights,
        c,
        masked_m,
        config=GroupedGemmConfig(mma_tiler_mn=(128, 128), persistent_clusters=8),
        sf_vec_size=16,
        alpha=w_gs.view(-1, 1).contiguous(),
        alpha_rows_per_half=n,
        tok_scale=s.contiguous(),
        direct_schedule=schedule,
        schedule_tiles=tiles,
    ).launch()
    torch.cuda.synchronize()
    ref = t_deq[0] @ w_deq[0].T
    assert ref.abs().max() > 2.0**-100
    err = _rel_l2(c[0].double(), ref)
    assert err < 2e-2, (
        "the tcgen05 block-scaled MMA does not take subnormal UE4M3 scales "
        f"exactly: rel L2 {float(err):.3g} against the dequantized reference"
    )


# --- The per-token epilogue LoRA through the separate accumulator --------------
# One expert per w2 global scale: the checkpoints' extremes (Qwen3.5-122B's
# dead experts, Qwen3.5-397B's, Beam's), the domain edges of the per-token
# epilogue at this K and the 11-05 study's values.
_FUSED_K = 512
_FUSED_ALPHAS = (
    ("q122_w2_min", 2.0711090409231365e-37),
    ("q397_w2_min", 1.55e-31),
    ("edge_low", 2.0**-126),
    ("study_1e-6", 1e-6),
    ("beam_w2_min", 7.96e-5),
    ("beam_w2_max", 1.23e-3),
    ("one", 1.0),
    ("study_1e2", 1e2),
    ("edge_high", torch.finfo(torch.float32).max / (_FUSED_K * (6.0 * 448.0) ** 2)),
)
_FUSED_MASKED_M = (96, 0, 70, 9, 96, 33, 64, 17, 96)
# The BF16-maximum row's output overflows BF16 unless alpha is tiny.
_FUSED_BF16_MAX_ALPHA = 1e-6
_FUSED_CAPACITY, _FUSED_RANK = 8, 16
_FUSED_SLOT_TINY = 6  # the rows at amax 2^-133 and 2^-126: a normal delta here
_FUSED_SLOT_TOP = 7  # bridges at the top of the BF16 range; its B is 2^-120 small
_FUSED_TOP_BRIDGE = 2.0**126


def _fused_row_slots(m_max: int) -> torch.Tensor:
    """The slot of each row: the eight slots and the base path in turn, with
    the special rows pinned: the zero row shows its delta alone (slot 0), the
    two tiniest rows take the normal-delta slot, the next two tiny rows and
    the BF16-maximum row stay base rows, the block cases take low slots."""
    slots = torch.arange(m_max) % (_FUSED_CAPACITY + 1) - 1
    slots[_ROW_ZERO] = 0
    slots[_ROWS_TINY[0]] = slots[_ROWS_TINY[1]] = _FUSED_SLOT_TINY
    slots[_ROWS_TINY[2]] = slots[_ROWS_TINY[3]] = slots[_ROW_BF16_MAX] = -1
    slots[_ROW_ZERO_SCALE_BLOCK], slots[_ROW_SUBNORMAL_SCALE_BLOCK] = 1, 2
    slots[_ROW_FLAT] = 3
    return slots


class _FusedDownProblem:
    """The per-token down GEMM over the alpha experts (one empty, partial
    tiles) on two persistent CTAs with the operands of an eight-slot pool:
    raw BF16 bridges on one plane per slot (per (slot, expert) slab for
    masked rows) and weights per (slot, expert)."""

    def __init__(self, rows_mode: str, width: int) -> None:
        from sglang.kernels.ops.lora.moe.cutedsl.api import (
            GroupedGemmConfig,
            prepare_contiguous_blockscaled,
            prepare_masked_blockscaled,
        )
        from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
            build_dual_stage_schedules_contiguous,
            build_dual_stage_schedules_masked,
        )
        from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

        device = torch.device("cuda")
        g = torch.Generator(device="cpu").manual_seed(23)
        experts, n, k, m_max, rank = len(_FUSED_ALPHAS), 256, _FUSED_K, 96, _FUSED_RANK
        self.experts, self.n, self.rank, self.device = experts, n, rank, device
        alphas = torch.tensor([a for _, a in _FUSED_ALPHAS], dtype=torch.float32)
        # Weights whose codes sit at a sane magnitude under each alpha; the
        # second half of the rows doubles its alpha (the gate/up selection).
        weights = (torch.randn(experts, n, k, generator=g) * 0.5).to(device)
        weights = weights * alphas.to(device)[:, None, None]
        w_data, w_sf, self.w_deq = _quant_nvfp4_ref(
            weights, alphas.to(device), dequant_dtype=torch.float64
        )
        self.w_deq = self.w_deq.clone()
        self.w_deq[:, n // 2 :] *= 2
        self.alpha = torch.stack([alphas, 2 * alphas], dim=1).contiguous().to(device)
        tokens = torch.stack([_per_token_rows(k, g)[:m_max] for _ in range(experts)])
        for e, (_, a) in enumerate(_FUSED_ALPHAS):
            if a > _FUSED_BF16_MAX_ALPHA:
                tokens[e, _ROW_BF16_MAX] = tokens[e, _ROW_FLAT + 1]
        tokens = tokens.to(device)
        t_data, t_sf, s, t_deq = _quant_nvfp4_per_token_ref(tokens)
        self.lora_w = (
            torch.randn(_FUSED_CAPACITY, experts, n, rank, generator=g) * 0.2
        ).to(device=device, dtype=torch.bfloat16)
        self.lora_w[_FUSED_SLOT_TOP] = (
            self.lora_w[_FUSED_SLOT_TOP].float() * 2.0**-120
        ).to(torch.bfloat16)
        bridge = torch.randn(experts, m_max, rank, generator=g) * 0.5
        row_slot = _fused_row_slots(m_max)
        top = row_slot == _FUSED_SLOT_TOP
        bridge[:, top] = (
            torch.sign(bridge[:, top])
            * _FUSED_TOP_BRIDGE
            * (1 + torch.rand(bridge[:, top].shape, generator=g))
        )
        bridge = bridge.to(torch.bfloat16)
        masked_m = torch.tensor(_FUSED_MASKED_M, dtype=torch.int32, device=device)
        config = GroupedGemmConfig(mma_tiler_mn=(128, width), persistent_clusters=2)
        geometry = dict(
            n_gemm1=n,
            n_gemm2=n,
            output_width=128,
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
        )
        sf_weights = pack_scale_atoms(w_sf)
        common = dict(
            config=config,
            sf_vec_size=16,
            alpha=self.alpha,
            alpha_rows_per_half=n // 2,
        )
        if rows_mode == "masked":
            m_rows = -(-m_max // width) * width
            pad = m_rows - m_max
            t_data_p = torch.nn.functional.pad(t_data, (0, 0, 0, pad))
            sf_tokens = pack_scale_atoms(torch.nn.functional.pad(t_sf, (0, 0, 0, pad)))
            s_p = torch.nn.functional.pad(s, (0, pad)).contiguous()
            schedule, tiles, _, _ = build_dual_stage_schedules_masked(
                masked_m, m_max=m_rows, token_width=width, **geometry
            )
            self.expert_of_row = torch.arange(experts, device=device).repeat_interleave(
                m_rows
            )
            self.slot_rows = (
                torch.nn.functional.pad(row_slot, (0, pad), value=-1).repeat(experts)
            ).to(device)
            self.bridge_rows = (
                torch.nn.functional.pad(bridge, (0, 0, 0, pad))
                .reshape(-1, rank)
                .to(device)
            )
            self.rows_deq = torch.nn.functional.pad(t_deq, (0, 0, 0, pad)).reshape(
                -1, k
            )
            self.s_rows = s_p.reshape(-1)
            self.row_in_slab = torch.arange(m_rows, device=device).repeat(experts)
            self.live = torch.cat(
                [
                    torch.arange(
                        e * m_rows, e * m_rows + int(masked_m[e]), device=device
                    )
                    for e in range(experts)
                ]
            )
            self.planes_per_slot, self.rows_per_plane = experts, m_rows

            def prepare(lora):
                c = torch.zeros(
                    (experts, m_rows, n), dtype=torch.bfloat16, device=device
                )
                prepared = prepare_masked_blockscaled(
                    t_data_p,
                    w_data,
                    sf_tokens,
                    sf_weights,
                    c,
                    masked_m,
                    tok_scale=s_p,
                    direct_schedule=schedule,
                    schedule_tiles=tiles,
                    **common,
                    **lora,
                )
                return c.reshape(-1, n), prepared

        else:
            seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
            seg_offsets[1:] = torch.cumsum(((masked_m + 127) // 128) * 128, 0)
            m_ceil = int(seg_offsets[-1])
            f_data = torch.zeros((m_ceil, k // 2), dtype=torch.uint8, device=device)
            f_sf = torch.zeros((m_ceil, k // 16), dtype=torch.uint8, device=device)
            f_s = torch.zeros((m_ceil,), device=device)
            self.rows_deq = torch.zeros((m_ceil, k), device=device, dtype=torch.float64)
            self.expert_of_row = torch.zeros(m_ceil, dtype=torch.long, device=device)
            self.slot_rows = torch.full((m_ceil,), -1, dtype=torch.long, device=device)
            self.bridge_rows = torch.zeros(
                (m_ceil, rank), dtype=torch.bfloat16, device=device
            )
            self.row_in_slab = torch.zeros(m_ceil, dtype=torch.long, device=device)
            live = []
            for e in range(experts):
                start, rows = int(seg_offsets[e]), int(masked_m[e])
                f_data[start : start + rows] = t_data[e, :rows]
                f_sf[start : start + rows] = t_sf[e, :rows]
                f_s[start : start + rows] = s[e, :rows]
                self.rows_deq[start : start + rows] = t_deq[e, :rows]
                self.expert_of_row[start : start + rows] = e
                self.slot_rows[start : start + rows] = row_slot[:rows].to(device)
                self.bridge_rows[start : start + rows] = bridge[e, :rows].to(device)
                self.row_in_slab[start : start + rows] = torch.arange(
                    rows, device=device
                )
                live.append(torch.arange(start, start + rows, device=device))
            self.live = torch.cat(live)
            self.s_rows = f_s
            sf_tokens = pack_scale_atoms(f_sf.unsqueeze(0))
            tok_scale = f_s.view(1, -1).contiguous()
            schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
                masked_m,
                m_pad_ceiling=m_ceil,
                max_expert_rows=m_max,
                m_alignment=128,
                token_width=width,
                **geometry,
            )
            self.planes_per_slot, self.rows_per_plane = 1, m_ceil

            def prepare(lora):
                c = torch.zeros((m_ceil, n), dtype=torch.bfloat16, device=device)
                prepared = prepare_contiguous_blockscaled(
                    f_data,
                    w_data,
                    sf_tokens,
                    sf_weights,
                    c,
                    seg_offsets,
                    tok_scale=tok_scale,
                    direct_schedule=schedule,
                    schedule_tiles=tiles,
                    **common,
                    **lora,
                )
                return c, prepared

        self._prepare = prepare
        # FP64 per row: the base from the dequantized operands (the rows carry
        # s, the weights alpha) and the delta of the row's own slot.
        self.base64 = torch.einsum(
            "rk,rnk->rn", self.rows_deq, self.w_deq[self.expert_of_row]
        )
        w_rows = self.lora_w.double()[self.slot_rows.clamp(min=0), self.expert_of_row]
        self.delta64 = torch.einsum("rq,rnq->rn", self.bridge_rows.double(), w_rows)
        self.summands = torch.einsum(
            "rk,rnk->rn", self.rows_deq.abs(), self.w_deq[self.expert_of_row].abs()
        ) + torch.einsum("rq,rnq->rn", self.bridge_rows.double().abs(), w_rows.abs())

    def operands(self, placement: list, bound: int | None = None) -> dict:
        """Operands for the resident slots ``placement`` (position -> slot of
        the rows it carries, or None for a resident without rows), the raw
        bridge of each row on its position's plane; ``bound`` overrides the
        slot bound the kernel applies."""
        resident = len(placement)
        weights = torch.empty(
            (resident, self.experts, self.n, self.rank),
            dtype=torch.bfloat16,
            device=self.device,
        )
        planes = torch.zeros(
            (resident, self.planes_per_slot * self.rows_per_plane, self.rank),
            dtype=torch.bfloat16,
            device=self.device,
        )
        for p, s in enumerate(placement):
            weights[p] = self.lora_w[p if s is None else s]
            if s is not None:
                mask = self.slot_rows == s
                planes[p, mask] = self.bridge_rows[mask]
        return dict(
            lora_w=weights.reshape(-1, self.n, self.rank),
            lora_planes=planes.reshape(-1, self.rows_per_plane, self.rank),
            lora_slots=torch.tensor(
                [resident if bound is None else bound],
                dtype=torch.int32,
                device=self.device,
            ),
            lora_acc="separate",
        )

    def run(self, lora: dict | None = None) -> torch.Tensor:
        c, prepared = self._prepare(lora or {})
        prepared.launch()
        _wait_bounded(60.0)
        return c

    def adapted(self, placement: list) -> torch.Tensor:
        placed = torch.zeros_like(self.slot_rows, dtype=torch.bool)
        for s in placement:
            if s is not None:
                placed |= self.slot_rows == s
        return placed

    def target(self, placement: list) -> torch.Tensor:
        return self.base64 + self.delta64 * self.adapted(placement).double()[:, None]


def _fused_check_rows(c, target, summands, s, adapted, tag) -> int:
    """Live rows against FP64: every value finite; base rows of zero tokens
    exactly zero; rows whose output is a normal BF16 within 2.5e-2 relative
    L2, rows below that (BF16 subnormal outputs, or values the output grid
    rounds to zero) within one grid step plus 2^-6 relative per element; and
    the per-element budget of design 6.2 for the kernel (half a BF16 ulp of
    the output, fp32 accumulation over the sum of absolute products, the
    epilogue's three roundings, the subnormal grid). Returns the number of
    elements past the budget."""
    assert bool(torch.isfinite(c).all()), tag
    c64 = c.double()
    assert float(_rel_l2(c64, target)) < 1.5e-2, (tag, float(_rel_l2(c64, target)))
    zero_base = (s == 0) & ~adapted
    assert bool((c64[zero_base] == 0).all()), tag
    magnitude = target.abs().amax(dim=1)
    err = (c64 - target).abs()
    row_err = err.norm(dim=1) / target.norm(dim=1).clamp(min=1e-300)
    normal = magnitude >= 2.0**-120
    assert bool((row_err[normal] < 2.5e-2).all()), (
        tag,
        float(row_err[normal].max()),
        int(torch.nonzero(normal & (row_err >= 2.5e-2))[0])
        if bool((normal & (row_err >= 2.5e-2)).any())
        else None,
    )
    small = ~normal
    assert bool((err[small] <= 2.0**-133 + 2.0**-6 * target[small].abs()).all()), tag
    budget = (
        2.0**-8 * target.abs()
        + (_FUSED_K + _FUSED_RANK) * 2.0**-24 * summands
        + 3 * 2.0**-24 * summands
        + 2.0**-134
    )
    return int((err > budget).sum())


@pytest.mark.parametrize(
    "rows_mode,width",
    [("masked", 64), ("masked", 128), ("contiguous", 64), ("contiguous", 128)],
)
def test_fused_per_token_epilogue_lora_matches_float64(rows_mode, width):
    """The per-token down GEMM with the down-B fused through the separate
    accumulator, ``(C * alpha) * s + D``, over the 6.4 case set: per expert
    an alpha at the checkpoints' extremes (2.07e-37 .. 1.23e-3), the domain
    edges (2^-126 and the K-dependent bound) and the study's values; rows
    with amax log-uniform over 1e-3..1e3, a zero row, tiny rows (amax 2^-133
    .. 2^-100), the BF16 maximum, blocks with zero or subnormal scales, a
    saturated row; an eight-slot pool with bridges at the top of the BF16
    range on one slot and a normal delta on the tiniest rows. Against the
    FP64 target built from the dequantized operands: every bound 1-8 and
    every resident-slot placement (a bound past the four TMA stages, one slot
    alone high in the pool, residents without rows); a bound of zero and the
    rows without an adapter are bitwise the plain per-token build; a row's
    result depends only on its own slot (bitwise the same adapter served
    alone from slot 0); the zero row shows exactly its delta and the tiniest
    rows show theirs."""
    _skip_unless_sm100()
    problem = _FusedDownProblem(rows_mode, width)
    live, slot_rows, s = problem.live, problem.slot_rows, problem.s_rows
    plain = problem.run()
    none = problem.adapted([])
    _fused_check_rows(
        plain[live],
        problem.base64[live],
        problem.summands[live],
        s[live],
        none[live],
        (rows_mode, width, "plain"),
    )
    # A bound of zero skips D: bitwise the plain build.
    assert torch.equal(problem.run(problem.operands([0], bound=0)), plain), (
        rows_mode,
        width,
    )
    alone = {}
    flagged = 0
    for slot in range(_FUSED_CAPACITY):
        alone[slot] = problem.run(problem.operands([slot]))
        adapted = problem.adapted([slot])
        flagged += _fused_check_rows(
            alone[slot][live],
            problem.target([slot])[live],
            problem.summands[live],
            s[live],
            adapted[live],
            (rows_mode, width, "alone", slot),
        )
        rows = live[adapted[live]]
        assert not torch.equal(alone[slot][rows], plain[rows]), (rows_mode, width, slot)
        base_rows = live[~adapted[live]]
        assert torch.equal(alone[slot][base_rows], plain[base_rows]), (
            rows_mode,
            width,
            slot,
        )
    for name, placement in _SLOT_PLACEMENTS.items():
        c = problem.run(problem.operands(placement))
        expected = plain.clone()
        for slot in placement:
            if slot is not None:
                rows = slot_rows == slot
                expected[rows] = alone[slot][rows]
        assert torch.equal(c[live], expected[live]), (rows_mode, width, name)
        adapted = problem.adapted(placement)
        flagged += _fused_check_rows(
            c[live],
            problem.target(placement)[live],
            problem.summands[live],
            s[live],
            adapted[live],
            (rows_mode, width, name),
        )
    assert flagged == 0, (rows_mode, width, flagged)
    # Every slot: the zero rows show exactly their delta (one BF16 rounding),
    # the tiniest rows show theirs (the base is invisible beside it), the
    # tiny base rows and the BF16-maximum rows stay the plain rows.
    c = problem.run(problem.operands(list(range(_FUSED_CAPACITY))))
    zero = live[(s[live] == 0) & (slot_rows[live] >= 0)]
    assert zero.numel() and bool(torch.isfinite(c[zero]).all())
    delta_bf16 = problem.delta64[zero].to(torch.bfloat16)
    assert torch.allclose(
        c[zero].double(), delta_bf16.double(), rtol=2.0**-7, atol=0.0
    ), (rows_mode, width)
    tiny = live[
        (problem.row_in_slab[live] == _ROWS_TINY[0])
        | (problem.row_in_slab[live] == _ROWS_TINY[1])
    ]
    assert bool((slot_rows[tiny] == _FUSED_SLOT_TINY).all())
    assert float(problem.base64[tiny].norm()) < 1e-3 * float(
        problem.delta64[tiny].norm()
    )
    assert float(_rel_l2(c[tiny].double(), problem.delta64[tiny])) < 2.5e-2, (
        rows_mode,
        width,
    )
    base_tiny = live[
        (problem.row_in_slab[live] == _ROWS_TINY[2])
        | (problem.row_in_slab[live] == _ROWS_TINY[3])
        | (problem.row_in_slab[live] == _ROW_BF16_MAX)
    ]
    assert torch.equal(c[base_tiny], plain[base_tiny])
    # The top-of-range bridges reached the output: their rows moved and are finite.
    top = live[slot_rows[live] == _FUSED_SLOT_TOP]
    assert (
        top.numel()
        and not torch.equal(c[top], plain[top])
        and bool(torch.isfinite(c[top]).all())
    )


# The epilogue's last multiply and the add of D round separately: alphas
# (full mantissa, 2^-10 exponent) for which fl(fl(t * s) + D) and
# fl(t * s + D) land on different BF16 values at t = fl(3 * 2^-17 * alpha),
# s = 192 and D = -bf16(t * s), found by search on 2026-11-08.
_FUSED_ROUNDING_ALPHAS = (
    0.0010936723556369543,
    0.0012206296669319272,
    0.001305096666328609,
    0.0011003249092027545,
)


def test_fused_per_token_epilogue_contract_rows():
    """The 4.1 contract rows under the fused build: with a zero bridge (D is
    an exact zero) the tail row, the edge row and the bits row are bitwise
    the fp32 model (C * alpha) * s of the unfused contract test, the ordinary
    rows within half a BF16 ulp plus the accumulation term with their deltas
    added; and the epilogue rounds (C * alpha) * s before adding D: on a
    bits row (one product, C = 3 * 2^-17, s = 192) under alphas whose product
    rounds in fp32, with D = -bf16(fl(C * alpha) * s) so the output is the
    rounding residual, the kernel lands on fl(fl(t * s) + D) and not on the
    fused multiply-add fl(t * s + D), which differs in BF16 for every alpha
    pinned here."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(8)
    n, k, m_max, rank = 256, 512, 64, 16
    tail_row, edge_row, bits_row = 1, 2, 3
    contract_alpha = (1.0 + 2.0**-7) * 2.0**-126
    experts = 1 + len(_FUSED_ROUNDING_ALPHAS)
    alphas = torch.tensor(
        [contract_alpha, *_FUSED_ROUNDING_ALPHAS], dtype=torch.float32, device=device
    )
    rows = torch.randn((m_max, k), generator=g)
    rows *= torch.pow(2.0, torch.rand((m_max, 1), generator=g) * 28 - 10)
    rows[0].zero_()
    rows[tail_row].zero_()
    rows[tail_row, 0] = 2.0**30
    rows[tail_row, 16] = 4700.0
    rows[tail_row, 17] = 388.0
    rows[edge_row].zero_()
    rows[edge_row, 0] = 512.0 * 2688.0
    rows[edge_row, 16] = 6.0
    rows[edge_row, 17] = 0.5
    rows[bits_row].zero_()
    rows[bits_row, 0] = 192.0 * 2688.0
    rows[bits_row, 16] = 2.25
    rows[bits_row, 17] = 0.5625
    tokens = rows.to(torch.bfloat16).unsqueeze(0).repeat(experts, 1, 1).to(device)
    t_data, t_sf, s, t_deq = _quant_nvfp4_per_token_ref(tokens)
    assert float(s[0, edge_row]) == 512.0 and float(s[0, bits_row]) == 192.0
    weights = torch.randn((1, n, k), generator=g) * 0.5
    weights[0, 0, :32] = 0.0
    weights[0, 0, 17] = 0.5 * 2.0**-6
    weights = weights.repeat(experts, 1, 1).to(device)
    w_data, w_sf, w_deq = _quant_nvfp4_ref(
        weights, torch.ones(experts, device=device), dequant_dtype=torch.float64
    )
    assert int(w_sf[0, 0, 1]) == 0x08 and int(w_data[0, 0, 8] >> 4) == 1
    w_deq = w_deq * alphas.double()[:, None, None]
    # One slot: weight -1 on output 0 from rank 0 (D = -z_0 there), zero
    # elsewhere for the rounding experts; random weights and bridges, with
    # the three contract rows' bridges zero, for the contract expert.
    lora_w = torch.zeros((experts, n, rank), dtype=torch.bfloat16, device=device)
    lora_w[0] = (torch.randn((n, rank), generator=g) * 0.2).to(torch.bfloat16)
    lora_w[1:, 0, 0] = -1.0
    planes = torch.zeros((experts, m_max, rank), dtype=torch.bfloat16, device=device)
    planes[0] = (torch.randn((m_max, rank), generator=g) * 0.5).to(torch.bfloat16)
    planes[0, [tail_row, edge_row, bits_row]] = 0.0
    c_one = 3.0 * 2.0**-17
    models = []
    for e, alpha in enumerate(_FUSED_ROUNDING_ALPHAS, start=1):
        t = torch.tensor(c_one, dtype=torch.float32) * torch.tensor(
            alpha, dtype=torch.float32
        )
        product = t.double() * 192.0  # exact
        zb = product.to(torch.bfloat16)
        planes[e, bits_row, 0] = zb
        two = (
            (t * torch.tensor(192.0, dtype=torch.float32)).double() - zb.double()
        ).float()
        fma = (product - zb.double()).float()
        assert two.to(torch.bfloat16) != fma.to(torch.bfloat16), alpha
        models.append((two.to(torch.bfloat16), fma.to(torch.bfloat16)))
    c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
    masked_m = torch.full((experts,), m_max, dtype=torch.int32, device=device)
    schedule, tiles, _, _ = build_dual_stage_schedules_masked(
        masked_m,
        m_max=m_max,
        token_width=64,
        n_gemm1=n,
        n_gemm2=n,
        output_width=128,
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
    )
    prepare_masked_blockscaled(
        t_data,
        w_data,
        pack_scale_atoms(t_sf),
        pack_scale_atoms(w_sf),
        c,
        masked_m,
        config=GroupedGemmConfig(mma_tiler_mn=(128, 64), persistent_clusters=8),
        sf_vec_size=16,
        alpha=alphas.view(-1, 1).contiguous(),
        alpha_rows_per_half=n,
        tok_scale=s.contiguous(),
        direct_schedule=schedule,
        schedule_tiles=tiles,
        lora_w=lora_w,
        lora_planes=planes,
        lora_slots=torch.ones(1, dtype=torch.int32, device=device),
        lora_acc="separate",
    ).launch()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(c).all())
    # The contract expert: the pinned rows as the unfused contract test, the
    # ordinary rows with their deltas.
    out = c[0].double()
    ref = t_deq[0] @ w_deq[0].T + planes[0].double() @ lora_w[0].double().T
    summands = (
        t_deq[0].abs() @ w_deq[0].abs().T
        + planes[0].double().abs() @ lora_w[0].double().abs().T
    )
    ordinary = torch.ones(m_max, dtype=torch.bool, device=device)
    ordinary[[tail_row, edge_row, bits_row]] = False
    budget = 0.5 * _bf16_ulp(ref) + 2.0**-14 * summands
    err = (out - ref).abs()
    assert bool((err[ordinary] <= budget[ordinary]).all()), float(
        (err[ordinary] / budget[ordinary].clamp(min=1e-300)).max()
    )
    t32 = torch.tensor(2.0**-17, dtype=torch.float32, device=device) * torch.tensor(
        contract_alpha, dtype=torch.float32, device=device
    )
    assert torch.equal(c[0, tail_row, 0], (t32 * s[0, tail_row]).to(torch.bfloat16))
    assert float(c[0, tail_row, 0]) != 0.0
    assert float(c[0, edge_row, 0]) == 0.0
    t32_bits = torch.tensor(
        3.0 * 2.0**-17, dtype=torch.float32, device=device
    ) * torch.tensor(contract_alpha, dtype=torch.float32, device=device)
    assert float(c[0, bits_row, 0]) == 2.0**-133 and torch.equal(
        c[0, bits_row, 0], (t32_bits * s[0, bits_row]).to(torch.bfloat16)
    )
    # The rounding experts: the residual lands on the two-rounding model.
    for e, (two, fma) in enumerate(models, start=1):
        assert torch.equal(c[e, bits_row, 0], two.to(device)), (
            _FUSED_ROUNDING_ALPHAS[e - 1],
            float(c[e, bits_row, 0]),
            float(two),
            float(fma),
        )


def test_separate_epilogue_lora_refuses_256_wide_tiles_and_builds_without_alpha():
    """The separate accumulator takes N TMEM columns beside the base
    accumulator's stages and the scale factors: a 256-wide token tile has no
    room for it, with static or per-token scales; and a build without alpha
    scales nothing for it to follow. Each is refused before anything
    compiles."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_masked_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_masked,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(3)
    experts, n, k, m_max, rank = 1, 128, 512, 256, 16
    tokens = (
        (torch.randn(experts, m_max, k, generator=g) * 0.5)
        .to(torch.bfloat16)
        .to(device)
    )
    t_data, t_sf, s, _ = _quant_nvfp4_per_token_ref(tokens)
    weights = (torch.randn(experts, n, k, generator=g) * 0.5).to(device)
    w_gs = torch.ones(experts, device=device)
    w_data, w_sf, _ = _quant_nvfp4_ref(weights, w_gs)
    lora = dict(
        lora_w=torch.zeros((experts, n, rank), dtype=torch.bfloat16, device=device),
        lora_planes=torch.zeros(
            (experts, m_max, rank), dtype=torch.bfloat16, device=device
        ),
        lora_slots=torch.ones(1, dtype=torch.int32, device=device),
        lora_acc="separate",
    )
    masked_m = torch.tensor([m_max], dtype=torch.int32, device=device)

    def build(width, tok_scale, alpha=w_gs.view(-1, 1).contiguous()):
        schedule, tiles, _, _ = build_dual_stage_schedules_masked(
            masked_m,
            m_max=m_max,
            token_width=width,
            n_gemm1=n,
            n_gemm2=n,
            output_width=128,
            cluster_shape_mn=(1, 1),
            use_2cta_instrs=False,
        )
        c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
        return prepare_masked_blockscaled(
            t_data,
            w_data,
            pack_scale_atoms(t_sf),
            pack_scale_atoms(w_sf),
            c,
            masked_m,
            config=GroupedGemmConfig(mma_tiler_mn=(128, width), persistent_clusters=2),
            sf_vec_size=16,
            alpha=alpha,
            alpha_rows_per_half=n,
            tok_scale=tok_scale,
            direct_schedule=schedule,
            schedule_tiles=tiles,
            **lora,
        )

    with pytest.raises(Exception, match="does not fit in TMEM"):
        build(256, s.contiguous())
    with pytest.raises(Exception, match="does not fit in TMEM"):
        build(256, None)
    with pytest.raises(Exception, match="build with alpha"):
        build(128, None, alpha=None)
