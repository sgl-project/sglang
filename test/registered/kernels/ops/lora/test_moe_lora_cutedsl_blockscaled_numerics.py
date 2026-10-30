"""Block-scaled CuTeDSL grouped GEMMs against exact dequantized references.

MXFP8 (E4M3 x E8M0/32), MXFP4 (E2M1 x E8M0/32) and NVFP4 (E2M1 x E4M3/16)
operands are quantized on the host with the same rules as the Triton row
quantizer; the scale atoms follow the tcgen05 128x4 block order.
"""

from __future__ import annotations

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")

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


def _quant_nvfp4_ref(x: torch.Tensor, global_scale: torch.Tensor):
    """NVFP4 with E4M3 group-16 scales under a per-slab global scale."""
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
    dequant = (values * denom[..., None]).reshape(*lead, k)
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
