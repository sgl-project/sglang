"""Block-scaled CuTeDSL GEMMs against independent dequantized references.
The scale atoms follow tcgen05 128x4 order."""

from __future__ import annotations

import time

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


def _slot_list(slots, device) -> torch.Tensor:
    """The kernel's slot operand: the count, then the slots it applies."""
    slots = list(slots)
    return torch.tensor([len(slots), *slots], dtype=torch.int32, device=device)


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


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


def _wait_bounded(seconds: float) -> None:
    """Fail on a stalled GEMM instead of hanging the process."""
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


def _launch_build(builds: dict, prepare, c: torch.Tensor, lora: dict | None) -> None:
    """Reuse each compiled build, rebinding output and LoRA operands per launch."""
    from sglang.kernels.ops.lora.moe.cutedsl.api import as_dynamic_cute_tensor

    key = lora is not None
    if key not in builds:
        builds[key] = prepare(c, lora or {})
    build = builds[key]
    args = list(build.operand_args)
    args[4] = as_dynamic_cute_tensor(
        c if c.dim() == 3 else c.unsqueeze(0), leading_dim=2
    )
    if lora is not None:
        args[7:10] = (
            as_dynamic_cute_tensor(lora["lora_w"], leading_dim=2),
            as_dynamic_cute_tensor(lora["lora_planes"], leading_dim=2),
            as_dynamic_cute_tensor(lora["lora_slots"], leading_dim=0),
        )
    build.compiled_fn(
        *args,
        build.group_m_arg,
        build.direct_schedule_arg,
        build.schedule_tiles_arg,
        build.stream,
    )


class _EpilogueDownProblem:
    """Shared fixture for static and MX epilogue LoRA builds."""

    capacity = 8

    def __init__(self, fmt: str, rows_mode: str) -> None:
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
        experts, n, k, m_max, rank = 4, 256, 512, 256, 16
        width = 64 if rows_mode == "masked" else 128
        self.width = width
        self.experts, self.n, self.rank = experts, n, rank
        weights = (torch.randn(experts, n, k, generator=g) * 0.05).to(device)
        tokens = (torch.randn(experts, m_max, k, generator=g) * 0.5).to(device)
        w_gs = torch.full((experts,), 0.01, dtype=torch.float32, device=device)
        t_gs = torch.full((experts,), 0.1, device=device)
        w_data, w_sf, self.w_deq = _quantize(fmt, weights, w_gs)
        sf_weights = pack_scale_atoms(w_sf)
        alpha = None
        if fmt == "nvfp4":
            alpha = (w_gs * t_gs).reshape(experts, 1).contiguous()
        # Multiple output tiles per CTA exercise reuse of the LoRA stages.
        config = GroupedGemmConfig(mma_tiler_mn=(128, width), persistent_clusters=2)
        masked_m = torch.tensor([200, 0, 128, 7], dtype=torch.int32, device=device)
        self.lora_w = (
            torch.randn(self.capacity, experts, n, rank, generator=g) * 0.2
        ).to(device=device, dtype=torch.bfloat16)
        bridge = (torch.randn(experts, m_max, rank, generator=g) * 0.5).to(device)
        row_slot = torch.arange(m_max, device=device) % (self.capacity + 1) - 1
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
            self._c_shape = (experts, m_max, n)

            def prepare(c, lora):
                return prepare_masked_blockscaled(
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

        else:
            seg_offsets = torch.zeros(experts + 1, dtype=torch.int32, device=device)
            seg_offsets[1:] = torch.cumsum(((masked_m + 127) // 128) * 128, 0)
            m_ceil = int(seg_offsets[-1])
            flat = torch.zeros((m_ceil, k), device=device)
            self.expert_of_row = torch.zeros(m_ceil, dtype=torch.long, device=device)
            self.slot_rows = torch.full((m_ceil,), -1, dtype=torch.long, device=device)
            self.bridge_rows = torch.zeros((m_ceil, rank), device=device)
            live = []
            for e in range(experts):
                start, rows = int(seg_offsets[e]), int(masked_m[e])
                flat[start : start + rows] = tokens[e, :rows]
                self.expert_of_row[start : start + rows] = e
                self.slot_rows[start : start + rows] = row_slot[:rows]
                self.bridge_rows[start : start + rows] = bridge[e, :rows]
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
            self._c_shape = (m_ceil, n)

            def prepare(c, lora):
                return prepare_contiguous_blockscaled(
                    f_data,
                    w_data,
                    sf_tokens,
                    sf_weights,
                    c,
                    seg_offsets,
                    masked_m,
                    direct_schedule=schedule,
                    schedule_tiles=tiles,
                    **common,
                    **lora,
                )

        self._prepare, self._builds = prepare, {}
        self.device = device

    def operands(self, placement: list, slots: list | None = None) -> dict:
        # Each resident position holds one adapter's weights and bridge, or a zero bridge.
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
            lora_slots=_slot_list(
                range(resident) if slots is None else slots,
                self.device,
            ),
        )

    def run(self, lora: dict | None = None) -> torch.Tensor:
        c = torch.full(self._c_shape, 7.0, dtype=torch.bfloat16, device=self.device)
        _launch_build(self._builds, self._prepare, c, lora)
        _wait_bounded(60.0)
        return c.view(-1, self.n)

    def reference(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the FP64 base and each live row's own LoRA delta once."""
        experts = self.expert_of_row[self.live]
        base = torch.empty(
            (len(self.live), self.n), dtype=torch.float64, device=self.device
        )
        for e in range(self.experts):
            rows = experts == e
            base[rows] = (
                self.rows_deq[self.live[rows]].double() @ self.w_deq[e].double().T
            )
        weights = self.lora_w[self.slot_rows[self.live].clamp(min=0), experts]
        delta = torch.einsum(
            "rq,rnq->rn", self.bridge_rows[self.live].double(), weights.double()
        )
        return base, delta


@pytest.mark.parametrize(
    "fmt,rows_mode", [("mxfp8", "contiguous"), ("nvfp4", "masked")]
)
def test_blockscaled_epilogue_lora_slot_bounds(fmt, rows_mode):
    """Check sparse slots, TMA stage reuse and untouched padding."""
    _skip_unless_sm100()
    problem = _EpilogueDownProblem(fmt, rows_mode)
    live = problem.live
    slot_rows = problem.slot_rows[live]
    c_base = problem.run()
    untouched = (c_base == 7.0).all(dim=1)
    assert not bool(untouched[live].any())
    # The final expert has 7 live rows. Check padding inside its scheduled tile.
    last_start = int(live[-7])
    assert bool(untouched[last_start + 7 : last_start + problem.width].any())
    base, delta = problem.reference()

    def check(c, slots):
        active = torch.zeros_like(slot_rows, dtype=torch.bool)
        for slot in slots:
            active |= slot_rows == slot
        target = base + delta * active[:, None]
        assert _rel_l2(c[live].double(), target) < 1.5e-2, (fmt, rows_mode, slots)

    slots = list(range(5))  # Wrap the four LoRA TMA stages.
    check(problem.run(problem.operands(slots)), slots)

    placement = [None, 1, None, None, 4, None, None, 7]
    sparse = problem.run(problem.operands(placement, slots=[1, 4, 7]))
    check(sparse, [1, 4, 7])
    assert torch.equal(sparse[live], problem.run(problem.operands(placement))[live])

    # Slot 1 has a nonzero bridge, but the explicit list omits it.
    first = problem.run(problem.operands([0, 1], slots=[0]))
    check(first, [0])
    assert torch.equal(first[live[slot_rows == 1]], c_base[live[slot_rows == 1]])
    assert torch.equal(problem.run(problem.operands([0, 1], slots=[])), c_base)


def test_static_epilogue_lora_scales_the_base_alone_and_rounds_it_before_d():
    """Keep the LoRA delta finite at tiny alpha and round the base before adding it."""
    _skip_unless_sm100()
    from sglang.kernels.ops.lora.moe.cutedsl.api import (
        GroupedGemmConfig,
        prepare_contiguous_blockscaled,
    )
    from sglang.kernels.ops.lora.moe.cutedsl.schedule_builder import (
        build_dual_stage_schedules_contiguous,
    )
    from sglang.kernels.ops.lora.moe.quant_blockscaled import pack_scale_atoms

    device = torch.device("cuda")
    n, k, m_max, rank = 256, 512, 128, 16
    delta_row, base_row, residual_row = 0, 1, 2
    # The exact product is 2^-14 + 2^-39. Only a non-fused multiply rounds it
    # to 2^-14 before the LoRA delta cancels it.
    alpha = ((128 << 18) + 1) // 3 * 2.0**-33
    assert (3.0 * 2.0**-6) * alpha == 2.0**-14 + 2.0**-39
    # The 122B scale makes the old inverse-alpha bridge overflow.
    tiny_alpha = 0.008603050373494625 * 2.0711090409231365e-37
    alphas = torch.tensor([alpha, tiny_alpha], dtype=torch.float32, device=device)
    experts = alphas.numel()
    rounded = [2.0**-14, 0.0]
    tokens = torch.zeros((experts, m_max, k), device=device)
    tokens[:-1, [base_row, residual_row], 17] = 6.0
    t_data, t_sf, _ = quantize_nvfp4_static_ref(
        tokens, torch.ones(experts, device=device)
    )
    assert int(t_sf[0, base_row, 1]) == 0x38 and int(t_data[0, base_row, 8] >> 4) == 7
    weights = torch.zeros((experts, n, k), device=device)
    weights[:, 0, 17] = 0.5 * 2.0**-6
    w_data, w_sf, _ = quantize_nvfp4_static_ref(
        weights, torch.ones(experts, device=device)
    )
    assert int(w_sf[0, 0, 1]) == 0x08 and int(w_data[0, 0, 8] >> 4) == 1
    lora_w = torch.zeros((experts, n, rank), dtype=torch.bfloat16, device=device)
    lora_w[:, 0, 0] = -1.0
    planes = torch.zeros((experts, m_max, rank), dtype=torch.bfloat16, device=device)
    planes[:, delta_row, 0] = 3.0
    planes[:, residual_row, 0] = torch.tensor(rounded, device=device).to(torch.bfloat16)
    c = torch.zeros((experts, m_max, n), dtype=torch.bfloat16, device=device)
    masked_m = torch.full((experts,), m_max, dtype=torch.int32, device=device)
    offsets = torch.arange(experts + 1, device=device, dtype=torch.int32) * m_max
    schedule, tiles, _, _ = build_dual_stage_schedules_contiguous(
        masked_m,
        m_pad_ceiling=experts * m_max,
        max_expert_rows=m_max,
        m_alignment=m_max,
        token_width=128,
        n_gemm1=n,
        n_gemm2=n,
        output_width=128,
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
    )

    prepare_contiguous_blockscaled(
        t_data.flatten(0, 1),
        w_data,
        pack_scale_atoms(t_sf),
        pack_scale_atoms(w_sf),
        c.flatten(0, 1),
        offsets,
        masked_m,
        config=GroupedGemmConfig(mma_tiler_mn=(128, 128), persistent_clusters=8),
        sf_vec_size=16,
        alpha=alphas.view(-1, 1).contiguous(),
        alpha_rows_per_half=n,
        direct_schedule=schedule,
        schedule_tiles=tiles,
        lora_w=lora_w,
        lora_planes=planes.flatten(0, 1).unsqueeze(0),
        lora_slots=_slot_list([0], device),
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


def _bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    """One BF16 ulp at each magnitude (the subnormal grid below 2^-126)."""
    exponent = torch.floor(torch.log2(x.abs().clamp(min=2.0**-126)))
    return torch.pow(2.0, exponent - 7)


def test_fused_per_token_epilogue_contract_rows():
    """Preserve tiny base outputs and round scaling before adding LoRA."""
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
    rounding_alpha = 0.0012206296669319272  # separates FP32 multiply/add from FMA
    experts = 2
    alphas = torch.tensor(
        [contract_alpha, rounding_alpha], dtype=torch.float32, device=device
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
    t_data, t_sf, s, t_deq, _ = quantize_nvfp4_per_token_ref(tokens)
    # The nonzero tail result requires the MMA to consume a subnormal E4M3 scale.
    assert int(t_sf[0, tail_row, 1]) == 0x01 and int(t_data[0, tail_row, 8] >> 4) == 1
    assert float(s[0, edge_row]) == 512.0 and float(s[0, bits_row]) == 192.0
    weights = torch.randn((1, n, k), generator=g) * 0.5
    weights[0, 0, :32] = 0.0
    weights[0, 0, 17] = 0.5 * 2.0**-6
    weights = weights.repeat(experts, 1, 1).to(device)
    w_data, w_sf, w_deq = quantize_nvfp4_static_ref(
        weights, torch.ones(experts, device=device), dequant_dtype=torch.float64
    )
    assert int(w_sf[0, 0, 1]) == 0x08 and int(w_data[0, 0, 8] >> 4) == 1
    w_deq = w_deq * alphas.double()[:, None, None]
    # The rounding experts cancel output 0; the pinned contract rows omit LoRA.
    lora_w = torch.zeros((experts, n, rank), dtype=torch.bfloat16, device=device)
    lora_w[0] = (torch.randn((n, rank), generator=g) * 0.2).to(torch.bfloat16)
    lora_w[1:, 0, 0] = -1.0
    planes = torch.zeros((experts, m_max, rank), dtype=torch.bfloat16, device=device)
    planes[0] = (torch.randn((m_max, rank), generator=g) * 0.5).to(torch.bfloat16)
    planes[0, [tail_row, edge_row, bits_row]] = 0.0
    c_one = 3.0 * 2.0**-17
    t = torch.tensor(c_one, dtype=torch.float32) * torch.tensor(
        rounding_alpha, dtype=torch.float32
    )
    product = t.double() * 192.0
    zb = product.to(torch.bfloat16)
    planes[1, bits_row, 0] = zb
    two = ((t * 192.0).double() - zb.double()).float().to(torch.bfloat16)
    fma = (product - zb.double()).float().to(torch.bfloat16)
    assert two != fma
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
    sf_tokens, sf_weights = pack_scale_atoms(t_sf), pack_scale_atoms(w_sf)
    plain = torch.zeros_like(c)
    common = dict(
        config=GroupedGemmConfig(mma_tiler_mn=(128, 64), persistent_clusters=8),
        sf_vec_size=16,
        alpha=alphas.view(-1, 1).contiguous(),
        alpha_rows_per_half=n,
        tok_scale=s.contiguous(),
        direct_schedule=schedule,
        schedule_tiles=tiles,
    )
    prepare_masked_blockscaled(
        t_data, w_data, sf_tokens, sf_weights, plain, masked_m, **common
    ).launch()
    prepare_masked_blockscaled(
        t_data,
        w_data,
        sf_tokens,
        sf_weights,
        c,
        masked_m,
        **common,
        lora_w=lora_w,
        lora_planes=planes,
        lora_slots=_slot_list([0], device),
    ).launch()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(c).all())
    # Use the FP64 tolerance on ordinary rows and exact checks on the pinned rows.
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
    assert torch.equal(
        plain[0, [tail_row, edge_row, bits_row]], c[0, [tail_row, edge_row, bits_row]]
    )
    t32_bits = torch.tensor(
        3.0 * 2.0**-17, dtype=torch.float32, device=device
    ) * torch.tensor(contract_alpha, dtype=torch.float32, device=device)
    assert float(c[0, bits_row, 0]) == 2.0**-133 and torch.equal(
        c[0, bits_row, 0], (t32_bits * s[0, bits_row]).to(torch.bfloat16)
    )
    assert torch.equal(c[1, bits_row, 0], two.to(device)), (two, fma)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
