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


def _slot_list(slots, device) -> torch.Tensor:
    """The kernel's slot operand: the count, then the slots it applies."""
    slots = list(slots)
    return torch.tensor([len(slots), *slots], dtype=torch.int32, device=device)


def _rel_l2(a, b):
    return (a - b).norm() / b.norm().clamp(min=1e-12)


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

    def __init__(self, fmt: str, rows_mode: str, rank: int) -> None:
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
        experts, n, k, m_max = 4, 256, 512, 256
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
    "fmt,rows_mode,rank",
    [("mxfp8", "contiguous", 16), ("nvfp4", "masked", 16), ("mxfp8", "contiguous", 96)],
)
def test_blockscaled_epilogue_lora_slot_bounds(fmt, rows_mode, rank):
    """Check sparse slots, TMA stage reuse and untouched padding."""
    _skip_unless_sm100()
    problem = _EpilogueDownProblem(fmt, rows_mode, rank)
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

    slots = list(range(5))  # Wrap the LoRA TMA stages, reduced at rank 96.
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
    n, k, token_width, rank = 256, 512, 64, 16
    m_max = token_width + 1
    tail_row, edge_row, bits_row = 1, 2, 3
    contract_alpha = (1.0 + 2.0**-7) * 2.0**-126
    rounding_alpha = 0.0012206296669319272  # separates FP32 multiply/add from FMA
    experts = 2
    alphas = torch.tensor(
        [contract_alpha, rounding_alpha], dtype=torch.float32, device=device
    )
    rows = torch.randn((token_width, k), generator=g)
    rows *= torch.pow(2.0, torch.rand((token_width, 1), generator=g) * 28 - 10)
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
    # Put one live row in a partial tile without changing the witnesses above.
    rows = torch.cat((rows, rows[4:5]))
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
    planes[0, :token_width] = (torch.randn((token_width, rank), generator=g) * 0.5).to(
        torch.bfloat16
    )
    planes[0, -1] = planes[0, 4]
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
        token_width=token_width,
        n_gemm1=n,
        n_gemm2=n,
        output_width=128,
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
    )
    sf_tokens, sf_weights = pack_scale_atoms(t_sf), pack_scale_atoms(w_sf)
    plain = torch.zeros_like(c)
    common = dict(
        config=GroupedGemmConfig(
            mma_tiler_mn=(128, token_width), persistent_clusters=8
        ),
        sf_vec_size=16,
        alpha=alphas.view(-1, 1).contiguous(),
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
    assert torch.equal(c[:, -1], c[:, 4])
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


@pytest.mark.parametrize("fmt", list(FORMATS))
def test_blockscaled_can_implement_rejects_partial_atoms_and_wide_separate_lora(fmt):
    """K must fill whole scale atoms; a separate LoRA accumulator needs N <= 128."""
    _skip_unless_sm100()
    import cutlass

    from sglang.kernels.ops.lora.moe.cutedsl.kernel_sm100_blockscaled import (
        GroupedGemmKernelSm100BlockScaled,
    )

    vec = FORMATS[fmt]["vec"]
    dtype = cutlass.Float8E4M3FN if fmt == "mxfp8" else cutlass.Float4E2M1FN
    sf_dtype = cutlass.Float8E4M3FN if fmt == "nvfp4" else cutlass.Float8E8M0FNU

    def can_implement(k, n_tile=128, lora_acc="shared"):
        gemm = GroupedGemmKernelSm100BlockScaled(
            cutlass.Float32,
            False,
            (128, n_tile),
            (1, 1),
            swap_ab=True,
            sf_vec_size=vec,
            sf_dtype=sf_dtype,
            lora_k=16,
            lora_acc=lora_acc,
        )
        return gemm.can_implement(
            (256, 128, k, 1), dtype, dtype, cutlass.BFloat16, "k", "k", "m"
        )

    assert can_implement(8 * vec)
    # Six groups pass the 16-byte alignment but leave the second atom half full.
    assert not can_implement(6 * vec)
    assert can_implement(8 * vec, n_tile=256)
    assert not can_implement(8 * vec, n_tile=256, lora_acc="separate")
    assert can_implement(8 * vec, n_tile=128, lora_acc="separate")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
