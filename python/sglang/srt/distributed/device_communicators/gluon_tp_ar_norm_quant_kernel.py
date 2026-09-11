"""TP4 all-reduce, Gemma RMSNorm, and BF16-to-group-FP8 conversion.

M2 overlaps scalar pointer-table reads and keeps its acquisition address scalar.
M16/M32 use scalar completion polling after the existing final-CTA election.
All shapes preserve the production arithmetic and caller-owned epoch protocol.
"""

import torch
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd.cdna4 import buffer_load, buffer_store


@gluon.jit
def _notify(locks, increment: gl.constexpr, active, publish: gl.constexpr):
    """One wave pushes integer notifications to the three peer-owned counters."""
    if publish:
        gl.inline_asm_elementwise(
            """
            v_cmp_ne_u32 vcc, 0, $3
            s_and_saveexec_b64 $0, vcc
            s_cbranch_execz 1f
            s_waitcnt vmcnt(0) lgkmcnt(0)
            global_atomic_add $1, $2, off sc1
            1:
            s_or_saveexec_b64 $0, $0
            """,
            "=&s,v,v,v,~{vcc},~{scc},~{memory}",
            [locks, increment, active.to(gl.int32)],
            dtype=gl.uint64,
            is_pure=False,
            pack=1,
        )
    else:
        gl.inline_asm_elementwise(
            """
            v_cmp_ne_u32 vcc, 0, $3
            s_and_saveexec_b64 $0, vcc
            global_atomic_add $1, $2, off sc1
            s_or_saveexec_b64 $0, $0
            """,
            "=&s,v,v,v,~{vcc},~{scc},~{memory}",
            [locks, increment, active.to(gl.int32)],
            dtype=gl.uint64,
            is_pure=False,
            pack=1,
        )


@gluon.jit
def _scalar_acquire(counter, target, thread):
    """Only wave zero polls; the caller subsequently joins the whole CTA."""
    address = counter.to(gl.uint64)
    gl.inline_asm_elementwise(
        """
        v_cmp_eq_u32 vcc, 0, $5
        s_cbranch_vccz 2f
        v_readfirstlane_b32 s0, $2
        v_readfirstlane_b32 s1, $3
        v_readfirstlane_b32 $1, $4
        1:
        s_load_dword $0, s[0:1], 0x0 glc
        s_waitcnt lgkmcnt(0)
        s_sub_u32 $0, $0, $1
        s_cmp_ge_i32 $0, 0
        s_cbranch_scc0 1b
        2:
        """,
        "=&s,=&s,v,v,v,v,~{s0},~{s1},~{scc},~{vcc},~{memory}",
        [address.to(gl.uint32), (address >> 32).to(gl.uint32), target, thread],
        dtype=(gl.uint32, gl.uint32),
        is_pure=False,
        pack=1,
    )


# M2 already has a scalar local pointer; retain it through arrival acquisition.
@gluon.jit
def _scalar_acquire_uniform(counter, target, thread):
    gl.inline_asm_elementwise(
        """
        v_cmp_eq_u32 vcc, 0, $4
        s_cbranch_vccz 2f
        v_readfirstlane_b32 $1, $3
        1:
        s_load_dword $0, $2, 0x0 glc
        s_waitcnt lgkmcnt(0)
        s_sub_u32 $0, $0, $1
        s_cmp_ge_i32 $0, 0
        s_cbranch_scc0 1b
        2:
        """,
        "=&s,=&s,s,v,v,~{scc},~{vcc},~{memory}",
        [counter.to(gl.uint64), target, thread],
        dtype=(gl.uint32, gl.uint32),
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _vector_acquire(counter, target, thread, materialize: gl.constexpr):
    condition = (thread == 0).to(gl.uint32) if materialize else thread
    compare: gl.constexpr = (
        "v_cmp_ne_u32 vcc, 0, $4" if materialize else "v_cmp_eq_u32 vcc, 0, $4"
    )
    gl.inline_asm_elementwise(
        compare
        + """
        s_and_saveexec_b64 $0, vcc
        s_cbranch_execz 2f
        1:
        global_load_dword $1, $2, off sc1
        s_waitcnt vmcnt(0)
        v_sub_u32 $1, $1, $3
        v_cmp_ge_i32 vcc, $1, 0
        s_cbranch_vccz 1b
        2:
        s_or_saveexec_b64 $0, $0
        """,
        "=&s,=&v,v,v,v,~{vcc},~{scc},~{memory}",
        [counter, target, condition],
        dtype=(gl.uint64, gl.uint32),
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _reserve_epoch(local, thread, step: gl.constexpr):
    """Reserve a CTA ticket. Only lane zero consumes its returned epoch."""
    _, ticket = gl.inline_asm_elementwise(
        """
        v_mov_b32 $1, 0
        v_cmp_eq_u32 vcc, 0, $4
        s_and_saveexec_b64 $0, vcc
        s_cbranch_execz 1f
        global_atomic_add $1, $2, $3, off offset:8 sc0 sc1
        s_waitcnt vmcnt(0)
        1:
        s_or_saveexec_b64 $0, $0
        """,
        "=&s,=&v,v,v,v,~{vcc},~{scc},~{memory}",
        [local, step, thread],
        dtype=(gl.uint64, gl.uint32),
        is_pure=False,
        pack=1,
    )
    return ticket


@gluon.jit
def _advance_single_row(local, next_progress, thread):
    """A one-CTA invocation needs no ticket-election atomic."""
    gl.inline_asm_elementwise(
        """
        v_cmp_eq_u32 vcc, 0, $3
        s_and_saveexec_b64 $0, vcc
        global_store_dword $1, $2, off offset:8
        s_or_saveexec_b64 $0, $0
        """,
        "=&s,v,v,v,~{vcc},~{scc},~{memory}",
        [local, next_progress, thread],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )
    return (thread == 0).to(gl.uint32)


@gluon.jit
def _finish(local, target, last):
    gl.inline_asm_elementwise(
        """
        v_cmp_ne_u32 vcc, 0, $4
        s_and_saveexec_b64 $0, vcc
        s_cbranch_execz 2f
        1:
        global_load_dword $1, $2, off offset:4 sc1
        s_waitcnt vmcnt(0)
        v_sub_u32 $1, $1, $3
        v_cmp_ge_i32 vcc, $1, 0
        s_cbranch_vccz 1b
        2:
        s_or_saveexec_b64 $0, $0
        """,
        "=&s,=&v,v,v,v,~{vcc},~{scc},~{memory}",
        [local, target, last],
        dtype=(gl.uint64, gl.uint32),
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _finish_late(local, target, thread, step: gl.constexpr):
    gl.inline_asm_elementwise(
        """
        v_cmp_eq_u32 vcc, 0, $6
        s_and_saveexec_b64 $0, vcc
        s_cbranch_execz 3f
        global_atomic_add $1, $3, $4, off offset:8 sc0
        s_waitcnt vmcnt(0)
        v_add_u32 $1, $4, $1
        v_mul_lo_u32 $1, 3, $1
        v_cmp_eq_u32 vcc, $1, $5
        s_cbranch_vccz 3f
        2:
        global_load_dword $2, $3, off offset:4 sc1
        s_waitcnt vmcnt(0)
        v_sub_u32 $2, $2, $5
        v_cmp_ge_i32 vcc, $2, 0
        s_cbranch_vccz 2b
        3:
        s_or_saveexec_b64 $0, $0
        """,
        "=&s,=&v,=&v,v,v,v,v,~{vcc},~{scc},~{memory}",
        [local, step, target, thread],
        dtype=(gl.uint64, gl.uint32, gl.uint32),
        is_pure=False,
        pack=1,
    )


# Only the final CTA polls. Earlier CTAs remain free to retire on a single CU.
@gluon.jit
def _finish_uniform_late(local, target, thread, step: gl.constexpr):
    gl.inline_asm_elementwise(
        """
        v_cmp_eq_u32 vcc, 0, $6
        s_and_saveexec_b64 $0, vcc
        s_cbranch_execz 3f
        global_atomic_add $1, $3, $4, off offset:8 sc0
        s_waitcnt vmcnt(0)
        v_add_u32 $1, $4, $1
        v_mul_lo_u32 $1, 3, $1
        v_cmp_eq_u32 vcc, $1, $5
        s_cbranch_vccz 3f
        v_readfirstlane_b32 s0, $5
        2:
        s_load_dword $2, $7, 0x4 glc
        s_waitcnt lgkmcnt(0)
        s_sub_u32 $2, $2, s0
        s_cmp_ge_i32 $2, 0
        s_cbranch_scc0 2b
        3:
        s_or_saveexec_b64 $0, $0
        """,
        "=&s,=&v,=&s,v,v,v,v,s,~{s0},~{vcc},~{scc},~{memory}",
        [local, step, target, thread, local.to(gl.uint64)],
        dtype=(gl.uint64, gl.uint32, gl.uint32),
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_pointer(table):
    return gl.inline_asm_elementwise(
        "s_load_dwordx2 $0, $1, 0x0\n s_waitcnt lgkmcnt(0)",
        "=s,s,~{memory}",
        [table.to(gl.uint64)],
        dtype=gl.uint64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _load_inputs(table, owner):
    return gl.inline_asm_elementwise(
        """
        s_load_dwordx2 $0, $4, $5
        s_load_dwordx2 $1, $4, $6
        s_load_dwordx2 $2, $4, $7
        s_load_dwordx2 $3, $4, $8
        s_waitcnt lgkmcnt(0)
        """,
        "=&s,=&s,=&s,=&s,s,s,s,s,s,~{memory}",
        [
            table.to(gl.uint64),
            owner * 8,
            ((owner + 1) % 4) * 8,
            ((owner + 2) % 4) * 8,
            ((owner + 3) % 4) * 8,
        ],
        dtype=(gl.uint64, gl.uint64, gl.uint64, gl.uint64),
        is_pure=False,
        pack=1,
    )


# Issue the five independent M2 pointer reads before a single scalar-memory wait.
@gluon.jit
def _load_tables(lock_table, input_table):
    return gl.inline_asm_elementwise(
        """
        s_load_dwordx2 $0, $5, 0x0
        s_load_dwordx2 $1, $6, 0x0
        s_load_dwordx2 $2, $6, 0x8
        s_load_dwordx2 $3, $6, 0x10
        s_load_dwordx2 $4, $6, 0x18
        s_waitcnt lgkmcnt(0)
        """,
        "=&s,=&s,=&s,=&s,=&s,s,s,~{memory}",
        [lock_table.to(gl.uint64), input_table.to(gl.uint64)],
        dtype=(gl.uint64, gl.uint64, gl.uint64, gl.uint64, gl.uint64),
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _store_local(pointer, offsets, values, use_buffer: gl.constexpr):
    if use_buffer:
        buffer_store(values.to(pointer.dtype.element_ty), pointer, offsets)
    else:
        gl.store(pointer + offsets, values)


@gluon.jit
def _fused(
    input_peer_ptrs,
    lock_peer_ptrs,
    residual_ptr,
    weight_ptr,
    normalized_ptr,
    residual_out_ptr,
    quantized_ptr,
    scales_ptr,
    reduced_ptr,
    M: gl.constexpr,
    ROUND: gl.constexpr,
    FP8_MAX: gl.constexpr,
    EPS: gl.constexpr,
):
    PAIR: gl.constexpr = M == 64
    NCTA: gl.constexpr = M // 2 if PAIR else M
    TICKET: gl.constexpr = M >= 2 and M <= 8
    NW: gl.constexpr = 16 if PAIR else 8
    pid = gl.program_id(0)
    if TICKET:
        thread = gl.arange(0, NW * 64, layout=gl.BlockedLayout([1], [64], [NW], [0]))
        peer = thread % 4
        locks = gl.load(lock_peer_ptrs + peer).to(gl.pointer_type(gl.uint32))
        if pid == 0:
            _notify(locks, 128, (thread > 0) & (thread < 4), M <= 4)
        if M == 2:
            local_address, a0, a1, a2, a3 = _load_tables(
                lock_peer_ptrs, input_peer_ptrs
            )
            local = local_address.to(gl.pointer_type(gl.uint32))
        else:
            local = _load_pointer(lock_peer_ptrs).to(gl.pointer_type(gl.uint32))
        # Returned tickets stay valid even when another CTA advances word 2.
        # Only lane zero uses target/last; arrival acquisition joins the CTA.
        ticket = _reserve_epoch(local, thread, 128 // NCTA)
        target = ((ticket & 0xFFFFFF80) + 128) * 3
        last = ((thread == 0) & ((ticket & 127) == 128 - 128 // NCTA)).to(gl.uint32)
    elif M == 1:
        local = _load_pointer(lock_peer_ptrs).to(gl.pointer_type(gl.uint32))
        # Local progress advances by 128 per call, regardless of grid size.
        # The epoch cannot change until every CTA has sampled it.
        progress = gl.inline_asm_elementwise(
            "global_load_dword $0, $1, off offset:8 sc1\n s_waitcnt vmcnt(0)",
            "=v,v,~{memory}",
            [local],
            dtype=gl.uint32,
            is_pure=False,
            pack=1,
        )
        next_progress = (progress & 0xFFFFFF80) + 128
        target = next_progress * 3
        thread = gl.arange(0, NW * 64, layout=gl.BlockedLayout([1], [64], [NW], [0]))
        peer = thread % 4
        locks = gl.load(lock_peer_ptrs + peer).to(gl.pointer_type(gl.uint32))
        if pid == 0:
            _notify(locks, 128, (thread > 0) & (thread < 4), True)
    else:
        thread = gl.arange(0, NW * 64, layout=gl.BlockedLayout([1], [64], [NW], [0]))
        peer = thread % 4
        locks = gl.load(lock_peer_ptrs + peer).to(gl.pointer_type(gl.uint32))
        if pid == 0:
            _notify(locks, 128, (thread > 0) & (thread < 4), False)
        local = _load_pointer(lock_peer_ptrs).to(gl.pointer_type(gl.uint32))
        progress = gl.inline_asm_elementwise(
            "global_load_dword $0, $1, off offset:8 sc1\n s_waitcnt vmcnt(0)",
            "=v,v,~{memory}",
            [local],
            dtype=gl.uint32,
            is_pure=False,
            pack=1,
        )
        next_progress = (progress & 0xFFFFFF80) + 128
        target = next_progress * 3

    # Four adjacent elements per lane preserve the production RMS sum tree.
    L: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 2, 32], [2 if PAIR else 1, 8, 1], [2, 1, 0]
    )
    rows = pid * (2 if PAIR else 1) + gl.arange(
        0, 2 if PAIR else 1, layout=gl.SliceLayout(1, gl.SliceLayout(2, L))
    )
    groups = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, L)))
    within = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, L)))
    cols = groups[None, :, None] * 128 + within[None, None, :]
    offsets = rows[:, None, None] * 2048 + cols
    if M == 2 or M == 16:
        residual = buffer_load(residual_ptr, offsets).to(gl.float32)
        weight = buffer_load(weight_ptr, cols).to(gl.float32)
    else:
        residual = gl.load(residual_ptr + offsets).to(gl.float32)
        weight = gl.load(weight_ptr + cols).to(gl.float32)
    owner = pid // 8 if PAIR else 0
    if M != 2:
        a0, a1, a2, a3 = _load_inputs(input_peer_ptrs, owner)
    p0 = a0.to(gl.pointer_type(gl.bfloat16))
    p1 = a1.to(gl.pointer_type(gl.bfloat16))
    p2 = a2.to(gl.pointer_type(gl.bfloat16))
    p3 = a3.to(gl.pointer_type(gl.bfloat16))
    if M == 2:
        _scalar_acquire_uniform(local, target, thread)
    elif M <= 4:
        _scalar_acquire(local, target, thread)
    else:
        _vector_acquire(local, target, thread, M == 8)
    gl.barrier()
    x0 = gl.load(p0 + offsets, cache_modifier=".ca").to(gl.float32)
    x1 = gl.load(p1 + offsets, cache_modifier=".ca").to(gl.float32)
    x2 = gl.load(p2 + offsets, cache_modifier=".ca").to(gl.float32)
    x3 = gl.load(p3 + offsets, cache_modifier=".ca").to(gl.float32)
    reduced = (((x0 + x1) + x2) + x3).to(gl.bfloat16, fp_downcast_rounding=ROUND)
    if M <= 2:
        reduced = gl.inline_asm_elementwise(
            "",
            "=v,0,~{memory}",
            [reduced],
            dtype=gl.bfloat16,
            is_pure=False,
            pack=2,
        )
    else:
        _store_local(reduced_ptr, offsets, reduced, M <= 2 or M == 16)
        reduced = gl.inline_asm_elementwise(
            "",
            "=v,0,~{memory}",
            [reduced],
            dtype=gl.bfloat16,
            is_pure=False,
            pack=2,
        )
    # Publish only after every wave has consumed all four peer payloads.
    # The final ticket holder waits; other CTAs can retire on a single-CU stream.
    gl.barrier()
    if M == 1:
        last = _advance_single_row(local, next_progress, thread)
    _notify(locks + 1, 128 // NCTA, (thread > 0) & (thread < 4), False)
    if M <= 2:
        _store_local(reduced_ptr, offsets, reduced, M <= 2 or M == 16)

    value = reduced.to(gl.float32) + residual
    residual_out = value.to(gl.bfloat16, fp_downcast_rounding=ROUND)
    variance = gl.sum(gl.sum(value * value, 2), 1) / 2048.0
    normalized = (value * gl.rsqrt(variance[:, None, None] + EPS)) * weight
    normalized = normalized.to(gl.bfloat16, fp_downcast_rounding=ROUND)
    _store_local(normalized_ptr, offsets, normalized, M <= 2 or M == 16)
    _store_local(residual_out_ptr, offsets, residual_out, M <= 2 or M == 16)
    values = normalized.to(gl.float32)
    maximum = gl.maximum(gl.max(gl.abs(values), 2), 1.0e-10)
    scales = maximum * (1.0 / FP8_MAX)
    if M == 4:
        _store_local(scales_ptr, rows[:, None] * 16 + groups[None, :], scales, False)
    quantized = gl.clamp(values * (1.0 / scales[:, :, None]), -FP8_MAX, FP8_MAX)
    _store_local(quantized_ptr, offsets, quantized, M <= 2 or M == 16)
    if M != 4:
        _store_local(
            scales_ptr,
            rows[:, None] * 16 + groups[None, :],
            scales,
            M <= 2 or M == 16,
        )
    if M <= 8:
        _finish(local, target, last)
    elif M == 16 or M == 32:
        _finish_uniform_late(local, target, thread, 128 // NCTA)
    else:
        _finish_late(local, target, thread, 128 // NCTA)


def tp4_allreduce_add_gemma_rmsnorm_group_fp8_quant_gluon(
    local_input: torch.Tensor,
    input_peer_ptrs: torch.Tensor,
    lock_peer_ptrs: torch.Tensor,
    residual: torch.Tensor,
    gemma_weight: torch.Tensor,
    *,
    eps: float = 1.0e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized, residual, FP8, scales, and the all-reduce witness.

    Caller-owned synchronization words start at zero and persist across ordered
    invocations. Words 0/1 count arrivals/completed reads in 384-unit epochs;
    word 2 tracks local CTA progress in 128-unit epochs. M2--M8 atomically
    reserve tickets to acquire the epoch and elect the final CTA together.
    M1 needs only one progress store; M16--M64 retain late completion election.
    Signed differences handle 32-bit rollover. Word 3 is unused. No tensors
    or host state are cached.
    """
    assert eps == 1.0e-6
    m, width = local_input.shape
    assert width == 2048 and m in (1, 2, 4, 8, 16, 32, 64)
    arch = torch.cuda.get_device_properties(local_input.device).gcnArchName.split(
        ":", 1
    )[0]
    assert arch == "gfx950"
    rounding = "rtne"
    fp8_dtype = torch.float8_e4m3fn
    fp8_max = 448.0
    normalized = torch.empty_like(local_input)
    residual_out = torch.empty_like(local_input)
    quantized = torch.empty_like(local_input, dtype=fp8_dtype)
    scales = torch.empty((m, 16), dtype=torch.float32, device=local_input.device)
    reduced = torch.empty_like(local_input)
    _fused[(m // 2 if m == 64 else m,)](
        input_peer_ptrs,
        lock_peer_ptrs,
        residual,
        gemma_weight,
        normalized,
        residual_out,
        quantized,
        scales,
        reduced,
        m,
        rounding,
        fp8_max,
        eps,
        num_warps=16 if m == 64 else 8,
        enable_fp_fusion=False,
    )
    return normalized, residual_out, quantized, scales, reduced
