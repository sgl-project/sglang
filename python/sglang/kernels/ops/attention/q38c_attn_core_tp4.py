# Addressing/store adapted from native MiniMax sparse decode and qsa_pool_direct.
# Copyright 2025 XunhaoLai. All rights reserved.
"""TP4 FN selected-KV TCGEN05 attention and BF16-boundary gate.

Adapt the qualified TP1 partial/merge algorithm to six Q heads and one KV head.
The TP1 code is unchanged. The service caller owns shape/mode dispatch.
"""
import torch
import triton
import triton.language as tl

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
)

from sglang.kernels.ops.pdl_small import ENABLED as _PDL_SMALL

try:
    # Triton 3.7.1 API
    from triton.experimental.gluon.language.nvidia.blackwell import get_tmem_reg_layout
except ImportError:
    # Triton 3.8 moved this onto the descriptor type (triton-lang/triton#9594)
    # TODO: once ROCm leaves triton 3.7 (swapab_gluon_hip imports _load_v4
    # from here), drop the try import above and call get_reg_layout directly.
    from triton.experimental.gluon.language.nvidia.blackwell import (
        tensor_memory_descriptor_type,
    )
    from triton.runtime.jit import constexpr_function

    @constexpr_function
    def get_tmem_reg_layout(element_ty, shape, layout, num_warps):
        ty = tensor_memory_descriptor_type(element_ty, shape, layout, shape)
        return ty.get_reg_layout(num_warps=num_warps)


@triton.jit
def _store_tp4(K, V, NK, NV, LOC, KS: tl.constexpr, VS: tl.constexpr,
           NKS: tl.constexpr, NVS: tl.constexpr, H: tl.constexpr, D: tl.constexpr, PDL_SMALL: tl.constexpr = False):
    # Protect dynamic reads and reused-buffer writes before relaxing this edge.
    if PDL_SMALL:
        tl.extra.cuda.gdc_wait()
    row = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    slot = tl.load(LOC + row).to(tl.int64)
    k = tl.load(NK + row * NKS + h * D + d)
    v = tl.load(NV + row * NVS + h * D + d)
    tl.store(K + slot * KS + h * D + d, k, slot != 0)
    tl.store(V + slot * VS + h * D + d, v, slot != 0)
    if PDL_SMALL:
        tl.extra.cuda.gdc_launch_dependents()


@gluon.jit
def _partial_tp4(
    Q, K, V, NK, NV, LOC, MAP, REQ, IDX, LENGTH, QPOS,
    PART, MAX, SUM,
    QS: gl.constexpr, QH: gl.constexpr,
    KS: gl.constexpr, VS: gl.constexpr,
    NKS: gl.constexpr, NVS: gl.constexpr, MS: gl.constexpr,
    IS: gl.constexpr, NT: gl.constexpr, GROUP: gl.constexpr,
    WIDTH: gl.constexpr, SCALE: gl.constexpr, KVH: gl.constexpr,
    PDL_SMALL: gl.constexpr = False,
):
    # Fixed tiles, rather than the old 32-split online log/exp recurrence.
    # Every CTA produces one raw FP32 state for one query and one KV head.
    # Protect dynamic reads and reused-buffer writes before relaxing this edge.
    if PDL_SMALL:
        gl.inline_asm_elementwise(
            "griddepcontrol.wait; mov.u32 $0, 0;",
            constraints="=r,~{memory}", args=[], dtype=gl.int32,
            is_pure=False, pack=1,
        )
    row, tile, head = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    kv_layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    n = gl.arange(0, 64, gl.SliceLayout(1, kv_layout))
    d = gl.arange(0, 256, gl.SliceLayout(0, kv_layout))
    col = tile * 64 + n
    req = gl.load(REQ + row).to(gl.int64)
    length = gl.load(LENGTH + row)
    qpos = gl.load(QPOS + row)
    active = gl.load(LOC + row) != 0
    selected = gl.minimum(length // 4, 512) * 4
    tail_start = (qpos + 1) // 4 * 4
    tail_count = gl.maximum(0, gl.minimum(qpos + 1, length) - tail_start)
    count = gl.where(active, selected + tail_count, 0)
    block = gl.load(IDX + row * IS + col // 4, col < 2048, -1)
    expanded = block * 4 + col % 4
    tail_off = col - selected
    tail = tail_start + tail_off
    pos = gl.where(
        (col < selected) & (block >= 0), expanded,
        gl.where((tail_off >= 0) & (tail_off < 3) & (tail < qpos + 1), tail, -1),
    )
    valid = (col < count) & (pos >= 0) & (pos < length)
    slot = gl.load(MAP + req * MS + pos, valid, 0).to(gl.int64)
    first_pos = length - 1 - row % WIDTH
    relative = pos - first_pos
    fresh = valid & (relative >= 0)
    fresh_row = gl.where(fresh, row // WIDTH * WIDTH + relative, 0).to(gl.int64)
    kp = gl.where(
        fresh[:, None],
        NK + fresh_row[:, None] * NKS + head * 256 + d[None, :],
        K + slot[:, None] * KS + head * 256 + d[None, :],
    )
    k = gl.load(kp, valid[:, None], 0)
    h = gl.arange(0, 16, gl.SliceLayout(1, kv_layout))
    q = gl.load(
        Q + row * QS + (head * GROUP + h[:, None]) * QH + d[None, :],
        (h < GROUP)[:, None], 0,
    )
    q_smem = gl.allocate_shared_memory(
        gl.bfloat16, [16, 256],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16), value=q,
    )
    k_smem = gl.allocate_shared_memory(
        gl.bfloat16, [64, 256],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16), value=k,
    )
    score_tmem = allocate_tensor_memory(
        gl.float32, [64, 16], TensorMemoryLayout(block=(64, 16), col_stride=1),
    )
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    fence_async_shared()
    # [Ktile64,D256] x [D256,H16] -> FP32 [64,16] in TMEM.
    tcgen05_mma(k_smem, q_smem.permute((1, 0)), score_tmem, use_acc=False)
    tcgen05_commit(bar)
    # Gather independent V while QK is in flight. No 64-token-page TMA overread:
    # each selected block expands to exactly four token addresses.
    vp = gl.where(
        fresh[:, None],
        NV + fresh_row[:, None] * NVS + head * 256 + d[None, :],
        V + slot[:, None] * VS + head * 256 + d[None, :],
    )
    v = gl.load(vp, valid[:, None], 0)
    # QK uses the async tensor-core proxy. K/V shared storage may alias:
    # finish QK reads before generic-proxy stores initialize V's storage.
    mbarrier.wait(bar, phase=0)
    v_smem = gl.allocate_shared_memory(
        gl.bfloat16, [64, 256],
        gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16), value=v,
    )
    score_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (64, 16), TensorMemoryLayout(block=(64, 16), col_stride=1), 4,
    )
    scores = score_tmem.load(score_layout) * SCALE
    valid_scores = gl.convert_layout(valid, gl.SliceLayout(1, score_layout))
    scores = gl.where(valid_scores[:, None], scores, -float('inf'))
    mx = gl.max(scores, 0)
    safe_max = gl.where(mx == -float('inf'), 0.0, mx)
    prob = gl.exp(scores - safe_max[None, :])
    denom = gl.sum(prob, 0)
    p_hi = prob.to(gl.bfloat16)
    p_smem = gl.allocate_shared_memory(
        gl.bfloat16, [64, 16],
        gl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16), value=p_hi,
    )
    out_tmem = allocate_tensor_memory(
        gl.float32, [256, 16], TensorMemoryLayout(block=(128, 16), col_stride=1),
    )
    fence_async_shared()
    tcgen05_mma(v_smem.permute((1, 0)), p_smem, out_tmem, use_acc=False)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=1)
    # Same compensated BF16 P hi/lo decomposition as mature DSv4; no FP8.
    p_lo = (prob - p_hi.to(gl.float32)).to(gl.bfloat16)
    p_smem.store(p_lo)
    fence_async_shared()
    tcgen05_mma(v_smem.permute((1, 0)), p_smem, out_tmem, use_acc=True)
    tcgen05_commit(bar)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)
    out_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, (256, 16), TensorMemoryLayout(block=(128, 16), col_stride=1), 4,
    )
    value = out_tmem.load(out_layout)
    hd_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    value_hd = gl.convert_layout(value.permute((1, 0)), hd_layout)
    hh = gl.arange(0, 16, gl.SliceLayout(1, hd_layout))
    dd = gl.arange(0, 256, gl.SliceLayout(0, hd_layout))
    state = (row * KVH + head) * NT + tile
    gl.store(PART + (state * 16 + hh[:, None]) * 256 + dd[None, :],
             value_hd, (hh < GROUP)[:, None])
    stat_layout: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    hs = gl.arange(0, 16, stat_layout)
    gl.store(MAX + state * 16 + hs, gl.convert_layout(mx, stat_layout), hs < GROUP)
    gl.store(SUM + state * 16 + hs, gl.convert_layout(denom, stat_layout), hs < GROUP)
    if PDL_SMALL:
        gl.inline_asm_elementwise(
            "griddepcontrol.launch_dependents; mov.u32 $0, 0;",
            constraints="=r,~{memory}", args=[], dtype=gl.int32,
            is_pure=False, pack=1,
        )


@triton.jit
def _merge_gate_tp4(P, MAX, SUM, OUT, GATE, GS: tl.constexpr, GH: tl.constexpr,
                    NT: tl.constexpr, ST: tl.constexpr,
                    QHEADS: tl.constexpr, KVH: tl.constexpr, GROUP: tl.constexpr):
    # Only independent o_proj weight DMA may run before our output is complete.
    tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    row, head, dim_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = tl.arange(0, ST)
    d = dim_tile * 64 + tl.arange(0, 64)
    state = (row * KVH + head // GROUP) * NT + t
    hh = head % GROUP
    mx = tl.load(MAX + state * 16 + hh, t < NT, -float('inf'))
    denom = tl.load(SUM + state * 16 + hh, t < NT, 0.)
    maximum = tl.max(mx, 0)
    safe_max = tl.where(maximum > -float('inf'), maximum, 0.)
    weights = tl.exp(mx - safe_max)
    total = tl.sum(denom * weights, 0)
    partial = tl.load(P + (state[:, None] * 16 + hh) * 256 + d[None, :],
                      (t < NT)[:, None], 0.)
    value = tl.sum(partial * weights[:, None], 0) / tl.where(total > 0, total, 1.)
    # Preserve the native BF16 attention-output boundary before the gate.
    rounded = value.to(tl.bfloat16).to(tl.float32)
    gate = tl.load(GATE + row * GS + head * GH + d).to(tl.float32)
    tl.store(OUT + (row * QHEADS + head) * 256 + d, rounded * tl.sigmoid(gate))


def run_gated(q, k_pool, v_pool, k_new, v_new, loc, req_to_token, row_requests,
              block_indices, lengths, qpos, scale, gate, width=4):
    """TARGET_VERIFY M4/Q6/KV1 only; retain current-KV store and fresh/tail ABI."""
    m, qheads, dim = q.shape
    kvheads = k_pool.shape[1]
    assert (m, qheads, kvheads, dim, width) == (4, 6, 1, 256, 4)
    assert block_indices.shape == (m, 512) and block_indices.dtype == torch.int32
    assert all(t.dtype == torch.bfloat16 for t in (q, k_pool, v_pool, k_new, v_new, gate))
    assert all(t.stride(-1) == 1 and t.stride(-2) == dim
               for t in (q, k_pool, v_pool, k_new, v_new))
    assert tuple(gate.shape) in ((4, 6, 256), (4, 1536)) and gate.stride(-1) == 1
    assert block_indices.stride(1) == 1
    nt = triton.cdiv(512 * 4 + 3, 64)
    partial = torch.empty((m, kvheads, nt, 16, dim), dtype=torch.float32, device=q.device)
    maximum = torch.empty((m, kvheads, nt, 16), dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maximum)
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _store_tp4[(m, kvheads)](
        k_pool, v_pool, k_new, v_new, loc,
        k_pool.stride(0), v_pool.stride(0), k_new.stride(0), v_new.stride(0), kvheads, dim,
        num_warps=4,
        launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
    )
    _partial_tp4[(m, nt, kvheads)](
        q, k_pool, v_pool, k_new, v_new, loc, req_to_token, row_requests,
        block_indices, lengths, qpos, partial, maximum, sums,
        QS=q.stride(0), QH=q.stride(1), KS=k_pool.stride(0), VS=v_pool.stride(0),
        NKS=k_new.stride(0), NVS=v_new.stride(0), MS=req_to_token.stride(0),
        IS=block_indices.stride(0), NT=nt, GROUP=qheads // kvheads, WIDTH=width,
        SCALE=float(scale), KVH=kvheads, num_warps=4,
        launch_pdl=_PDL_SMALL,
        PDL_SMALL=_PDL_SMALL,
    )
    _merge_gate_tp4[(m, qheads, triton.cdiv(dim, 64))](
        partial, maximum, sums, output, gate, gate.stride(0),
        gate.stride(1) if gate.ndim == 3 else 256,
        nt, triton.next_power_of_2(nt), qheads, kvheads, qheads // kvheads,
        num_warps=4, launch_pdl=True,
    )
    return output
