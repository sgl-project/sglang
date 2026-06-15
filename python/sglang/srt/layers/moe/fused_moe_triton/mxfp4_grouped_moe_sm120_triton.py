"""Expert-grouped GEMM for the SM120 MXFP4 MoE prefill path.

Drop-in alternative to mxfp4_moe_sm120_triton.py::mxfp4_moe_forward_triton (the
per-slot GEMV) for prefill-sized batches. Standard MoE block-grouped GEMM: sort
the M*topk (token, expert) slots into expert-contiguous BLOCK_M blocks, then one
fused Triton GEMM per (m-block, n-block) tile reuses each expert's weights
across its BLOCK_M tokens via tl.dot (tensor cores). Same MXFP4 E2M1 + block-32
scale dequant as the per-slot kernel; same output within bf16 accumulation.

The per-slot GEMV stays in place for decode and CUDA-graph capture: this path
sorts slots with torch.argsort, which is not graph-capturable.
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 64  # fixed: must equal the moe_align block size


@triton.jit
def _deq_fp4(nibble):
    sign = (nibble >> 3) & 1
    exp = (nibble >> 1) & 3
    man = nibble & 1
    sub = exp == 0
    mant = 1.0 + man.to(tl.float32) * 0.5
    expo = tl.math.exp2((exp - 1).to(tl.float32))
    v = tl.where(sub, man.to(tl.float32) * 0.5, mant * expo)
    return tl.where(sign != 0, -v, v)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ],
    key=["N", "K"],
)
@triton.jit
def _mxfp4_grouped_gemm_kernel(
    A_ptr,  # [num_A_rows, K] bf16
    B_packed_ptr,  # [E, N, K//2] uint8
    B_scale_ptr,  # [E, N, K//32] f32
    C_ptr,  # [num_sorted, N] bf16 (sorted order)
    sorted_ids_ptr,  # [num_sorted] int32 (slot id; >= num_valid = padding)
    expert_ids_ptr,  # [num_blocks_m] int32
    num_valid,  # int32 (M*topk)
    TOPK: tl.constexpr,
    GATHER: tl.constexpr,  # gemm1: a_row = slot//TOPK; gemm2: a_row = sorted position
    N,
    K,
    stride_am,
    stride_ak,
    eb_stride,
    es_stride,
    stride_bn,
    stride_bk2,
    stride_bsn,
    stride_bsk32,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    # offs_m is always within the padded sorted array (grid = nblocks * BM rows);
    # validity is decided by the loaded slot value (padding entries == num_valid).
    slot = tl.load(sorted_ids_ptr + offs_m)
    m_valid = slot < num_valid
    # Padding entries hold slot == num_valid (== M*TOPK), so slot // TOPK == M is
    # out of bounds for A. Clamp to a safe row before the address arithmetic; the
    # loads are masked by m_valid anyway, this only avoids the OOB pointer compute.
    safe_slot = tl.where(m_valid, slot, 0)
    a_row = (safe_slot // TOPK) if GATHER else offs_m
    expert = tl.load(expert_ids_ptr + pid_m)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_base = expert.to(tl.int64) * eb_stride
    s_base = expert.to(tl.int64) * es_stride
    acc = tl.zeros((BM, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k2 = k_start // 2 + tl.arange(0, BLOCK_K // 2)
        b_mask = (offs_n[:, None] < N) & (offs_k2[None, :] < K // 2)
        b_packed = tl.load(
            B_packed_ptr
            + b_base
            + offs_n[:, None] * stride_bn
            + offs_k2[None, :] * stride_bk2,
            mask=b_mask,
            other=0,
        ).to(tl.int32)
        val_lo = _deq_fp4(b_packed & 0x0F)
        val_hi = _deq_fp4((b_packed >> 4) & 0x0F)

        group_ids = tl.arange(0, BLOCK_K // 2) // 16
        s = tl.load(
            B_scale_ptr
            + s_base
            + offs_n[:, None] * stride_bsn
            + (k_start // 32 + group_ids[None, :]) * stride_bsk32,
            mask=(offs_n[:, None] < N)
            & ((k_start // 32 + group_ids[None, :]) < K // 32),
            other=1.0,
        )
        val_lo = val_lo * s
        val_hi = val_hi * s

        offs_ke = k_start + tl.arange(0, BLOCK_K // 2) * 2
        offs_ko = offs_ke + 1
        # Activations are bf16 already; keep them bf16 and cast the dequantized
        # weights to bf16 so tl.dot uses the SM120 bf16 tensor cores rather than
        # the fp32 path. The accumulator stays fp32, and the 4-bit MXFP4 weights
        # are exact in bf16, so the result still matches the per-slot kernel
        # within bf16 accumulation.
        a_e = tl.load(
            A_ptr + a_row[:, None] * stride_am + offs_ke[None, :] * stride_ak,
            mask=m_valid[:, None] & (offs_ke[None, :] < K),
            other=0.0,
        ).to(tl.bfloat16)
        a_o = tl.load(
            A_ptr + a_row[:, None] * stride_am + offs_ko[None, :] * stride_ak,
            mask=m_valid[:, None] & (offs_ko[None, :] < K),
            other=0.0,
        ).to(tl.bfloat16)
        acc += tl.dot(a_e, tl.trans(val_lo.to(tl.bfloat16)))
        acc += tl.dot(a_o, tl.trans(val_hi.to(tl.bfloat16)))

    c_mask = m_valid[:, None] & (offs_n[None, :] < N)
    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(tl.bfloat16),
        mask=c_mask,
    )


def moe_align(topk_ids, E, block_m=BLOCK_M):
    """Return (sorted_ids [num_sorted], expert_ids [num_blocks], num_valid).
    sorted_ids holds slot indices grouped by expert, each expert padded up to a
    block_m multiple; padding entries = num_valid (sentinel).

    Slots whose expert id is negative are dropped from the grouping. topk_ids
    carries -1 for padded or off-rank tokens under expert parallelism (the same
    invalid slots the per-slot GEMV masks out), so they must stay out of the
    bincount/sort and the down-projection scatter; their output rows remain zero.
    This runs in prefill only (the caller skips the grouped path during CUDA
    graph capture), so the host syncs below are acceptable."""
    dev = topk_ids.device
    flat = topk_ids.reshape(-1).to(torch.int64)  # [ns] expert per slot
    ns = flat.numel()
    valid_slots = torch.nonzero(flat >= 0, as_tuple=True)[
        0
    ]  # original slot ids, valid only
    slot_e = flat[valid_slots]  # expert per valid slot
    order = torch.argsort(slot_e, stable=True)  # valid slots grouped by expert
    sorted_slots = valid_slots[order]
    sorted_e = slot_e[order]
    counts = torch.bincount(slot_e, minlength=E)  # [E], valid slots only
    blocks = (counts + block_m - 1) // block_m
    padded = blocks * block_m
    out_off = torch.cumsum(padded, 0) - padded  # padded start per expert
    in_off = torch.cumsum(counts, 0) - counts  # compact start per expert
    total = int(padded.sum().item())
    sorted_ids = torch.full((total,), ns, dtype=torch.int32, device=dev)
    ranks = torch.arange(sorted_slots.numel(), device=dev) - in_off[sorted_e]
    dest = out_off[sorted_e] + ranks
    sorted_ids[dest] = sorted_slots.to(torch.int32)
    expert_ids = torch.repeat_interleave(torch.arange(E, device=dev), blocks).to(
        torch.int32
    )
    return sorted_ids, expert_ids, ns


def mxfp4_moe_forward_grouped(
    hidden,
    w13p,
    w2p,
    w13s,
    w2s,
    topk_ids,
    topk_w,
    hidden_size,
    intermediate_size,
    inplace=False,
    routed_scaling_factor=None,
    clamp_limit=None,
):
    """Signature matches mxfp4_moe_forward_triton (the slot kernel) so it is a
    drop-in for the prefill branch. `inplace` is ignored (returns a new tensor,
    as the slot path does)."""
    import torch.nn.functional as F

    M, K = hidden.shape
    I = intermediate_size
    E = w13p.shape[0]
    topk = topk_ids.shape[1]
    dev = hidden.device
    w13u = w13p.view(torch.uint8)
    w2u = w2p.view(torch.uint8)
    w13s = w13s.float() if w13s.dtype != torch.float32 else w13s
    w2s = w2s.float() if w2s.dtype != torch.float32 else w2s

    sorted_ids, expert_ids, num_valid = moe_align(topk_ids, E)
    total = sorted_ids.shape[0]
    if total == 0:
        # Every slot was dropped (e.g. all tokens off-rank under expert
        # parallelism). There are no expert blocks, so the grouped GEMMs would
        # launch with a grid dimension of 0 (an invalid launch). Return zeros,
        # matching the per-slot kernel's all-masked output.
        return torch.zeros(M, K, dtype=hidden.dtype, device=dev)
    nblocks = expert_ids.shape[0]

    # GEMM1: gate_up [total, 2I] (sorted), gather A from hidden via slot//topk
    inter = torch.empty(total, 2 * I, dtype=torch.bfloat16, device=dev)
    grid1 = lambda m: (nblocks, triton.cdiv(2 * I, m["BLOCK_N"]))
    _mxfp4_grouped_gemm_kernel[grid1](
        hidden,
        w13u,
        w13s,
        inter,
        sorted_ids,
        expert_ids,
        num_valid,
        topk,
        True,
        2 * I,
        K,
        hidden.stride(0),
        hidden.stride(1),
        w13u.stride(0),
        w13s.stride(0),
        w13u.stride(1),
        w13u.stride(2),
        w13s.stride(1),
        w13s.stride(2),
        inter.stride(0),
        inter.stride(1),
        BM=BLOCK_M,
    )
    gate = inter[:, :I].float()
    up = inter[:, I:].float()
    if clamp_limit is not None and clamp_limit > 0:
        gate = torch.clamp(gate, max=clamp_limit)
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    act = (F.silu(gate) * up).to(torch.bfloat16)

    # GEMM2: down [total, K] (sorted), A = act (identity row = sorted position)
    down_s = torch.empty(total, K, dtype=torch.bfloat16, device=dev)
    grid2 = lambda m: (nblocks, triton.cdiv(K, m["BLOCK_N"]))
    _mxfp4_grouped_gemm_kernel[grid2](
        act,
        w2u,
        w2s,
        down_s,
        sorted_ids,
        expert_ids,
        num_valid,
        topk,
        False,
        K,
        I,
        act.stride(0),
        act.stride(1),
        w2u.stride(0),
        w2s.stride(0),
        w2u.stride(1),
        w2u.stride(2),
        w2s.stride(1),
        w2s.stride(2),
        down_s.stride(0),
        down_s.stride(1),
        BM=BLOCK_M,
    )

    # Unsort + weighted reduce over topk
    down = torch.zeros(num_valid, K, dtype=torch.bfloat16, device=dev)
    valid = sorted_ids < num_valid
    down[sorted_ids[valid].long()] = down_s[valid]
    out = (down * topk_w.reshape(-1, 1).to(torch.bfloat16)).view(M, topk, K).sum(1)
    if routed_scaling_factor not in (None, 1.0):
        out.mul_(routed_scaling_factor)
    return out
