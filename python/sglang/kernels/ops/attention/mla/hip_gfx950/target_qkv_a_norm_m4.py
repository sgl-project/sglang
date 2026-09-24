# Adapted from OpenAI-Partners/artemis-kernel-integrations PR #11.
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _striped_program_id(
    pid,
    N: gl.constexpr,
    BN: gl.constexpr,
    SPLITS: gl.constexpr,
    GROUP: gl.constexpr,
    STRIPES: gl.constexpr,
):
    if STRIPES > 1:
        TOTAL: gl.constexpr = gl.cdiv(N, BN * GROUP) * GROUP * SPLITS
        stripe = pid % STRIPES
        pid = (
            stripe * (TOTAL // STRIPES)
            + gl.minimum(stripe, TOTAL % STRIPES)
            + pid // STRIPES
        )
    return pid


@gluon.jit
def _copy_panel(a_slot, b_slot, X, W, a_offsets, b_offsets, k_start):
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a_slot, X, a_offsets + k_start, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b_slot, W, b_offsets + k_start, cache_modifier=".ca"
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _wait_weight_wave(DEPTH: gl.constexpr, CHUNKS: gl.constexpr, STEP: gl.constexpr):
    OUTSTANDING: gl.constexpr = 4 * min(DEPTH - 1, CHUNKS - 1 - STEP)
    gl.inline_asm_elementwise(
        f"s_waitcnt vmcnt({OUTSTANDING})\n v_mov_b32 $0, 0",
        constraints="=v,~{memory}",
        args=[],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _project_async(
    X,
    W,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PITCH: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    SPLITS: gl.constexpr,
    NW: gl.constexpr,
    STAGES: gl.constexpr,
    GROUP: gl.constexpr,
    STRIPES: gl.constexpr,
    PHASE: gl.constexpr = 1,
    MAX_PHASE: gl.constexpr = 16,
    PRELOAD_A: gl.constexpr = False,
    WAVE_B: gl.constexpr = False,
):
    gl.static_assert(BM == M and N % (BN * GROUP) == 0 and (K % (BK * SPLITS) == 0))
    pid = gl.program_id(0).to(gl.uint32)
    pid = _striped_program_id(pid, N, BN, SPLITS, GROUP, STRIPES)
    tile = pid // (GROUP * SPLITS) * GROUP + pid % GROUP
    split = pid // GROUP % SPLITS
    CHUNKS: gl.constexpr = gl.cdiv(K, BK * SPLITS)
    DEPTH: gl.constexpr = min(STAGES, CHUNKS)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=M != 16, warps_per_cta=[1, NW]
    )
    ad: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    bd: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    WORDS: gl.constexpr = M == 8
    COPY_K: gl.constexpr = BK // 2 if WORDS else BK
    COPY_VEC: gl.constexpr = 4 if WORDS else 8
    if WORDS:
        X = X.to(gl.pointer_type(gl.uint32))
        W = W.to(gl.pointer_type(gl.uint32))
    if WAVE_B:
        gl.static_assert(
            (M == 4 or M == 8) and BK == 128 and PRELOAD_A and (BN == 16 * NW)
        )
        gl.static_assert(NW == 2 or NW == 4)
        if WORDS:
            linear: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[1], [2], [256], [512]],
                lane_bases=[[4], [8], [16], [32], [64], [128]],
                warp_bases=[[1024]] if NW == 2 else [[1024], [2048]],
                block_bases=[],
                shape=[BN * COPY_K],
            )
        else:
            linear: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[1], [2], [4], [512], [1024]],
                lane_bases=[[8], [16], [32], [64], [128], [256]],
                warp_bases=[[2048]] if NW == 2 else [[2048], [4096]],
                block_bases=[],
                shape=[BN * COPY_K],
            )
    else:
        linear: gl.constexpr = gl.BlockedLayout([COPY_VEC], [64], [NW], [0])
    a_linear: gl.constexpr = gl.BlockedLayout(
        [
            COPY_VEC
            if PRELOAD_A
            else (1 if WORDS else 2)
            if BM * BK < NW * 512
            else COPY_VEC
        ],
        [64],
        [NW],
        [0],
    )
    flat_shared: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])
    a_shared: gl.constexpr = gl.SwizzledSharedLayout(8, PHASE, MAX_PHASE, [1, 0])
    b_shared: gl.constexpr = gl.SwizzledSharedLayout(8, PHASE, MAX_PHASE, [0, 1])
    A_DEPTH: gl.constexpr = CHUNKS if PRELOAD_A else DEPTH
    A_ITEMS: gl.constexpr = (CHUNKS if PRELOAD_A else 1) * BM * COPY_K
    a_storage = gl.allocate_shared_memory(
        gl.uint32 if WORDS else gl.bfloat16, [A_DEPTH, BM * COPY_K], flat_shared
    )
    if WAVE_B:
        b_slots = ()
        for stage in gl.static_range(DEPTH):
            b_slots += (
                gl.allocate_shared_memory(
                    gl.uint32 if WORDS else gl.bfloat16, [BN * COPY_K], flat_shared
                ),
            )
    else:
        b_storage = gl.allocate_shared_memory(
            gl.uint32 if WORDS else gl.bfloat16, [DEPTH, BN * COPY_K], flat_shared
        )
    ai = gl.arange(0, A_ITEMS, a_linear)
    bi = gl.arange(0, BN * COPY_K, linear)
    am = ai // COPY_K % BM if PRELOAD_A else ai // COPY_K
    bn = tile * BN + bi // COPY_K
    ak = ai % COPY_K ^ am // PHASE % MAX_PHASE * COPY_VEC
    if PRELOAD_A:
        ak += ai // (BM * COPY_K) * COPY_K
    bk = bi % COPY_K ^ bi // COPY_K // PHASE % MAX_PHASE * COPY_VEC
    ak = gl.max_contiguous(gl.multiple_of(ak, COPY_VEC), COPY_VEC)
    bk = gl.max_contiguous(gl.multiple_of(bk, COPY_VEC), COPY_VEC)
    a_offsets = am * (SX // 2 if WORDS else SX) + ak
    b_offsets = bn * (SW // 2 if WORDS else SW) + bk
    start = split * CHUNKS * COPY_K
    if PRELOAD_A:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            a_storage._reinterpret(shape=[CHUNKS * BM * COPY_K], layout=flat_shared),
            X,
            a_offsets + start,
            cache_modifier=".ca",
            mask=am < M,
            other=0,
        )
    for stage in gl.static_range(DEPTH):
        b_dest = b_slots[stage] if WAVE_B else b_storage.index(stage)
        if PRELOAD_A:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                b_dest, W, b_offsets + start + stage * COPY_K, cache_modifier=".ca"
            )
            gl.amd.cdna4.async_copy.commit_group()
        else:
            _copy_panel(
                a_storage.index(stage),
                b_dest,
                X,
                W,
                a_offsets,
                b_offsets,
                start + stage * COPY_K,
            )
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for i in gl.static_range(CHUNKS):
        a_slot = a_storage.index(i if PRELOAD_A else i % DEPTH)
        if WAVE_B and i > 0:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_slot._reinterpret(dtype=gl.bfloat16, shape=[BM, BK], layout=a_shared),
                ad,
            )
        if WAVE_B and i > 0:
            _wait_weight_wave(DEPTH, CHUNKS, i)
        else:
            gl.amd.cdna4.async_copy.wait_group(min(DEPTH - 1, CHUNKS - 1 - i))
        if NW > 1 and (not WAVE_B or i == 0):
            gl.barrier()
        b_slot = b_slots[i % DEPTH] if WAVE_B else b_storage.index(i % DEPTH)
        if not WAVE_B or i == 0:
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_slot._reinterpret(dtype=gl.bfloat16, shape=[BM, BK], layout=a_shared),
                ad,
            )
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_slot._reinterpret(dtype=gl.bfloat16, shape=[BK, BN], layout=b_shared), bd
        )
        if i + DEPTH < CHUNKS:
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                constraints="=v,~{memory}",
                args=[],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            if NW > 1 and (not WAVE_B):
                gl.barrier()
            if PRELOAD_A:
                gl.amd.cdna4.async_copy.buffer_load_to_shared(
                    b_slot,
                    W,
                    b_offsets + start + (i + DEPTH) * COPY_K,
                    cache_modifier=".ca",
                )
                gl.amd.cdna4.async_copy.commit_group()
            else:
                _copy_panel(
                    a_slot,
                    b_slot,
                    X,
                    W,
                    a_offsets,
                    b_offsets,
                    start + (i + DEPTH) * COPY_K,
                )
        acc = gl.amd.cdna4.mfma(a, b, acc)
    om = gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.amd.cdna4.buffer_store(
        ptr=P + split * M * PITCH + tile * BN,
        offsets=om[:, None] * PITCH + on[None, :],
        stored_value=acc,
        mask=(om[:, None] < M) & (tile * BN + on[None, :] < N),
        cache=".wt",
    )


@gluon.jit
def _finish_part(
    P,
    G,
    Out,
    eps,
    row,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    WIDTH: gl.constexpr,
    OFFSET: gl.constexpr,
    SPLITS: gl.constexpr,
    B: gl.constexpr,
    NORM: gl.constexpr,
    VEC: gl.constexpr,
    BUFFER: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, B, layout)
    value = gl.full((B,), 0, gl.float32, layout)
    base = P + row * PITCH + OFFSET
    for s in gl.static_range(SPLITS):
        if BUFFER:
            value += gl.amd.cdna4.buffer_load(base + s * M * PITCH, d, d < WIDTH, 0)
        else:
            value += gl.load(P + s * M * PITCH + row * PITCH + OFFSET + d, d < WIDTH, 0)
    if NORM:
        value = value.to(gl.bfloat16).to(gl.float32)
        gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
        inv_rms = gl.rsqrt(gl.sum(value * value, 0) / WIDTH + eps)
        value = value * inv_rms * gamma
    gl.store(Out + row * WIDTH + d, value, d < WIDTH)


@gluon.jit
def _finish(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    eps,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    SPLITS: gl.constexpr,
    BQ: gl.constexpr,
    BKV: gl.constexpr,
    BR: gl.constexpr,
    VEC: gl.constexpr,
    BUFFER: gl.constexpr,
):
    row = gl.program_id(0)
    if M == 16:
        row = row.to(gl.uint32)
    part = gl.program_id(1)
    if M <= 2:
        part = 2 - part
    if part == 0:
        _finish_part(P, QG, QO, eps, row, M, PITCH, Q, 0, SPLITS, BQ, True, VEC, BUFFER)
    elif part == 1:
        _finish_part(
            P, KG, KO, eps, row, M, PITCH, KV, Q, SPLITS, BKV, True, VEC, BUFFER
        )
    else:
        _finish_part(
            P, QG, RO, eps, row, M, PITCH, R, Q + KV, SPLITS, BR, False, VEC, BUFFER
        )


def mla_qkv_a_norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    q_gamma: torch.Tensor,
    kv_gamma: torch.Tensor,
    *,
    rope_dim: int = 64,
    eps: float = 1e-5,
):
    """Exact M=4 GLM-5.2 QKV-A projection and dual RMSNorm."""
    assert hidden.shape == (4, 6144) and weight.shape == (2624, 6144)
    assert q_gamma.shape == (2048,) and kv_gamma.shape == (512,)
    assert rope_dim == 64
    assert hidden.dtype == weight.dtype == torch.bfloat16
    assert q_gamma.dtype == kv_gamma.dtype == torch.bfloat16
    assert all(tensor.is_contiguous() for tensor in (hidden, weight, q_gamma, kv_gamma))

    partial = torch.empty((6, 4, 4096), device=hidden.device, dtype=torch.float32)
    q_out = torch.empty((4, 2048), device=hidden.device, dtype=torch.bfloat16)
    kv_out = torch.empty((4, 512), device=hidden.device, dtype=torch.bfloat16)
    rope_out = torch.empty((4, 64), device=hidden.device, dtype=torch.bfloat16)
    _project_async[triton.cdiv(2624, 64) * 6,](
        hidden,
        weight,
        partial,
        4,
        2624,
        6144,
        hidden.stride(0),
        weight.stride(0),
        4096,
        BM=4,
        BN=64,
        BK=128,
        SPLITS=6,
        NW=4,
        STAGES=4,
        GROUP=1,
        STRIPES=4,
        PRELOAD_A=True,
        WAVE_B=True,
        num_warps=4,
    )
    _finish[4, 3](
        partial,
        q_gamma,
        kv_gamma,
        q_out,
        kv_out,
        rope_out,
        eps,
        4,
        4096,
        2048,
        512,
        64,
        6,
        2048,
        512,
        64,
        4,
        False,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return q_out, kv_out, rope_out
