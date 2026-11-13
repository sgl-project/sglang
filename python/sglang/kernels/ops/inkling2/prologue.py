import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit

PDL = is_arch_support_pdl()
REL_WORKERS = 4


@triton.jit
def _rsqrt_approx(x):
    return tl.inline_asm_elementwise(
        "rsqrt.approx.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _lane_sum8(acc, ROWS: tl.constexpr):
    # Butterfly over 8 lanes with xor offsets 4, 2, 1.
    acc = tl.sum(tl.reshape(acc, (ROWS, 2, 4)), 1)
    acc = tl.sum(tl.reshape(acc, (ROWS, 2, 2)), 1)
    return tl.sum(acc, 1)


@triton.jit
def _lane_sum16(acc, ROWS: tl.constexpr):
    # Butterfly over 16 lanes with xor offsets 8, 4, 2, 1.
    acc = tl.sum(tl.reshape(acc, (ROWS, 2, 8)), 1)
    return _lane_sum8(acc, ROWS)


@triton.jit
def _l2_clip(
    X,
    Scale,
    Y,
    tau,
    threshold,
    eps,
    ROWS: tl.constexpr,
    ROWS_P: tl.constexpr,
    D: tl.constexpr,
    SCALED: tl.constexpr,
    TAU: tl.constexpr,
    GROUP: tl.constexpr = 0,
    PADDED_GROUP: tl.constexpr = 0,
):
    """qk_l2_clip_fwd_cutedsl on ROWS rows of D (128 or 16) contiguous elements:
    8 lanes per row, lane t owning [8t, 8t + 8) and, for D = 128, [64 + 8t, 64 + 8t + 8).
    Returns a (next_power_of_2(ROWS), D) tensor."""
    rows = tl.arange(0, ROWS_P)
    row_mask = rows < ROWS
    if PADDED_GROUP:
        # Rows are padded-group output heads; padding heads read as zero.
        row_mask = row_mask & (rows % PADDED_GROUP < GROUP)
        rows = (rows // PADDED_GROUP) * GROUP + rows % PADDED_GROUP
    lanes = tl.arange(0, 8)
    acc = tl.zeros((ROWS_P, 8), tl.float32)
    for half in tl.static_range(D // 64 if D >= 64 else 1):
        for j in tl.static_range(8):
            cols = half * 64 + lanes * 8 + j
            mask = row_mask[:, None] & (cols[None, :] < D)
            x = tl.load(X + rows[:, None] * D + cols[None, :], mask, other=0).to(
                tl.float32
            )
            if SCALED:
                x = (
                    tl.load(Scale + rows[:, None] * D + cols[None, :], mask, other=0)
                    * x
                )
            acc = tl.fma(x, x, acc)
    total = _lane_sum8(acc, ROWS_P)
    norm = tl.where(total > 0.0, total * _rsqrt_approx(total), 0.0)
    coeff = tl.div_rn(threshold, eps + norm)
    coeff = tl.where(coeff >= 1.0, 1.0, coeff)
    cols = tl.arange(0, D)
    mask = row_mask[:, None] & (cols[None, :] < D)
    x = tl.load(X + rows[:, None] * D + cols[None, :], mask, other=0).to(tl.float32)
    if SCALED:
        x = tl.load(Scale + rows[:, None] * D + cols[None, :], mask, other=0) * x
    y = x * coeff[:, None]
    if TAU:
        y = y.to(Y.dtype.element_ty).to(tl.float32) * tau
    return y


@triton.jit
def _rms_norm128(X, eps, ROWS: tl.constexpr):
    """RMSNorm without weight on ROWS rows of 128: 16 lanes per row, lane t owning
    [4t, 4t + 4) and [64 + 4t, 64 + 4t + 4)."""
    rows = tl.arange(0, ROWS)
    lanes = tl.arange(0, 16)
    acc = tl.zeros((ROWS, 16), tl.float32)
    for half in tl.static_range(2):
        for j in tl.static_range(4):
            cols = half * 64 + lanes * 4 + j
            x = tl.load(X + rows[:, None] * 128 + cols[None, :]).to(tl.float32)
            acc = tl.fma(x, x, acc)
    total = _lane_sum16(acc, ROWS)
    rstd = _rsqrt_approx(tl.fma(total, 0.0078125, eps))
    cols = tl.arange(0, 128)
    x = tl.load(X + rows[:, None] * 128 + cols[None, :]).to(tl.float32)
    return x * rstd[:, None]


@triton.jit
def _dynamic_weight(Dynamic, row, col, lag, dyn_rows, D, W, HEAD_DIM, GROUPS):
    """Row row's conv weight offset at lag; with GROUPS, Dynamic holds the
    projection's (GROUPS, rows, n) split-K partials, summed in group order and
    rounded to bfloat16 like _linear_k_group_sum."""
    n = (col // HEAD_DIM) * W + lag
    N: tl.constexpr = D // HEAD_DIM * W
    if GROUPS:
        total = tl.load(Dynamic + row * N + n)
        for group in tl.static_range(1, GROUPS):
            total = total + tl.load(Dynamic + (group * dyn_rows + row) * N + n)
        return total.to(tl.bfloat16).to(tl.float32)
    return tl.load(Dynamic + row * N + n).to(tl.float32)


@triton.jit
def _decode_sconv(
    X,
    Weight,
    Dynamic,
    Cache,
    Y,
    TrackMask,
    TrackIndices,
    row,
    slot,
    has_cache,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SC0: tl.constexpr,
    SC1: tl.constexpr,
    SC2: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DYNAMIC: tl.constexpr,
    TRACK: tl.constexpr,
    SCALE: tl.constexpr,
    dyn_rows,
    DYN_GROUPS: tl.constexpr,
):
    """The decode branch of _sconv_kernel (residual on) for one row of width D."""
    col = tl.arange(0, D)
    current = tl.load(X + col).to(tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    for tap in tl.static_range(W):
        lag = tap if DYNAMIC else W - 1 - tap
        if lag == 0:
            value = current
        else:
            value = tl.load(
                Cache + slot * SC0 + (W - 1 - lag) * SC1 + col * SC2,
                has_cache,
                other=0,
            ).to(tl.float32)
        if DYNAMIC:
            dyn = _dynamic_weight(
                Dynamic, row, col, lag, dyn_rows, D, W, HEAD_DIM, DYN_GROUPS
            )
            weight = tl.load(Weight + lag * SW0 + col * SW1).to(tl.float32) + dyn
        else:
            weight = (
                tl.load(Weight + col * SW0 + tap * SW1)
                .to(X.dtype.element_ty)
                .to(tl.float32)
            )
        if DYNAMIC and tap == 0:
            acc = value * weight
        else:
            acc = tl.fma(value, weight, acc)
    acc = acc + current
    acc = acc * SCALE
    tl.store(Y + col, acc)
    for tap in tl.static_range(W - 1):
        if tap == W - 2:
            value = current
        else:
            value = tl.load(
                Cache + slot * SC0 + (tap + 1) * SC1 + col * SC2, has_cache, other=0
            )
        tl.store(Cache + slot * SC0 + tap * SC1 + col * SC2, value, has_cache)
        if TRACK:
            track = tl.load(TrackMask + row)
            dest = tl.load(TrackIndices + row).to(tl.int64)
            tl.store(
                Cache + dest * SC0 + tap * SC1 + col * SC2, value, has_cache & track
            )


@triton.jit
def _verify_sconv(
    X,
    Weight,
    Dynamic,
    Cache,
    Y,
    Window,
    row,
    seq,
    bos,
    slot,
    has_cache,
    in_pool,
    SX: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SC0: tl.constexpr,
    SC1: tl.constexpr,
    SC2: tl.constexpr,
    SWB: tl.constexpr,
    SWT: tl.constexpr,
    SWW: tl.constexpr,
    D: tl.constexpr,
    W: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DYNAMIC: tl.constexpr,
    SCALE: tl.constexpr,
    dyn_rows,
    DYN_GROUPS: tl.constexpr,
):
    """The extend branch of _sconv_kernel (residual on) for one verify row of
    width D, plus save_intermediate_conv_windows' window for this row."""
    col = tl.arange(0, D)
    current = tl.load(X + row * SX + col).to(tl.float32)
    acc = tl.full((D,), 0, tl.float32)
    for tap in tl.static_range(W):
        lag = tap if DYNAMIC else W - 1 - tap
        src = row - lag
        from_x = src >= bos
        value = tl.load(X + src * SX + col, from_x, other=0).to(tl.float32)
        prefix = tl.load(
            Cache + slot * SC0 + (src - bos + W - 1) * SC1 + col * SC2,
            ~from_x & has_cache,
            other=0,
        ).to(tl.float32)
        value = tl.where(from_x, value, prefix)
        if DYNAMIC:
            dyn = _dynamic_weight(
                Dynamic, row, col, lag, dyn_rows, D, W, HEAD_DIM, DYN_GROUPS
            )
            weight = tl.load(Weight + lag * SW0 + col * SW1).to(tl.float32) + dyn
        else:
            weight = (
                tl.load(Weight + col * SW0 + tap * SW1)
                .to(X.dtype.element_ty)
                .to(tl.float32)
            )
        if DYNAMIC and tap == 0:
            acc = value * weight
        else:
            acc = tl.fma(value, weight, acc)
        if lag < W - 1:
            raw = tl.load(X + src * SX + col, from_x & in_pool, other=0)
            raw_prefix = tl.load(
                Cache + slot * SC0 + (src - bos + W - 1) * SC1 + col * SC2,
                ~from_x & in_pool,
                other=0,
            )
            tl.store(
                Window + seq * SWB + (row - bos) * SWT + (W - 2 - lag) * SWW + col,
                tl.where(from_x, raw, raw_prefix),
                in_pool,
            )
    acc = acc + current
    acc = acc * SCALE
    tl.store(Y + col, acc)


@triton.jit
def _attention_prologue(
    QX,
    KX,
    VX,
    RX,
    KWeight,
    VWeight,
    KDynamic,
    VDynamic,
    KCache,
    VCache,
    Indices,
    Mask,
    TrackMask,
    TrackIndices,
    Positions,
    QScale,
    RScale,
    KConv,
    VConv,
    Q,
    K,
    V,
    Rel,
    RelProj,
    SeqLens,
    Cu,
    Si,
    KWindow,
    VWindow,
    qk_threshold,
    rel_threshold,
    clip_eps,
    norm_eps,
    SP: tl.constexpr,
    SW0: tl.constexpr,
    SW1: tl.constexpr,
    SC0: tl.constexpr,
    SC1: tl.constexpr,
    SC2: tl.constexpr,
    SWB: tl.constexpr,
    SWT: tl.constexpr,
    SWW: tl.constexpr,
    HEADS: tl.constexpr,
    HEADS_P: tl.constexpr,
    KV_HEADS: tl.constexpr,
    PADDED_GROUP: tl.constexpr,
    D_REL: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    REL_PADDED: tl.constexpr,
    OUT_HEADS: tl.constexpr,
    OUT_HEADS_P: tl.constexpr,
    LEFT_PAD: tl.constexpr,
    REL_WORKERS: tl.constexpr,
    COMPACT: tl.constexpr,
    W: tl.constexpr,
    DYNAMIC: tl.constexpr,
    TRACK: tl.constexpr,
    TAU: tl.constexpr,
    TAU_FLOOR: tl.constexpr,
    TAU_ALPHA: tl.constexpr,
    VERIFY: tl.constexpr = False,
    dyn_rows=0,
    DYN_GROUPS: tl.constexpr = 0,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    row = tl.program_id(0).to(tl.int64)
    part = tl.program_id(1)
    tau = 1.0
    if TAU:
        ratio = tl.maximum(
            (tl.load(Positions + row) + 1).to(tl.float32) * (1.0 / TAU_FLOOR), 1.0
        )
        tau = 1.0 + TAU_ALPHA * libdevice.log(ratio)
    KV_WIDTH: tl.constexpr = KV_HEADS * 128
    SCALE: tl.constexpr = (W * (2 if DYNAMIC else 1) + 1) ** -0.5
    if part == 0:
        if VERIFY:
            seq = tl.load(Si + row).to(tl.int64)
            bos = tl.load(Cu + seq).to(tl.int64)
            slot = tl.load(Indices + seq).to(tl.int64)
            in_pool = tl.load(Mask + seq) != 0
            # CUDA-graph padding rows share the reserved slot 0 and must not read it.
            has_cache = in_pool & (slot > 0)
            _verify_sconv(
                KX,
                KWeight,
                KDynamic,
                KCache,
                KConv + row * KV_WIDTH,
                KWindow,
                row,
                seq,
                bos,
                slot,
                has_cache,
                in_pool,
                SP,
                SW0,
                SW1,
                SC0,
                SC1,
                SC2,
                SWB,
                SWT,
                SWW,
                KV_WIDTH,
                W,
                128,
                DYNAMIC,
                SCALE,
                dyn_rows,
                DYN_GROUPS,
            )
            _verify_sconv(
                VX,
                VWeight,
                VDynamic,
                VCache,
                VConv + row * KV_WIDTH,
                VWindow,
                row,
                seq,
                bos,
                slot,
                has_cache,
                in_pool,
                SP,
                SW0,
                SW1,
                SC0,
                SC1,
                SC2,
                SWB,
                SWT,
                SWW,
                KV_WIDTH,
                W,
                128,
                DYNAMIC,
                SCALE,
                dyn_rows,
                DYN_GROUPS,
            )
        else:
            slot = tl.load(Indices + row).to(tl.int64)
            # CUDA-graph padding rows share the reserved slot 0 and must not touch it.
            has_cache = (tl.load(Mask + row) != 0) & (slot > 0)
            _decode_sconv(
                KX + row * SP,
                KWeight,
                KDynamic,
                KCache,
                KConv + row * KV_WIDTH,
                TrackMask,
                TrackIndices,
                row,
                slot,
                has_cache,
                SW0,
                SW1,
                SC0,
                SC1,
                SC2,
                KV_WIDTH,
                W,
                128,
                DYNAMIC,
                TRACK,
                SCALE,
                dyn_rows,
                DYN_GROUPS,
            )
            _decode_sconv(
                VX + row * SP,
                VWeight,
                VDynamic,
                VCache,
                VConv + row * KV_WIDTH,
                TrackMask,
                TrackIndices,
                row,
                slot,
                has_cache,
                SW0,
                SW1,
                SC0,
                SC1,
                SC2,
                KV_WIDTH,
                W,
                128,
                DYNAMIC,
                TRACK,
                SCALE,
                dyn_rows,
                DYN_GROUPS,
            )
        tl.debug_barrier()
        heads_kv = tl.arange(0, KV_HEADS)
        cols_kv = tl.arange(0, 128)
        k = _l2_clip(
            KConv + row * KV_WIDTH,
            None,
            K,
            1.0,
            qk_threshold,
            clip_eps,
            KV_HEADS,
            KV_HEADS,
            128,
            False,
            False,
        )
        tl.store(K + row * KV_WIDTH + heads_kv[:, None] * 128 + cols_kv[None, :], k)
        v = _rms_norm128(VConv + row * KV_WIDTH, norm_eps, KV_HEADS)
        tl.store(V + row * KV_WIDTH + heads_kv[:, None] * 128 + cols_kv[None, :], v)
    elif part == 1:
        heads_q = tl.arange(0, HEADS_P)
        cols_q = tl.arange(0, 128)
        q = _l2_clip(
            QX + row * SP,
            QScale,
            Q,
            tau,
            qk_threshold,
            clip_eps,
            HEADS,
            HEADS_P,
            128,
            True,
            TAU,
        )
        group: tl.constexpr = HEADS // KV_HEADS
        out_heads = heads_q + (heads_q // group) * (PADDED_GROUP - group)
        q_width: tl.constexpr = HEADS // group * PADDED_GROUP * 128
        tl.store(
            Q + row * q_width + out_heads[:, None] * 128 + cols_q[None, :],
            q,
            (heads_q < HEADS)[:, None],
        )
        if PADDED_GROUP > group:
            pad = tl.arange(0, KV_HEADS * (PADDED_GROUP - group))
            pad_heads = (
                (pad // (PADDED_GROUP - group)) * PADDED_GROUP
                + group
                + pad % (PADDED_GROUP - group)
            )
            tl.store(
                Q + row * q_width + pad_heads[:, None] * 128 + cols_q[None, :],
                tl.zeros((KV_HEADS * (PADDED_GROUP - group), 128), tl.float32),
            )
    else:
        heads_r = tl.arange(0, OUT_HEADS_P)
        cols_r = tl.arange(0, D_REL)
        r = _l2_clip(
            RX + row * SP,
            RScale,
            Rel,
            tau,
            rel_threshold,
            clip_eps,
            OUT_HEADS,
            OUT_HEADS_P,
            D_REL,
            True,
            TAU,
            HEADS // KV_HEADS,
            PADDED_GROUP,
        )
        # rel_logits_proj: the K = 16 dot matches the einsum's GEMM bitwise.
        r = r.to(Rel.dtype.element_ty)
        if COMPACT:
            right = 0
            left = 0
        else:
            # ShearingBias's BlockInfo bounds for this verify token; the
            # sequence's packed-Q rows fit one 128-row attention block.
            seq = tl.load(Si + row).to(tl.int32)
            bos = tl.load(Cu + seq).to(tl.int32)
            qlen = tl.load(Cu + seq + 1).to(tl.int32) - bos
            seqlen = tl.load(SeqLens + seq).to(tl.int32)
            n_right = row.to(tl.int32) - bos + 1 + seqlen - qlen
            vals = tl.minimum(n_right, REL_EXTENT)
            n_block_max = tl.minimum(
                (seqlen + 127) // 128,
                ((128 + PADDED_GROUP - 1) // PADDED_GROUP + seqlen - qlen + 127) // 128,
            )
            right = tl.where(
                vals > 0,
                tl.maximum(REL_PADDED // 128 - (n_block_max - (n_right - 1) // 128), 0)
                * 128
                + (n_right - 1) % 128
                + 1,
                0,
            )
            left = tl.maximum(right - vals, 0)
        rel_row = Rel + row * OUT_HEADS * REL_PADDED + heads_r[:, None] * REL_PADDED
        head_mask = (heads_r < OUT_HEADS)[:, None]
        worker = part - 2
        steps: tl.constexpr = (REL_EXTENT // 128 + REL_WORKERS - 1) // REL_WORKERS
        for step in tl.static_range(steps):
            chunk = worker * steps + step
            ext = chunk * 128 + tl.arange(0, 128)
            in_extent = (ext < REL_EXTENT)[None, :]
            proj = tl.load(
                RelProj + cols_r[:, None] * REL_EXTENT + ext[None, :],
                in_extent,
                other=0,
            )
            rel = tl.dot(r, proj)
            if COMPACT:
                tl.store(rel_row + ext[None, :], rel, head_mask & in_extent)
            else:
                tl.store(
                    rel_row + (right - 1 - ext)[None, :],
                    rel,
                    head_mask & in_extent & (right - 1 - ext >= left)[None, :],
                )
        if not COMPACT:
            for step in tl.static_range(
                (REL_PADDED // 128 + REL_WORKERS - 1) // REL_WORKERS
            ):
                col = (step * REL_WORKERS + worker) * 128 + tl.arange(0, 128)
                fill = tl.where(col >= right, float("-inf"), LEFT_PAD)
                tl.store(
                    rel_row + col[None, :],
                    tl.zeros((OUT_HEADS_P, 128), tl.float32) + fill[None, :],
                    head_mask
                    & ((col >= right) | (col < left))[None, :]
                    & (col < REL_PADDED)[None, :],
                )


_CUDA_TYPES = {torch.float32: "float", torch.bfloat16: "__nv_bfloat16"}


@cache_once
def _prologue_module(
    heads,
    kv_heads,
    padded_group,
    rel_extent,
    width,
    dynamic,
    tau,
    decode,
    weight_dtype,
    scale_dtype,
):
    flags = (heads, kv_heads, padded_group, rel_extent, width, dynamic, tau, decode)
    args = ", ".join(str(v).lower() if isinstance(v, bool) else str(v) for v in flags)
    return load_jit(
        "inkling2_prologue",
        *(str(v) for v in flags),
        str(weight_dtype).split(".")[-1],
        str(scale_dtype).split(".")[-1],
        cuda_files=["inkling2/prologue.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"inkling2_prologue::run<{args}, {_CUDA_TYPES[weight_dtype]}, "
                f"{_CUDA_TYPES[scale_dtype]}>",
            )
        ],
    )


def _cuda_prologue(
    q_in, k_in, v_in, r_in, heads, kv_heads, d_rel, padded_group, k_sconv, v_sconv,
    k_cache, v_cache, metadata, q_scale, r_scale, rel_proj, outputs, windows,
    k_dynamic, v_dynamic, positions, tau_floor, tau_alpha, thresholds, eps, track,
    shear=None,
):  # fmt: skip
    """The compact-rel _attention_prologue in CUDA, VERIFY when windows is given
    (else decode, with track = (track_mask, track_indices) or None); False when
    unsupported. With shear = (seqlens_k, window_left, left_pad), verify writes
    the rel rows sheared as FA4's ShearingBias would."""
    q, k, v, rel = outputs
    decode = windows is None
    k_window, v_window = (k_cache, v_cache) if decode else windows
    track_mask, track_indices = track or (None, None)
    dynamic = k_dynamic is not None
    width = k_sconv.shape[0] if dynamic else k_sconv.shape[1]
    tokens = q_in.shape[0]
    tensors = (q_in, k_in, v_in, r_in, k_cache, v_cache, k_window, v_window)
    if not (
        heads * 8 <= 128
        and kv_heads in (1, 2)
        and heads <= kv_heads * padded_group <= 16
        and heads % kv_heads == 0
        and d_rel == 16
        and width == 4
        and rel_proj.shape[1] % 128 == 0
        and all(t.dtype == torch.bfloat16 for t in (*tensors, rel_proj, *outputs))
        and k_sconv.dtype in _CUDA_TYPES
        and k_sconv.dtype == v_sconv.dtype
        and k_cache.stride(2) == 1
        and k_cache.stride() == v_cache.stride()
        and (
            decode
            or k_window.stride(3) == 1
            and k_window.stride() == v_window.stride()
            and all(stride % 2 == 0 for stride in k_window.stride()[:3])
        )
        and rel_proj.is_contiguous()
        and q_scale.dtype == r_scale.dtype
        and q_scale.dtype in _CUDA_TYPES
        and q_scale.is_contiguous()
        and r_scale.is_contiguous()
        and all(t.is_contiguous() for t in outputs)
        and q_in.stride(0) % 8 == 0
        and all(
            t.data_ptr() % 16 == 0
            for t in (q_in, r_in, q_scale, r_scale, rel_proj, *outputs)
        )
        and all(t.data_ptr() % 4 == 0 for t in (k_in, v_in, k_cache, v_cache))
        and all(t.data_ptr() % 4 == 0 for t in (k_window, v_window))
        and k_cache.stride(0) % 2 == k_cache.stride(1) % 2 == 0
        and metadata["safe_idx"].dtype == torch.int64
        and metadata["cache_mask"].dtype == torch.bool
        and (
            decode
            or metadata["cu"].dtype == torch.int64
            and metadata["si"].dtype == torch.int32
        )
        and (
            track_mask is None
            or track_mask.dtype == torch.bool
            and track_indices.dtype == torch.int64
            and track_mask.is_contiguous()
            and track_indices.is_contiguous()
        )
        and (positions is None or positions.dtype == torch.int64)
        and (
            shear is None or shear[0].dtype == torch.int32 and shear[0].is_contiguous()
        )
        and (
            not dynamic
            or (k_dynamic.dtype in (torch.float32, torch.bfloat16))
            and k_dynamic.is_contiguous()
            and v_dynamic.is_contiguous()
            and k_dynamic.dtype == v_dynamic.dtype
        )
    ):
        return False
    dyn_groups = k_dynamic.shape[0] if dynamic and k_dynamic.ndim == 3 else 0
    ptr = lambda t: 0 if t is None else t.data_ptr()  # noqa: E731
    _prologue_module(
        heads,
        kv_heads,
        padded_group,
        rel_proj.shape[1],
        width,
        dynamic,
        positions is not None,
        decode,
        k_sconv.dtype,
        q_scale.dtype,
    ).run(
        q_in,
        k_in.data_ptr(),
        v_in.data_ptr(),
        r_in.data_ptr(),
        q_in.stride(0),
        k_sconv.data_ptr(),
        v_sconv.data_ptr(),
        *k_sconv.stride(),
        ptr(k_dynamic),
        ptr(v_dynamic),
        dyn_groups,
        k_cache.data_ptr(),
        v_cache.data_ptr(),
        *k_cache.stride()[:2],
        metadata["safe_idx"].data_ptr(),
        metadata["cache_mask"].data_ptr(),
        ptr(track_mask),
        ptr(track_indices),
        0 if decode else metadata["cu"].data_ptr(),
        0 if decode else metadata["si"].data_ptr(),
        ptr(positions),
        q_scale.data_ptr(),
        r_scale.data_ptr(),
        rel_proj.data_ptr(),
        q.data_ptr(),
        k.data_ptr(),
        v.data_ptr(),
        rel.data_ptr(),
        0 if decode else k_window.data_ptr(),
        0 if decode else v_window.data_ptr(),
        *((0, 0, 0) if decode else k_window.stride()[:3]),
        tokens,
        *thresholds,
        *eps,
        1.0 / (tau_floor or 1),
        tau_alpha or 0.0,
        *((0, -1, 0.0) if shear is None else (shear[0].data_ptr(), *shear[1:])),
    )
    return True


def attention_prologue(
    q_in,
    k_in,
    v_in,
    r_in,
    heads,
    kv_heads,
    d_rel,
    padded_group,
    k_sconv,
    v_sconv,
    k_cache,
    v_cache,
    metadata,
    q_scale,
    r_scale,
    rel_proj,
    seq_lens,
    local_window,
    qk_threshold,
    rel_threshold,
    clip_eps,
    norm_eps,
    *,
    k_dynamic=None,
    v_dynamic=None,
    positions=None,
    tau_floor=None,
    tau_alpha=None,
    track_mask=None,
    track_indices=None,
    verify_windows=None,
    shear=None,
):
    """Decode attention prologue in one launch: K/V short convolutions (with cache
    update), Q/K/rel L2 clips (with log-scaling tau when positions is given), and
    V RMSNorm, and the rel-logits projection as a (tokens, heads, rel_extent) rel
    bias, bitwise equal to the separate kernels. Q and the rel bias are written with
    each group of heads // kv_heads query heads padded to padded_group heads.
    With verify_windows (k_window, v_window), the rows are MTP verify tokens: the
    convolutions read earlier tokens of the same sequence (metadata cu/si) instead
    of updating the cache, and each row's conv window is saved like
    save_intermediate_conv_windows. k_dynamic / v_dynamic are the projected conv
    weights, or their (groups, tokens, n) FP32 split-K partials. With shear =
    (seqlens_k, window_left) (verify only), the rel bias is written sheared as
    FA4's ShearingBias builds it from the compact one for that verify attention
    call: seqlens_k its cache_seqlens, window_left its left window or None (a
    window must be rel_extent - 1, as FA4 requires)."""
    tokens = q_in.shape[0]
    kv_width = kv_heads * 128
    device = q_in.device
    dtype = q_in.dtype
    stride = q_in.stride(0)
    assert all(t.stride() == (stride, 1) for t in (q_in, k_in, v_in, r_in)), (
        "projected slices must share one row stride"
    )
    q = torch.empty((tokens, kv_heads * padded_group, 128), device=device, dtype=dtype)
    k = torch.empty((tokens, kv_heads, 128), device=device, dtype=dtype)
    v = torch.empty((tokens, kv_heads, 128), device=device, dtype=dtype)
    rel_extent = rel_proj.shape[1]
    out_heads = kv_heads * padded_group
    sheared = shear is not None
    rel_padded = rel_extent + 256 if sheared else rel_extent
    rel = torch.empty(
        (
            triton.cdiv(tokens, 128) * 128 if sheared else tokens,
            out_heads,
            rel_padded,
        ),
        device=device,
        dtype=dtype,
    )
    k_conv = torch.empty((tokens, kv_width), device=device, dtype=dtype)
    v_conv = torch.empty_like(k_conv)
    dynamic = k_dynamic is not None
    width = k_sconv.shape[0] if dynamic else k_sconv.shape[1]
    assert k_sconv.stride() == v_sconv.stride() and k_cache.stride() == v_cache.stride()
    assert (track_mask is None) == (track_indices is None)
    verify = verify_windows is not None
    k_window, v_window = verify_windows if verify else (None, None)
    if verify:
        assert track_mask is None
        assert k_window.stride() == v_window.stride() and k_window.stride(-1) == 1
    assert shear is None or verify
    if (
        tokens
        and _cuda_prologue(
            q_in,
            k_in,
            v_in,
            r_in,
            heads,
            kv_heads,
            d_rel,
            padded_group,
            k_sconv,
            v_sconv,
            k_cache,
            v_cache,
            metadata,
            q_scale,
            r_scale,
            rel_proj,
            (q, k, v, rel),
            verify_windows,
            k_dynamic,
            v_dynamic,
            positions,
            tau_floor,
            tau_alpha,
            (qk_threshold, rel_threshold),
            (clip_eps, norm_eps),
            None if track_mask is None else (track_mask, track_indices),
            None
            if shear is None
            else (
                shear[0],
                -1 if shear[1] is None else shear[1],
                0.0 if shear[1] is None else float("-inf"),
            ),
        )  # fmt: skip
    ):
        return q, k, v, rel
    if tokens:
        _attention_prologue[(tokens, 2 + REL_WORKERS)](
            q_in,
            k_in,
            v_in,
            r_in,
            k_sconv,
            v_sconv,
            k_dynamic,
            v_dynamic,
            k_cache,
            v_cache,
            metadata["safe_idx"],
            metadata["cache_mask"],
            track_mask,
            track_indices,
            positions,
            q_scale,
            r_scale,
            k_conv,
            v_conv,
            q,
            k,
            v,
            rel,
            rel_proj,
            seq_lens if shear is None else shear[0],
            metadata["cu"] if verify else None,
            metadata["si"] if verify else None,
            k_window,
            v_window,
            qk_threshold,
            rel_threshold,
            clip_eps,
            norm_eps,
            stride,
            *k_sconv.stride(),
            *k_cache.stride(),
            *(k_window.stride()[:3] if verify else (0, 0, 0)),
            heads,
            triton.next_power_of_2(heads),
            kv_heads,
            padded_group,
            d_rel,
            rel_extent,
            rel_padded,
            out_heads,
            max(16, triton.next_power_of_2(out_heads)),
            float("-inf") if local_window else 0.0,
            REL_WORKERS,
            not sheared,
            width,
            dynamic,
            track_mask is not None,
            positions is not None,
            tau_floor or 1,
            tau_alpha or 0.0,
            verify,
            tokens,
            k_dynamic.shape[0] if dynamic and k_dynamic.ndim == 3 else 0,
            num_warps=4,
            enable_fp_fusion=False,
            launch_pdl=PDL,
        )
    return q, k, v, rel
