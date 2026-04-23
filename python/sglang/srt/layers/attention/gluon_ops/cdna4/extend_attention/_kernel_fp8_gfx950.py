# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified FP8 KV Gluon extend-attention kernel for gfx950 (symmetric heads).

Prefix phase uses native FP8 MFMA on the FP8 KV cache buffers; extend
phase uses BF16 on the live Q / K / V extend tensors. Two-phase shared
memory allocation covers the different layout requirements of each phase.

One @gluon.jit entry folds the basic, persistent-CTA, and split-K
launches behind two constexpr gates (same shape as the BF16 kernel):

    IS_PERSISTENT   : False = basic 3D grid (one CTA per (seq, head, m_tile)),
                      True  = persistent CTA tile-sweep loop (WCA scheduler)
    SPLIT_K         : 1     = no split,
                      >1    = partition prefix KV range across CTAs + reduce

Supports BLOCK_DMODEL in {64, 128}. FP8 D=256 is unsupported on gfx950
(MFMA_F8 at D>=256 hits an `unrealized_conversion_cast` in the LLVM
backend that cannot be materialized); a `gl.static_assert` below
enforces that, and the dispatcher refuses FP8 D=256 at launch time.

Inner-loop dispatch (same as BF16, selected by num_warps / NUM_STAGES / D):
    USE_PINGPONG=False, NS>=2, D>=128 -> 4-warp sw-pipeline
    USE_PINGPONG=False, else          -> 4-warp serial (NS=1)
    USE_PINGPONG=True  (8 warps)      -> 8-warp pingpong

The sw_pipeline_4w and pingpong_8w prefix helpers accept an unaligned
`seq_len_prefix`; they floor-align to BLOCK_N internally, run the
pipelined bulk on the aligned portion, and close with a single
masked-tail block when the length isn't a clean multiple of BLOCK_N.
Short prefixes (n_full_prefix < NUM_STAGES) bypass the pipeline and go
straight to `attn_fwd_inner_prefix_unpipelined`.
"""

from ._common import *  # noqa: F403
# ===-----------------------------------------------------------------------===#
# Unified FP8 Kernel (basic + persistent + split-K via constexpr gates)
# ===-----------------------------------------------------------------------===#


@gluon.jit
def gluon_extend_attn_fwd(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,  #
    K_Buffer,
    V_Buffer,  #
    qo_indptr,
    kv_indptr,
    kv_indices,  #
    Mask,
    MaskIndptr,
    WindowKvOffsets,  #
    sm_scale,
    kv_group_num,  #
    stride_qbs,
    stride_qh,  #
    stride_kbs,
    stride_kh,  #
    stride_vbs,
    stride_vh,  #
    stride_obs,
    stride_oh,  #
    stride_buf_kbs,
    stride_buf_kh,  #
    stride_buf_vbs,
    stride_buf_vh,  #
    IS_CAUSAL: gl.constexpr,  #
    USE_CUSTOM_MASK: gl.constexpr,
    SKIP_PREFIX_CUSTOM_MASK: gl.constexpr,  #
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    ENABLE_MASK_SPLIT: gl.constexpr,  #
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DV: gl.constexpr,
    NUM_STAGES: gl.constexpr,  #
    EXT_BLOCK_N: gl.constexpr,  #
    EXT_NUM_STAGES: gl.constexpr,  #
    ASYNC_PAD_K: gl.constexpr,
    ASYNC_PAD_V: gl.constexpr,  #
    Sinks,
    HAS_SINK: gl.constexpr,  #
    LOGIT_CAP: gl.constexpr,  #
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    v_scale,  #
    num_heads,  #         int32 scalar -- total Q heads (persistent)
    n_m_tiles,  #         int32 scalar -- ceil(max_len_extend / BLOCK_M)
    total_valid_tiles,  # int32 scalar -- batch * heads * tiles [* SPLIT_K]
    total_programs,  #    int32 scalar (= grid dim 0 for persistent)
    partial_out,  #       workspace for split-K partials
    partial_lse,  #       workspace for split-K LSE
    tile_done,  #         int32 atomic counter per output tile (split-K)
    actual_batch_size,  # int32 scalar -- batch count (upper bound for per-CTA inline scan)
    IS_PERSISTENT: gl.constexpr = False,  #
    SPLIT_K: gl.constexpr = 1,  #
    PREFIX_MASK_MODE: gl.constexpr = 0,  # 0=auto (scalar on persistent, pipelined on basic), 1=force scalar_mask, 2=force pipelined
):
    num_warps: gl.constexpr = gl.num_warps()
    PFX_SMEM_TY: gl.constexpr = K_Buffer.dtype.element_ty

    # FP8 D=256 hard guard: MFMA_F8 at D>=256 hits an unrealized_conversion_cast
    # in the LLVM backend on gfx950. The dispatcher should never route D=256
    # here, but fail compilation loudly if it does.
    tl.static_assert(
        BLOCK_DMODEL != 256,
        "FP8 D=256 is unsupported on gfx950 (MFMA_F8 unrealized_conversion_cast). "
        "Use BF16 KV cache for D=256 models (Gemma3 etc) or set "
        "SGLANG_GLUON_FP8_KV_FORCE_BF16=1.",
    )

    # Basic-path safety rail: the 4-warp / 8-warp DMA layouts in the
    # IS_PERSISTENT=False branches below assume BLOCK_DMODEL >= 128. With
    # D<128 + NS>=2 the v_async_layout register bases overflow tensor rows
    # and LLVM fails with unrealized_conversion_cast. Persistent path has
    # a dedicated D<128 branch and does not need the gate.
    if not IS_PERSISTENT:
        tl.static_assert(
            not (num_warps == 4 and NUM_STAGES >= 2 and BLOCK_DMODEL < 128),
            "FP8 basic: NUM_STAGES>=2 requires BLOCK_DMODEL>=128 on 4-warp path. "
            "Use NUM_STAGES=1 or route to persistent kernel for D<128.",
        )
        tl.static_assert(
            not (num_warps == 8 and BLOCK_DMODEL < 128),
            "FP8 basic: 8-warp DMA path is incomplete for BLOCK_DMODEL<128. "
            "Route to persistent kernel which has a dedicated D<128 branch.",
        )

    # layouts
    threads_per_warp: gl.constexpr = 64
    _mfma: gl.constexpr = make_mfma_dot_layouts(num_warps, 16, 16, 32, 8, 4)
    mma_layout: gl.constexpr = _mfma[0]
    q_dot_layout: gl.constexpr = _mfma[1]
    kt_dot_layout: gl.constexpr = _mfma[2]
    p_dot_layout: gl.constexpr = _mfma[3]
    v_dot_layout: gl.constexpr = _mfma[4]
    _fp8: gl.constexpr = make_fp8_dot_layouts(mma_layout, 16, 8)
    fp8_q_dot_layout: gl.constexpr = _fp8[0]
    fp8_kt_dot_layout: gl.constexpr = _fp8[1]
    fp8_p_dot_layout: gl.constexpr = _fp8[2]
    fp8_v_dot_layout: gl.constexpr = _fp8[3]
    _blk: gl.constexpr = make_blocked_and_slice_layouts(num_warps, mma_layout)
    blocked_layout: gl.constexpr = _blk[0]
    offs_m_layout: gl.constexpr = _blk[1]
    offs_d_layout: gl.constexpr = _blk[2]
    mma_offs_n_col: gl.constexpr = _blk[3]
    mma_offs_m_row: gl.constexpr = _blk[4]
    mma_m_layout: gl.constexpr = _blk[5]

    offs_m = gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=offs_d_layout)

    # USE_PINGPONG switches between the 8-warp pingpong body (requires
    # dedicated memory/compute warp groups and >=2 LDS stages) and the
    # 4-warp body (serial or sw-pipeline). num_warps>=8 guarantees two
    # warp groups for pingpong scheduling.
    USE_PINGPONG: gl.constexpr = num_warps >= 8
    qk_scale = sm_scale * LOG2E

    # Unified loop: basic = 1 iteration (exits via return), persistent = CTA tile loop.
    # Each persistent CTA picks up work from the shared tile_idx counter; Python
    # sized `total_valid_tiles` via a tight upper bound on sum(ceil(ext_i/BM))
    # so we honor it directly here (no CPU sync, no extra cum_tiles tensor).
    if IS_PERSISTENT:
        cta_id = gl.program_id(0)
        tile_idx = cta_id
        actual_total_valid_tiles = total_valid_tiles
    else:
        tile_idx = 0
        actual_total_valid_tiles = 1  # unused in non-persistent branch

    while tile_idx < (actual_total_valid_tiles if IS_PERSISTENT else 1):
        # Per-tile scheduling
        if not IS_PERSISTENT:
            cur_seq = gl.program_id(0)
            cur_head = gl.program_id(1)
            cur_block_m = gl.program_id(2)
            cur_kv_head = cur_head // kv_group_num

            cur_seq_q_start_idx = gl.load(qo_indptr + cur_seq)
            seq_len_extend = (gl.load(qo_indptr + cur_seq + 1) - cur_seq_q_start_idx).to(tl.int32)
            is_valid_tile = cur_block_m * BLOCK_M < seq_len_extend
            seq_len_extend = tl.where(is_valid_tile, seq_len_extend, 0)
            cur_seq_kv_start_idx = gl.load(kv_indptr + cur_seq)
            seq_len_prefix_raw = (gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx).to(tl.int32)
            seq_len_prefix = tl.where(is_valid_tile, seq_len_prefix_raw, 0)
        else:
            if SPLIT_K > 1:
                output_tile = tile_idx // SPLIT_K
                k_split_id = tile_idx % SPLIT_K
            else:
                output_tile = tile_idx
                k_split_id = 0

            # WCA tile map: inline linear scan over qo_indptr to resolve
            # output_tile -> (seq, head, block_m). The grid is sized with a
            # tight upper bound on sum(ceil(ext_i/BM)), so some CTAs may claim
            # an over-provisioned slot for which no seq has enough tiles; the
            # scan walks off the end with found=0 and we mark the tile invalid.
            # Linear O(B) work, trivial for typical serving batches (B<=32).
            cur_seq = 0
            cum_tiles = 0
            found = 0
            while (cur_seq < actual_batch_size) & (found == 0):
                _s_start = gl.load(qo_indptr + cur_seq)
                _s_end = gl.load(qo_indptr + cur_seq + 1)
                s_ext = (_s_end - _s_start).to(tl.int32)
                s_tiles = tl.maximum((s_ext + BLOCK_M - 1) // BLOCK_M, 0) * num_heads
                next_cum = cum_tiles + s_tiles
                if next_cum > output_tile:
                    found = 1
                else:
                    cum_tiles = next_cum
                    cur_seq = cur_seq + 1
            is_valid_tile = found == 1
            # Clamp cur_seq to [0, actual_batch_size-1] so subsequent qo/kv
            # indptr loads stay in bounds for invalid tiles (we mask off the
            # work via seq_len_extend=0 below).
            cur_seq = tl.minimum(cur_seq, actual_batch_size - 1)
            local_tile = output_tile - cum_tiles
            seq_ext_len = (gl.load(qo_indptr + cur_seq + 1) - gl.load(qo_indptr + cur_seq)).to(tl.int32)
            tiles_per_head = tl.maximum((seq_ext_len + BLOCK_M - 1) // BLOCK_M, 1)
            cur_head = local_tile // tiles_per_head
            cur_block_m = local_tile % tiles_per_head
            cur_kv_head = cur_head // kv_group_num

            cur_seq_q_start_idx = gl.load(qo_indptr + cur_seq)
            seq_len_extend = tl.where(is_valid_tile, seq_ext_len, 0)

            cur_seq_kv_start_idx = gl.load(kv_indptr + cur_seq)
            seq_len_prefix_raw = (gl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx).to(tl.int32)
            seq_len_prefix = tl.where(is_valid_tile, seq_len_prefix_raw, 0)

        if USE_CUSTOM_MASK:
            mask_base_idx = gl.load(MaskIndptr + cur_seq).to(tl.int64)
            window_kv_offset = 0
            if SLIDING_WINDOW_SIZE > 0:
                window_kv_offset = gl.load(WindowKvOffsets + cur_seq)
            cur_seq_len = seq_len_prefix + seq_len_extend
            mask_row_stride = (cur_seq_len + window_kv_offset).to(tl.int64)
            mask_base_idx = mask_base_idx + window_kv_offset.to(tl.int64)
            mask_kv_col_offset = (seq_len_prefix).to(tl.int64)
        else:
            mask_base_idx = tl.cast(0, tl.int64)
            mask_row_stride = tl.cast(0, tl.int64)
            mask_kv_col_offset = tl.cast(0, tl.int64)
        # Q load
        q_ptrs = (
            Q_Extend
            + (cur_seq_q_start_idx + cur_block_m * BLOCK_M + offs_m[:, None]) * stride_qbs
            + cur_head * stride_qh
            + offs_d[None, :]
        )
        q_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
        q = gl.load(q_ptrs, mask=q_mask, other=0.0)

        # softmax state
        m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=mma_m_layout)
        l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_m_layout)
        acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=mma_layout)

        q_abs_pos = (
            seq_len_prefix
            + cur_block_m * BLOCK_M
            + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        )
        q_extend_raw = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=mma_offs_m_row)
        if USE_CUSTOM_MASK:
            q_extend_offs = tl.minimum(q_extend_raw, tl.maximum(seq_len_extend - 1, 0))
        else:
            q_extend_offs = q_extend_raw

        if XAI_TEMPERATURE_LEN > 0:
            inv_log2_len = 1.0 / tl.log2(float(XAI_TEMPERATURE_LEN))
            xai_temperature_reg = gl.where(
                q_abs_pos > XAI_TEMPERATURE_LEN,
                tl.log2(q_abs_pos.to(gl.float32)) * inv_log2_len,
                gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row),
            )
        else:
            xai_temperature_reg = gl.full(
                [BLOCK_M], 1.0, dtype=gl.float32, layout=mma_offs_m_row
            )

        # SWA prefix skip: jump past prefix tiles entirely outside the window.
        # For the M-tile, min q_abs_pos = seq_len_prefix + cur_block_m * BLOCK_M.
        # Any prefix block whose last key position < (min_q - SWS) is fully masked.
        pfx_kv_start = cur_seq_kv_start_idx
        pfx_seq_len = seq_len_prefix
        pfx_q_abs_pos = q_abs_pos
        pfx_mask_base = mask_base_idx
        if SLIDING_WINDOW_SIZE > 0:
            q_min_abs = seq_len_prefix + cur_block_m * BLOCK_M
            first_useful_pos = tl.maximum(q_min_abs - SLIDING_WINDOW_SIZE, 0)
            prefix_skip_n = (first_useful_pos // BLOCK_N) * BLOCK_N
            pfx_kv_start = cur_seq_kv_start_idx + prefix_skip_n
            pfx_seq_len = seq_len_prefix - prefix_skip_n
            pfx_q_abs_pos = q_abs_pos - prefix_skip_n
            if USE_CUSTOM_MASK:
                pfx_mask_base = mask_base_idx + prefix_skip_n.to(tl.int64)

        orig_seq_len_extend = seq_len_extend
        if IS_PERSISTENT and SPLIT_K > 1:
            n_pfx_blocks = (pfx_seq_len + BLOCK_N - 1) // BLOCK_N
            blocks_per_split = (n_pfx_blocks + SPLIT_K - 1) // SPLIT_K
            my_block_start = k_split_id * blocks_per_split
            my_block_end = tl.minimum((k_split_id + 1) * blocks_per_split, n_pfx_blocks)
            split_start_offset = my_block_start * BLOCK_N
            pfx_kv_start = pfx_kv_start + split_start_offset
            pfx_seq_len = tl.minimum(my_block_end * BLOCK_N, pfx_seq_len) - split_start_offset
            pfx_seq_len = tl.maximum(pfx_seq_len, 0)
            pfx_q_abs_pos = pfx_q_abs_pos - split_start_offset
            if USE_CUSTOM_MASK:
                pfx_mask_base = pfx_mask_base + split_start_offset.to(tl.int64)
            if k_split_id < SPLIT_K - 1:
                seq_len_extend = 0

        # --- Extend preamble (path-independent) ---
        # Derive the causal/SWA-clamped extend range, decide how many full
        # vs masked extend blocks we have, and stage the K/V extend base
        # pointers. All three inner-loop paths (4w sw-pipeline, 4w serial,
        # 8w pingpong) consume the same values, so compute once here.
        #
        # FP8 specificity: the 4w sw-pipeline path uses EXT_BLOCK_N / EXT_NUM_STAGES
        # for the extend phase (FP8 prefix -> BF16 extend can run a different
        # tile size); the 4w serial and 8w pingpong paths reuse BLOCK_N /
        # NUM_STAGES. The constexpr selects the right pair at compile time,
        # so the preamble folds to a single block.
        _SW_PIPELINE_4W: gl.constexpr = (not USE_PINGPONG) and NUM_STAGES >= 2 and BLOCK_DMODEL >= 128
        _EXT_N: gl.constexpr = EXT_BLOCK_N if _SW_PIPELINE_4W else BLOCK_N
        _EXT_NS: gl.constexpr = EXT_NUM_STAGES if _SW_PIPELINE_4W else NUM_STAGES
        if IS_CAUSAL:
            causal_kv_end = (cur_block_m + 1) * BLOCK_M
            effective_end = tl.minimum(seq_len_extend, causal_kv_end)
        else:
            effective_end = seq_len_extend
        n_extend_blocks = (effective_end + _EXT_N - 1) // _EXT_N
        if (not ENABLE_MASK_SPLIT) or USE_CUSTOM_MASK or SLIDING_WINDOW_SIZE > 0:
            # One combined dispatch covers the whole range with per-step masking.
            # FP8 merges the SWS>0 case into the "no fast-path" bucket: with
            # sliding windows, every extend block needs per-step masking, so
            # there's nothing to gain from splitting bulk vs tail.
            n_full_blocks = 0
        else:
            # Split into unmasked bulk (fast path, no per-step mask) + masked tail.
            partial_block = ((effective_end % _EXT_N) != 0).to(tl.int32)
            if IS_CAUSAL:
                masked_blocks = ((BLOCK_M + _EXT_N - 1) // _EXT_N) + partial_block
            else:
                masked_blocks = partial_block
            masked_blocks = tl.minimum(masked_blocks, n_extend_blocks)
            n_full_blocks = n_extend_blocks - masked_blocks

        # SWA fast-skip: any extend block whose last key-pos <
        # (min_q - SWS) is fully outside the causal+SWA intersection,
        # so skip the load and compute entirely. `masked_start` below
        # is clamped by this value. No-op when SWA inactive.
        if SLIDING_WINDOW_SIZE > 0 and IS_CAUSAL:
            swa_first_useful = tl.maximum(
                cur_block_m * BLOCK_M - SLIDING_WINDOW_SIZE, 0
            )
            swa_skip_n_blocks = swa_first_useful // _EXT_N
        else:
            swa_skip_n_blocks = 0

        k_extend_base = (
            K_Extend + cur_seq_q_start_idx * stride_kbs + cur_kv_head * stride_kh
        )
        v_extend_base = (
            V_Extend + cur_seq_q_start_idx * stride_vbs + cur_kv_head * stride_vh
        )

        if not USE_PINGPONG:
            if NUM_STAGES >= 2 and BLOCK_DMODEL >= 128:
                # ---- 4w sw-pipeline ----
                # 4-warp, async DMA K/V loads with NUM_STAGES prefetch depth.
                # Prefix uses native FP8 MFMA on FP8 KV; extend uses BF16 MFMA
                # on the live extend tensors (hence two sets of layouts).
                # Layouts via _common.make_fp8_{kt,v}_*.
                # (see that module for per-(NW,D,Dv,N) base tables).
                kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
                kt_async_layout: gl.constexpr = make_fp8_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
                v_offset_bases: gl.constexpr = make_fp8_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N)
                v_async_layout: gl.constexpr = make_fp8_v_dll(num_warps, BLOCK_DMODEL, BLOCK_DV, BLOCK_N)

                # BF16 extend: shares the FP8 offset-bases ladder; the DLL differs.
                bf16_kt_offset_bases: gl.constexpr = kt_offset_bases
                bf16_kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
                bf16_v_offset_bases: gl.constexpr = v_offset_bases
                bf16_v_async_layout: gl.constexpr = make_fp8_extend_v_dll(num_warps, BLOCK_DV, BLOCK_N)

                # FP8 prefix SMEM: 1024-byte interval for 128-bit direct-to-LDS.
                fp8_kt_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[1024, ASYNC_PAD_K], [2048, 32]])
                fp8_v_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[1024, ASYNC_PAD_V], [2048, 32]])
                # BF16 extend SMEM: standard 512-byte interval; extend tile may
                # differ from prefix tile, in which case EXT-specific factories.
                if EXT_BLOCK_N == BLOCK_N:
                    ext_kt_offset_bases: gl.constexpr = bf16_kt_offset_bases
                    ext_kt_async_layout: gl.constexpr = bf16_kt_async_layout
                    ext_v_offset_bases: gl.constexpr = bf16_v_offset_bases
                    ext_v_async_layout: gl.constexpr = bf16_v_async_layout
                else:
                    ext_kt_offset_bases: gl.constexpr = make_ext_kt_offset_bases(num_warps, BLOCK_DMODEL, EXT_BLOCK_N)
                    ext_kt_async_layout: gl.constexpr = make_ext_kt_dll(num_warps, BLOCK_DMODEL, EXT_BLOCK_N)
                    ext_v_offset_bases: gl.constexpr = make_ext_v_offset_bases(num_warps, BLOCK_DV, EXT_BLOCK_N)
                    ext_v_async_layout: gl.constexpr = make_ext_v_dll(num_warps, BLOCK_DV, EXT_BLOCK_N)

                bf16_kt_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, EXT_BLOCK_N], ext_kt_offset_bases, [[512, ASYNC_PAD_K]])
                bf16_v_smem_layout: gl.constexpr = make_padded_smem([EXT_BLOCK_N, BLOCK_DV], ext_v_offset_bases, [[512, ASYNC_PAD_V]])

                kt_smem = gl.allocate_shared_memory(
                    PFX_SMEM_TY,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=fp8_kt_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    PFX_SMEM_TY,
                    [NUM_STAGES, BLOCK_N, BLOCK_DV],
                    layout=fp8_v_smem_layout,
                )

                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DV],
                        dtype=PFX_SMEM_TY,
                        layout=v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

                fp8_q_dot = gl.convert_layout(q.to(tl.float8e4nv), fp8_q_dot_layout)
                q_dot = gl.convert_layout(q, q_dot_layout)

                # Prefix phase uses FP8 dot layouts for native FP8 MFMA.
                # sw_pipeline_4w floor-aligns internally and runs a masked
                # tail block when the full length isn't BLOCK_N-aligned; on
                # the persistent path that's a no-op because split-K already
                # aligned pfx_seq_len.
                if pfx_seq_len > 0:
                    n_full_prefix = pfx_seq_len // BLOCK_N
                    n_extend_est = (seq_len_extend + BLOCK_N - 1) // BLOCK_N
                    # LOGIT_CAP gating: if the extend phase is itself large
                    # enough to pipeline, running a pipelined prefix ahead of
                    # it inflates LOGIT_CAP register pressure. Fall back to
                    # the unpipelined prefix to keep occupancy.
                    use_pipe_prefix = n_full_prefix >= NUM_STAGES
                    if LOGIT_CAP > 0:
                        use_pipe_prefix = use_pipe_prefix and (n_extend_est < NUM_STAGES)
                    if use_pipe_prefix:
                        acc, l_i, m_i = attn_fwd_inner_prefix_sw_pipeline_4w(
                            acc,
                            l_i,
                            m_i,
                            fp8_q_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            BLOCK_DV,
                            NUM_STAGES,
                            kt_async_layout,
                            v_async_layout,
                            fp8_kt_dot_layout,
                            fp8_p_dot_layout,
                            fp8_v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                        )
                    else:
                        acc, l_i, m_i = attn_fwd_inner_prefix_unpipelined(
                            acc,
                            l_i,
                            m_i,
                            fp8_q_dot,
                            K_Buffer,
                            V_Buffer,
                            kv_indices,
                            pfx_kv_start,
                            cur_kv_head,
                            pfx_seq_len,
                            stride_buf_kbs,
                            stride_buf_kh,
                            stride_buf_vbs,
                            stride_buf_vh,
                            kt_smem,
                            v_smem,
                            qk_scale,
                            LOGIT_CAP,
                            xai_temperature_reg,
                            XAI_TEMPERATURE_LEN,
                            pfx_q_abs_pos,
                            SLIDING_WINDOW_SIZE,
                            Mask,
                            pfx_mask_base,
                            mask_row_stride,
                            q_extend_offs,
                            USE_CUSTOM_MASK,
                            SKIP_PREFIX_CUSTOM_MASK,
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            BLOCK_DV,
                            kt_async_layout,
                            v_async_layout,
                            fp8_kt_dot_layout,
                            fp8_p_dot_layout,
                            fp8_v_dot_layout,
                            mma_layout,
                            mma_offs_n_col,
                        )

                # Transition FP8 prefix smem -> BF16 extend smem
                kt_smem._keep_alive()
                v_smem._keep_alive()
                kt_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [EXT_NUM_STAGES, BLOCK_DMODEL, EXT_BLOCK_N],
                    layout=bf16_kt_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [EXT_NUM_STAGES, EXT_BLOCK_N, BLOCK_DV],
                    layout=bf16_v_smem_layout,
                )
                for _s in gl.static_range(EXT_NUM_STAGES):
                    v_zero = gl.zeros(
                        [EXT_BLOCK_N, BLOCK_DV],
                        dtype=Q_Extend.dtype.element_ty,
                        layout=ext_v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

                # ===---- EXTEND hotloop (4w sw-pipeline, FP8 kernel) ---===
                # Extend phase runs in BF16 MFMA on the live K/V extend
                # tensors (FP8 is only used for the prefix / KV-cache side),
                # so this block is structurally identical to the BF16 kernel:
                #
                #   [0, n_full_blocks)                       -- UNMASKED BULK
                #   [max(n_full_blocks, swa_skip_n_blocks),
                #                            n_extend_blocks) -- MASKED TAIL
                #
                # Each phase picks pipelined (>=_EXT_NS) or unpipelined
                # (<_EXT_NS) helpers. _EXT_N and _EXT_NS are the extend-
                # side BLOCK_N / NUM_STAGES (distinct from the prefix-side
                # FP8 BLOCK_N / NUM_STAGES, which use a wider K tile).
                #
                # --- UNMASKED BULK ---
                if n_full_blocks >= _EXT_NS:
                    acc, l_i, m_i = attn_fwd_inner_extend_sw_pipeline_4w(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        EXT_BLOCK_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        EXT_NUM_STAGES,
                        ext_kt_async_layout,
                        ext_v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                        SKIP_BOUNDS_CHECK=True,
                    )
                elif n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_unpipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        EXT_BLOCK_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        ext_kt_async_layout,
                        ext_v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                # --- MASKED TAIL ---
                masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
                remaining_blocks = n_extend_blocks - masked_start
                if remaining_blocks >= _EXT_NS:
                    acc, l_i, m_i = attn_fwd_inner_extend_sw_pipeline_4w(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        EXT_BLOCK_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        EXT_NUM_STAGES,
                        ext_kt_async_layout,
                        ext_v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                elif remaining_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_unpipelined(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_smem,
                        v_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        EXT_BLOCK_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        ext_kt_async_layout,
                        ext_v_async_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )

            else:
                # ---- 4w serial ----
                # 4-warp, synchronous (NUM_STAGES=1 or D<128). K/V are loaded
                # into smem via blocked-layout gl.load, no DMA prefetch.
                # Handles both bulk and masked tail with one helper (serial
                # helper masks per step).
                kt_blocked_layout: gl.constexpr = make_serial_kt_blocked(num_warps)
                kt_serial_smem_layout: gl.constexpr = SERIAL_KT_SMEM
                v_serial_smem_layout: gl.constexpr = SERIAL_V_SMEM
                q_smem_layout: gl.constexpr = SERIAL_Q_SMEM

                kt_serial_smem = gl.allocate_shared_memory(
                    PFX_SMEM_TY,
                    [BLOCK_DMODEL, BLOCK_N],
                    layout=kt_serial_smem_layout,
                )
                v_serial_smem = gl.allocate_shared_memory(
                    PFX_SMEM_TY,
                    [BLOCK_N, BLOCK_DV],
                    layout=v_serial_smem_layout,
                )
                q_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_M, BLOCK_DMODEL],
                    layout=q_smem_layout,
                )

                q_smem.store(q)
                q_dot = q_smem.load(q_dot_layout)
                fp8_q_dot = gl.convert_layout(q.to(tl.float8e4nv), fp8_q_dot_layout)

                if pfx_seq_len > 0:
                    acc, l_i, m_i = attn_fwd_inner_prefix_serial_4w(
                        acc,
                        l_i,
                        m_i,
                        fp8_q_dot,
                        K_Buffer,
                        V_Buffer,
                        kv_indices,
                        pfx_kv_start,
                        cur_kv_head,
                        pfx_seq_len,
                        stride_buf_kbs,
                        stride_buf_kh,
                        stride_buf_vbs,
                        stride_buf_vh,
                        kt_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        pfx_q_abs_pos,
                        SLIDING_WINDOW_SIZE,
                        Mask,
                        pfx_mask_base,
                        mask_row_stride,
                        q_extend_offs,
                        USE_CUSTOM_MASK,
                        SKIP_PREFIX_CUSTOM_MASK,
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        kt_blocked_layout,
                        blocked_layout,
                        fp8_kt_dot_layout,
                        fp8_p_dot_layout,
                        fp8_v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                    )

                # Transition serial smem from FP8 (prefix) to BF16 (extend),
                # mirroring the DMA path's smem transition.
                kt_serial_smem._keep_alive()
                v_serial_smem._keep_alive()
                kt_serial_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_DMODEL, BLOCK_N],
                    layout=kt_serial_smem_layout,
                )
                v_serial_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_N, BLOCK_DV],
                    layout=v_serial_smem_layout,
                )

                # ===---- EXTEND hotloop (4w serial, FP8 kernel) --------===
                # Extend phase runs in BF16 on the live K/V extend tensors.
                # Same two-range structure as the BF16 kernel -- one helper,
                # two ranges, IS_MASKED toggled:
                #   [0, n_full_blocks)              -- bulk (IS_MASKED=False)
                #   [masked_start, n_extend_blocks) -- tail (IS_MASKED=True)
                # The serial helper does its own per-step masking, so we
                # don't need separate pipelined/unpipelined helpers here.
                if n_full_blocks > 0:
                    acc, l_i, m_i = attn_fwd_inner_extend_serial_4w(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        0,
                        n_full_blocks,
                        kt_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        False,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        kt_blocked_layout,
                        blocked_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )
                # --- MASKED TAIL (same helper, different range + IS_MASKED) ---
                masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
                if n_extend_blocks > masked_start:
                    acc, l_i, m_i = attn_fwd_inner_extend_serial_4w(
                        acc,
                        l_i,
                        m_i,
                        q_dot,
                        k_extend_base,
                        v_extend_base,
                        cur_block_m,
                        seq_len_extend,
                        stride_kbs,
                        stride_vbs,
                        masked_start,
                        n_extend_blocks,
                        kt_serial_smem,
                        v_serial_smem,
                        qk_scale,
                        LOGIT_CAP,
                        xai_temperature_reg,
                        XAI_TEMPERATURE_LEN,
                        SLIDING_WINDOW_SIZE,
                        IS_CAUSAL,
                        Mask,
                        mask_base_idx,
                        mask_row_stride,
                        mask_kv_col_offset,
                        USE_CUSTOM_MASK,
                        True,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        kt_blocked_layout,
                        blocked_layout,
                        kt_dot_layout,
                        p_dot_layout,
                        v_dot_layout,
                        mma_layout,
                        mma_offs_n_col,
                        mma_offs_m_row,
                    )

        else:
            # ---- 8w pingpong ----
            # 8-warp ping-pong: 4 warps issue async DMA loads while the other
            # 4 warps MFMA on the previous tile, alternating each iteration.
            # Uniform across BLOCK_DMODEL={64, 128, 256}: FP8 layout factories
            # accept BLOCK_DMODEL directly, so there's a single body.
            # Layouts from _common factory (FP8 prefix + BF16 extend).
            kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
            kt_async_layout: gl.constexpr = make_fp8_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
            v_offset_bases: gl.constexpr = make_fp8_v_offset_bases(num_warps, BLOCK_DV, BLOCK_N)
            v_async_layout: gl.constexpr = make_fp8_v_dll(num_warps, BLOCK_DMODEL, BLOCK_DV, BLOCK_N)

            # BF16 extend: shares FP8 offset-bases ladder; DLL differs.
            bf16_kt_offset_bases: gl.constexpr = kt_offset_bases
            bf16_kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
            bf16_v_offset_bases: gl.constexpr = v_offset_bases
            bf16_v_async_layout: gl.constexpr = make_fp8_extend_v_dll(num_warps, BLOCK_DV, BLOCK_N)

            # FP8 prefix smem: 1024-byte interval for 128-bit direct-to-LDS.
            fp8_kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[1024, ASYNC_PAD_K], [2048, 32]])
            fp8_v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[1024, ASYNC_PAD_V], [2048, 32]])
            # BF16 extend smem: standard 512-byte interval.
            bf16_kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], bf16_kt_offset_bases, [[512, ASYNC_PAD_K]])
            bf16_v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], bf16_v_offset_bases, [[512, ASYNC_PAD_V]])
            kt_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=fp8_kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=fp8_v_async_smem_layout,
            )

            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=PFX_SMEM_TY,
                    layout=v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            fp8_q_dot = gl.convert_layout(q.to(tl.float8e4nv), fp8_q_dot_layout)
            q_dot = gl.convert_layout(q, q_dot_layout)

            # Prefix dispatch (FP8 dot layouts for native FP8 MFMA): the
            # pingpong helper floor-aligns internally and runs a one-block
            # masked tail when needed. Persistent path's pfx_seq_len is
            # BLOCK_N-aligned by split-K, so the tail is a runtime no-op.
            # PREFIX_MASK_MODE: 0=auto, 1=force scalar_mask, 2=force pipelined.
            # Auto picks scalar on persistent (parity-to-wins with FP8
            # per bench_scalar_mask.py --fp8 Apr 2026), pipelined on
            # basic (near-identical but pipelined has a tiny edge).
            n_full_prefix = pfx_seq_len // BLOCK_N
            _use_scalar_mask: gl.constexpr = (PREFIX_MASK_MODE == 1) or (
                PREFIX_MASK_MODE == 0 and IS_PERSISTENT
            )
            if n_full_prefix >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_prefix_pingpong_8w(
                    acc,
                    l_i,
                    m_i,
                    fp8_q_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    fp8_kt_dot_layout,
                    fp8_p_dot_layout,
                    fp8_v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    _use_scalar_mask,
                )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_unpipelined(
                    acc,
                    l_i,
                    m_i,
                    fp8_q_dot,
                    K_Buffer,
                    V_Buffer,
                    kv_indices,
                    pfx_kv_start,
                    cur_kv_head,
                    pfx_seq_len,
                    stride_buf_kbs,
                    stride_buf_kh,
                    stride_buf_vbs,
                    stride_buf_vh,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    pfx_q_abs_pos,
                    SLIDING_WINDOW_SIZE,
                    Mask,
                    pfx_mask_base,
                    mask_row_stride,
                    q_extend_offs,
                    USE_CUSTOM_MASK,
                    SKIP_PREFIX_CUSTOM_MASK,
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    fp8_kt_dot_layout,
                    fp8_p_dot_layout,
                    fp8_v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                )

            # After prefix, before extend - transition FP8 smem -> BF16 smem
            kt_smem._keep_alive()
            v_smem._keep_alive()
            kt_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=bf16_kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                Q_Extend.dtype.element_ty,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=bf16_v_async_smem_layout,
            )
            for _s in gl.static_range(NUM_STAGES):
                v_zero = gl.zeros(
                    [BLOCK_N, BLOCK_DV],
                    dtype=Q_Extend.dtype.element_ty,
                    layout=bf16_v_async_layout,
                )
                v_smem.index(_s).store(v_zero)
            gl.barrier()

            # ===---- EXTEND hotloop (8w pingpong, FP8 kernel) ----------===
            # Extend phase runs in BF16 on the live K/V extend tensors
            # (FP8 is only used on the prefix / KV-cache side). Structure
            # matches the BF16 kernel's 8w pingpong extend dispatch:
            #   [0, n_full_blocks)                       -- UNMASKED BULK
            #   [max(n_full_blocks, swa_skip_n_blocks),
            #                            n_extend_blocks) -- MASKED TAIL
            # Each phase picks pingpong_8w (>=_EXT_NS) or unpipelined
            # (<_EXT_NS); _EXT_N/_EXT_NS are the extend-side block/stage
            # counts (on this path they equal BLOCK_N / NUM_STAGES).
            #
            # --- UNMASKED BULK ---
            if n_full_blocks >= _EXT_NS:
                acc, l_i, m_i = attn_fwd_inner_extend_pingpong_8w(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DV,
                    NUM_STAGES,
                    bf16_kt_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            elif n_full_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_unpipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    0,
                    n_full_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    False,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DV,
                    bf16_kt_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            # --- MASKED TAIL ---
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= _EXT_NS:
                acc, l_i, m_i = attn_fwd_inner_extend_pingpong_8w(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DV,
                    NUM_STAGES,
                    bf16_kt_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )
            elif remaining_blocks > 0:
                acc, l_i, m_i = attn_fwd_inner_extend_unpipelined(
                    acc,
                    l_i,
                    m_i,
                    q_dot,
                    k_extend_base,
                    v_extend_base,
                    cur_block_m,
                    seq_len_extend,
                    stride_kbs,
                    stride_vbs,
                    masked_start,
                    n_extend_blocks,
                    kt_smem,
                    v_smem,
                    qk_scale,
                    LOGIT_CAP,
                    xai_temperature_reg,
                    XAI_TEMPERATURE_LEN,
                    SLIDING_WINDOW_SIZE,
                    IS_CAUSAL,
                    Mask,
                    mask_base_idx,
                    mask_row_stride,
                    mask_kv_col_offset,
                    USE_CUSTOM_MASK,
                    True,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DV,
                    bf16_kt_async_layout,
                    bf16_v_async_layout,
                    kt_dot_layout,
                    p_dot_layout,
                    v_dot_layout,
                    mma_layout,
                    mma_offs_n_col,
                    mma_offs_m_row,
                )

        if HAS_SINK:
            cur_sink = gl.load(Sinks + cur_head)
            l_i = l_i + gl.exp2(cur_sink * LOG2E - m_i)

        if IS_PERSISTENT and SPLIT_K > 1:
            l_recip_sk = 1.0 / l_i
            acc_normed = acc * l_recip_sk[:, None]
            lse = m_i + tl.log2(l_i)
            split_idx = output_tile * SPLIT_K + k_split_id

            po_base = partial_out + split_idx * BLOCK_M * BLOCK_DV
            po_ptrs = po_base + offs_m[:, None] * BLOCK_DV + offs_dv[None, :]
            po_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < orig_seq_len_extend
            po_val = gl.convert_layout(acc_normed, blocked_layout)
            gl.store(po_ptrs, po_val, mask=po_mask)

            pl_base = partial_lse + split_idx * BLOCK_M
            pl_ptrs = pl_base + offs_m
            pl_mask = (cur_block_m * BLOCK_M + offs_m) < orig_seq_len_extend
            lse_val = gl.convert_layout(lse, offs_m_layout)
            gl.store(pl_ptrs, lse_val, mask=pl_mask)

            done = tl.atomic_add(tile_done + output_tile, 1)
            if done == SPLIT_K - 1:
                r_m_mask = (cur_block_m * BLOCK_M + offs_m) < orig_seq_len_extend

                r_base_0 = output_tile * SPLIT_K
                r_lse = gl.load(
                    partial_lse + r_base_0 * BLOCK_M + offs_m,
                    mask=r_m_mask, other=float("-inf"),
                )
                r_acc = gl.load(
                    partial_out + r_base_0 * BLOCK_M * BLOCK_DV
                    + offs_m[:, None] * BLOCK_DV + offs_dv[None, :],
                    mask=r_m_mask[:, None], other=0.0,
                )
                for _sk in tl.static_range(1, SPLIT_K):
                    r_base_k = r_base_0 + _sk
                    lse_k = gl.load(
                        partial_lse + r_base_k * BLOCK_M + offs_m,
                        mask=r_m_mask, other=float("-inf"),
                    )
                    acc_k = gl.load(
                        partial_out + r_base_k * BLOCK_M * BLOCK_DV
                        + offs_m[:, None] * BLOCK_DV + offs_dv[None, :],
                        mask=r_m_mask[:, None], other=0.0,
                    )
                    max_lse = gl.maximum(r_lse, lse_k)
                    w_old = gl.exp2(r_lse - max_lse)
                    w_new = gl.exp2(lse_k - max_lse)
                    denom = w_old + w_new
                    r_acc = (r_acc * w_old[:, None] + acc_k * w_new[:, None]) / denom[:, None]
                    r_lse = max_lse + tl.log2(denom)

                r_acc = r_acc * v_scale
                r_o_base = O_Extend + cur_seq_q_start_idx * stride_obs + cur_head * stride_oh
                r_o_offsets = ((cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs + offs_dv[None, :]).to(tl.int32)
                r_o_mask = r_m_mask[:, None]
                out_r = gl.convert_layout(r_acc, blocked_layout).to(O_Extend.dtype.element_ty)
                cdna4_buffer_store(out_r, r_o_base, r_o_offsets, mask=r_o_mask)
        else:
            l_recip = 1.0 / l_i
            acc = acc * l_recip[:, None]
            acc = acc * v_scale

            o_base = O_Extend + cur_seq_q_start_idx * stride_obs + cur_head * stride_oh
            o_offsets = ((cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs + offs_dv[None, :]).to(tl.int32)
            o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
            out = gl.convert_layout(acc, blocked_layout).to(O_Extend.dtype.element_ty)
            cdna4_buffer_store(out, o_base, o_offsets, mask=o_mask)

        if IS_PERSISTENT:
            tile_idx += total_programs
        else:
            tile_idx = 1
