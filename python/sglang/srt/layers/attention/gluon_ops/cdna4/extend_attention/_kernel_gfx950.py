# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unified Gluon extend-attention kernel for gfx950 (BF16 + FP8 KV).

Symmetric heads only (D64/D128/D256 for BF16, D64/D128 for FP8). One
@gluon.jit entry folds the basic, persistent-CTA, split-K launches *and*
both KV dtypes behind three constexpr gates:

    IS_PERSISTENT   : False = basic 3D grid (one CTA per (seq, head, m_tile)),
                      True  = persistent CTA tile-sweep loop (WCA scheduler)
    SPLIT_K         : 1     = no split,
                      >1    = partition prefix KV range across CTAs + reduce
    IS_FP8          : False = BF16 KV, single-phase smem + BF16 MFMA everywhere
                      True  = FP8 KV cache; prefix runs native FP8 MFMA on the
                              cache buffers, extend runs BF16 MFMA on the live
                              extend tensors. Two-phase smem: the prefix loads
                              into a wide FP8-padded layout, then _keep_alive()
                              releases it and the extend phase re-allocates
                              smem with the narrower BF16 padding.

When IS_PERSISTENT=False the scheduling preamble collapses to a one-shot
iteration and the split-K branches DCE away, leaving a kernel functionally
identical to a plain basic launch. The persistent path walks a single
output_tile -> (seq, head, block_m) inline scan over qo_indptr (no CPU-side
cum_tiles tensor); split-K layers on top by striping the prefix across CTAs.

The only internally derived (not user-tunable) constexpr is BLOCK_DV
(=BLOCK_DMODEL). There is no BLOCK_DPE -- DeepSeek/MLA lives in a
separate kernel. The masked-tail vs unmasked-bulk split is enabled for
BLOCK_DMODEL < 256; at D=256 the smem budget forces a single masked
dispatch over the whole range.

Inner-loop dispatch (selected by num_warps, NUM_STAGES, D):
    USE_PINGPONG=False, NS>=2, D>=128 -> 4-warp sw-pipeline (async DMA, pipelined)
    USE_PINGPONG=False, else          -> 4-warp serial      (synchronous, NS=1)
    USE_PINGPONG=True  (8 warps)      -> 8-warp pingpong    (async DMA, pipelined)

The sw_pipeline_4w and pingpong_8w prefix helpers accept an unaligned
`seq_len_prefix`; they floor-align to BLOCK_N internally, run the
pipelined bulk on the aligned portion, and close with a single
masked-tail block when the length isn't a clean multiple of BLOCK_N.
Short prefixes (n_full_prefix < NUM_STAGES) bypass the pipeline and go
straight to `attn_fwd_inner_prefix_unpipelined`, which masks every block.
"""

from ._common import *  # noqa: F403
# ===-----------------------------------------------------------------------===#
# Unified BF16 + FP8 KV Kernel
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
    ENABLE_PREFIX_UNMASKED: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,  #
    BLOCK_DMODEL: gl.constexpr,
    NUM_STAGES: gl.constexpr,  #
    Sinks,
    HAS_SINK: gl.constexpr,  #
    LOGIT_CAP: gl.constexpr,  #
    XAI_TEMPERATURE_LEN: gl.constexpr,  #
    SLIDING_WINDOW_SIZE: gl.constexpr,  #
    v_scale,  #
    num_heads,  #         int32 scalar -- total Q heads (persistent)
    total_valid_tiles,  # int32 scalar -- batch * heads * tiles [* SPLIT_K]
    total_programs,  #    int32 scalar (= grid dim 0 for persistent)
    partial_out,  #       workspace for split-K partials
    partial_lse,  #       workspace for split-K LSE
    tile_done,  #         int32 atomic counter per output tile (split-K)
    actual_batch_size,  # int32 scalar -- batch count (upper bound for per-CTA inline scan)
    IS_PERSISTENT: gl.constexpr = False,  #
    SPLIT_K: gl.constexpr = 1,  #
    IS_FP8: gl.constexpr = False,  # True when K_Buffer/V_Buffer are FP8 KV cache
    EXT_BLOCK_N: gl.constexpr = 0,  # FP8-only extend-phase BLOCK_N (0 => BLOCK_N)
    EXT_NUM_STAGES: gl.constexpr = 0,  # FP8-only extend-phase NUM_STAGES (0 => NUM_STAGES)
):

    num_warps: gl.constexpr = gl.num_warps()

    # All dot / blocked / slice layouts live on a single aggregate. The
    # IS_FP8 gate flips ``pfx_*`` dot layouts to FP8 k_width (extend stays
    # BF16), and ``PFX_SMEM_TY`` picks the prefix smem element type.
    layouts: gl.constexpr = ExtendAttentionLayouts(
        IS_FP8,
        num_warps,
        Q_Extend.dtype.element_ty,
        K_Buffer.dtype.element_ty,
    )

    BLOCK_DV: gl.constexpr = BLOCK_DMODEL

    # FP8 safety rails: MFMA_F8 at D>=256 hits an unrealized_conversion_cast
    # in the LLVM backend on gfx950; the dispatcher should never route
    # FP8 D=256 here, but fail compilation loudly if it does.
    if IS_FP8:
        tl.static_assert(
            BLOCK_DMODEL != 256,
            "FP8 D=256 is unsupported by this kernel. "
            "Use BF16 KV cache for D=256 models (Gemma3 etc).",
        )
        # FP8 basic-path rails: the D<128 async-DMA layouts don't fit the
        # v_async_layout register bases; LLVM fails with
        # unrealized_conversion_cast. Persistent path has a dedicated D<128
        # branch and does not need the gate.
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

    offs_m = gl.arange(0, BLOCK_M, layout=layouts.offs_m_layout)
    offs_d = gl.arange(0, BLOCK_DMODEL, layout=layouts.offs_d_layout)
    offs_dv = gl.arange(0, BLOCK_DV, layout=layouts.offs_d_layout)

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
        tile_idx = gl.program_id(0)
    else:
        tile_idx = 0

    while tile_idx < (total_valid_tiles if IS_PERSISTENT else 1):
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
        m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=layouts.mma_m_layout)
        l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=layouts.mma_m_layout)
        acc = gl.zeros([BLOCK_M, BLOCK_DV], dtype=gl.float32, layout=layouts.mma_layout)

        q_abs_pos = (
            seq_len_prefix
            + cur_block_m * BLOCK_M
            + gl.arange(0, BLOCK_M, layout=layouts.mma_offs_m_row)
        )
        q_extend_raw = cur_block_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=layouts.mma_offs_m_row)
        if USE_CUSTOM_MASK:
            q_extend_offs = tl.minimum(q_extend_raw, tl.maximum(seq_len_extend - 1, 0))
        else:
            q_extend_offs = q_extend_raw

        if XAI_TEMPERATURE_LEN > 0:
            inv_log2_len = 1.0 / tl.log2(float(XAI_TEMPERATURE_LEN))
            xai_temperature_reg = gl.where(
                q_abs_pos > XAI_TEMPERATURE_LEN,
                tl.log2(q_abs_pos.to(gl.float32)) * inv_log2_len,
                gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=layouts.mma_offs_m_row),
            )
        else:
            xai_temperature_reg = gl.full(
                [BLOCK_M], 1.0, dtype=gl.float32, layout=layouts.mma_offs_m_row
            )

        # SWA prefix skip: jump past prefix tiles entirely outside the window.
        # For the M-tile, min q_abs_pos = seq_len_prefix + cur_block_m * BLOCK_M.
        # Any prefix block whose last key position < (min_q - SWS) is fully masked.
        pfx_kv_start = cur_seq_kv_start_idx
        pfx_seq_len = seq_len_prefix
        pfx_q_abs_pos = q_abs_pos
        if SLIDING_WINDOW_SIZE > 0:
            q_min_abs = seq_len_prefix + cur_block_m * BLOCK_M
            first_useful_pos = tl.maximum(q_min_abs - SLIDING_WINDOW_SIZE, 0)
            prefix_skip_n = (first_useful_pos // BLOCK_N) * BLOCK_N
            pfx_kv_start = cur_seq_kv_start_idx + prefix_skip_n
            pfx_seq_len = seq_len_prefix - prefix_skip_n
            pfx_q_abs_pos = q_abs_pos - prefix_skip_n


        # Split-K: partition prefix KV range (persistent only)
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
            if k_split_id < SPLIT_K - 1:
                seq_len_extend = 0

        # --- Extend preamble (path-independent) ---
        # Derive the causal/SWA-clamped extend range, decide how many full
        # vs masked extend blocks we have, and stage the K/V extend base
        # pointers. All three inner-loop paths (4w sw-pipeline, 4w serial,
        # 8w pingpong) consume the same values, so compute once here.
        # FP8 specificity: the 4w sw-pipeline path uses EXT_BLOCK_N / EXT_NUM_STAGES
        # for the extend phase (FP8 prefix -> BF16 extend can run a different
        # tile size); the 4w serial and 8w pingpong paths reuse BLOCK_N /
        # NUM_STAGES. BF16 always has EXT==PFX (0 sentinel -> BLOCK_N). The
        # constexpr selects the right pair at compile time, so the preamble
        # folds to a single block.
        _SW_PIPELINE_4W: gl.constexpr = (not USE_PINGPONG) and NUM_STAGES >= 2 and BLOCK_DMODEL >= 128
        _EXT_BN_ACTUAL: gl.constexpr = EXT_BLOCK_N if EXT_BLOCK_N > 0 else BLOCK_N
        _EXT_NS_ACTUAL: gl.constexpr = EXT_NUM_STAGES if EXT_NUM_STAGES > 0 else NUM_STAGES
        _EXT_N: gl.constexpr = _EXT_BN_ACTUAL if (IS_FP8 and _SW_PIPELINE_4W) else BLOCK_N
        _EXT_NS: gl.constexpr = _EXT_NS_ACTUAL if (IS_FP8 and _SW_PIPELINE_4W) else NUM_STAGES

        if IS_CAUSAL:
            causal_kv_end = (cur_block_m + 1) * BLOCK_M
            effective_end = tl.minimum(seq_len_extend, causal_kv_end)
        else:
            effective_end = seq_len_extend
        n_extend_blocks = (effective_end + _EXT_N - 1) // _EXT_N
        # Split the extend range into a fully-unmasked bulk + a masked tail.
        # This is only profitable when D<256; at D=256 the tile math has no
        # room for a separate masked-tail kernel, so we fall through to the
        # single-dispatch masked path below.
        if BLOCK_DMODEL >= 256 or USE_CUSTOM_MASK:
            # One combined dispatch covers the whole range with per-step masking.
            n_full_blocks = 0
        elif SLIDING_WINDOW_SIZE > 0 and (IS_FP8 or effective_end > SLIDING_WINDOW_SIZE):
            # Mixed SWA coverage: fall back to masked dispatch for all blocks.
            # FP8 takes the fallback whenever SWS>0 (FP8 FA sliding windows
            # rarely leave a large unmasked bulk; one dispatch is simpler).
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
                # Requires BLOCK_DMODEL>=128 so the register bases fit the
                # async v-layout; smaller D stays on the serial path.
                # FP8: prefix uses wide FP8 K tile with 1024-byte smem pad,
                # then _keep_alive transitions to BF16 smem for the extend
                # phase (IS_FP8=True runs MFMA_F8 on the prefix side and BF16
                # MFMA on the live extend tensors).
                kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
                v_offset_bases: gl.constexpr = prefix_v_offset_bases(layouts, num_warps, BLOCK_DV, BLOCK_N)
                kt_async_layout: gl.constexpr = prefix_kt_dll(layouts, num_warps, BLOCK_DMODEL, BLOCK_N)
                v_async_layout: gl.constexpr = prefix_v_dll(layouts, num_warps, BLOCK_DMODEL, BLOCK_DV, BLOCK_N)

                kt_smem_layout: gl.constexpr = prefix_kt_smem_layout(layouts, BLOCK_DMODEL, BLOCK_N, kt_offset_bases)
                v_smem_layout: gl.constexpr = prefix_v_smem_layout(layouts, BLOCK_N, BLOCK_DV, v_offset_bases)

                # Extend-phase layouts (only consumed when IS_FP8: extend
                # runs in BF16 after a smem dtype transition). BF16 reuses
                # the prefix layouts and smem directly, so these are
                # ignored outside the IS_FP8 branch.
                if IS_FP8:
                    if _EXT_N == BLOCK_N:
                        ext_kt_offset_bases: gl.constexpr = kt_offset_bases
                        ext_v_offset_bases: gl.constexpr = v_offset_bases
                        ext_kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
                        ext_v_async_layout: gl.constexpr = make_fp8_extend_v_dll(num_warps, BLOCK_DV, BLOCK_N)
                    else:
                        ext_kt_offset_bases: gl.constexpr = make_ext_kt_offset_bases(num_warps, BLOCK_DMODEL, _EXT_N)
                        ext_kt_async_layout: gl.constexpr = make_ext_kt_dll(num_warps, BLOCK_DMODEL, _EXT_N)
                        ext_v_offset_bases: gl.constexpr = make_ext_v_offset_bases(num_warps, BLOCK_DV, _EXT_N)
                        ext_v_async_layout: gl.constexpr = make_ext_v_dll(num_warps, BLOCK_DV, _EXT_N)
                    ext_kt_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, _EXT_N], ext_kt_offset_bases, [[512, 16]])
                    ext_v_smem_layout: gl.constexpr = make_padded_smem([_EXT_N, BLOCK_DV], ext_v_offset_bases, [[512, 16]])
                else:
                    ext_kt_async_layout: gl.constexpr = kt_async_layout
                    ext_v_async_layout: gl.constexpr = v_async_layout

                kt_smem = gl.allocate_shared_memory(
                    layouts.PFX_SMEM_TY,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=kt_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    layouts.PFX_SMEM_TY,
                    [NUM_STAGES, BLOCK_N, BLOCK_DV],
                    layout=v_smem_layout,
                )

                # Zero-fill V smem before the first async load: FP8 always
                # (the prefix -> BF16-smem transition reads all NUM_STAGES
                # slots even on short prefixes), BF16 only for valid tiles
                # (skipped tiles bail before consuming the smem).
                if IS_FP8 or is_valid_tile:
                    for _s in gl.static_range(NUM_STAGES):
                        v_zero = gl.zeros(
                            [BLOCK_N, BLOCK_DV],
                            dtype=layouts.PFX_SMEM_TY,
                            layout=v_async_layout,
                        )
                        v_smem.index(_s).store(v_zero)
                    gl.barrier()

                q_dot = gl.convert_layout(q, layouts.q_dot_layout)
                if IS_FP8:
                    pfx_q = gl.convert_layout(q.to(tl.float8e4nv), layouts.pfx_q_dot_layout)
                else:
                    pfx_q = gl.convert_layout(q, layouts.pfx_q_dot_layout)

                # Prefix loop: the pipelined helper floor-aligns internally
                # and runs a masked tail block when needed. Fall back to the
                # unpipelined helper for short prefixes (< NUM_STAGES blocks)
                # or when LOGIT_CAP would bloat register pressure.
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
                            pfx_q,
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
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            BLOCK_DV,
                            NUM_STAGES,
                            kt_async_layout,
                            v_async_layout,
                            layouts.pfx_kt_dot_layout,
                            layouts.pfx_p_dot_layout,
                            layouts.pfx_v_dot_layout,
                            layouts.mma_layout,
                            layouts.mma_offs_n_col,
                        )
                    else:
                        acc, l_i, m_i = attn_fwd_inner_prefix_unpipelined(
                            acc,
                            l_i,
                            m_i,
                            pfx_q,
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
                            ENABLE_PREFIX_UNMASKED,
                            BLOCK_M,
                            BLOCK_N,
                            BLOCK_DMODEL,
                            BLOCK_DV,
                            kt_async_layout,
                            v_async_layout,
                            layouts.pfx_kt_dot_layout,
                            layouts.pfx_p_dot_layout,
                            layouts.pfx_v_dot_layout,
                            layouts.mma_layout,
                            layouts.mma_offs_n_col,
                        )

                cdna4_async.wait_group(0)

                # FP8 smem transition: prefix smem was allocated with FP8 dtype
                # + 1024-byte padding; extend phase runs BF16 MFMA, so release
                # the prefix smem via _keep_alive (kernel-local dealloc) and
                # re-allocate smem with BF16 dtype + standard 512-byte padding.
                # IS_FP8=False skips this entirely (single-phase smem).
                if IS_FP8:
                    kt_smem._keep_alive()
                    v_smem._keep_alive()
                    kt_smem = gl.allocate_shared_memory(
                        Q_Extend.dtype.element_ty,
                        [_EXT_NS, BLOCK_DMODEL, _EXT_N],
                        layout=ext_kt_smem_layout,
                    )
                    v_smem = gl.allocate_shared_memory(
                        Q_Extend.dtype.element_ty,
                        [_EXT_NS, _EXT_N, BLOCK_DV],
                        layout=ext_v_smem_layout,
                    )
                    for _s in gl.static_range(_EXT_NS):
                        v_zero = gl.zeros(
                            [_EXT_N, BLOCK_DV],
                            dtype=Q_Extend.dtype.element_ty,
                            layout=ext_v_async_layout,
                        )
                        v_smem.index(_s).store(v_zero)
                    gl.barrier()

                # ===---- EXTEND hotloop (4w sw-pipeline) ---------------===
                # Walks the causal/SWA-clamped extend KV range in two phases,
                # using the hoisted n_full_blocks / swa_skip_n_blocks / bases:
                #
                #   [0, n_full_blocks)                       -- UNMASKED BULK
                #       All rows fully in range, no per-step softmax mask.
                #   [max(n_full_blocks, swa_skip_n_blocks),
                #                            n_extend_blocks) -- MASKED TAIL
                #       Causal diagonal / SWA edge / partial BLOCK_N block.
                #
                # Each phase has two helpers selected at compile time by the
                # number of blocks at runtime:
                #   >= _EXT_NS  -> pipelined (async DMA, NS-deep prefetch)
                #   <  _EXT_NS  -> unpipelined (synchronous, one K/V at a time)
                # so short runs skip the DMA setup cost. On FP8 _EXT_N/_EXT_NS
                # may differ from BLOCK_N/NUM_STAGES (extend can run a
                # different tile size from the wider FP8 prefix K tile);
                # on BF16 they're always equal.
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
                        _EXT_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        _EXT_NS,
                        ext_kt_async_layout,
                        ext_v_async_layout,
                        layouts.kt_dot_layout,
                        layouts.p_dot_layout,
                        layouts.v_dot_layout,
                        layouts.mma_layout,
                        layouts.mma_offs_n_col,
                        layouts.mma_offs_m_row,
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
                        _EXT_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        ext_kt_async_layout,
                        ext_v_async_layout,
                        layouts.kt_dot_layout,
                        layouts.p_dot_layout,
                        layouts.v_dot_layout,
                        layouts.mma_layout,
                        layouts.mma_offs_n_col,
                        layouts.mma_offs_m_row,
                    )
                # --- MASKED TAIL ---
                # Start at max(n_full_blocks, swa_skip_n_blocks): n_full_blocks
                # is where the unmasked bulk ended, swa_skip_n_blocks fast-
                # forwards past blocks that lie entirely outside the SWA
                # window (no-op when SWA is inactive). If both are zero we
                # mask the whole extend range, which is what the D>=256 /
                # custom-mask / mixed-SWA paths request above.
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
                        _EXT_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        _EXT_NS,
                        ext_kt_async_layout,
                        ext_v_async_layout,
                        layouts.kt_dot_layout,
                        layouts.p_dot_layout,
                        layouts.v_dot_layout,
                        layouts.mma_layout,
                        layouts.mma_offs_n_col,
                        layouts.mma_offs_m_row,
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
                        _EXT_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        ext_kt_async_layout,
                        ext_v_async_layout,
                        layouts.kt_dot_layout,
                        layouts.p_dot_layout,
                        layouts.v_dot_layout,
                        layouts.mma_layout,
                        layouts.mma_offs_n_col,
                        layouts.mma_offs_m_row,
                    )

            else:
                # ---- 4w serial ----
                # 4-warp, synchronous (NUM_STAGES=1 or D<128). K/V are loaded
                # into smem via blocked-layout gl.load, no DMA prefetch.
                # Handles both bulk and masked tail with one helper (serial
                # helper masks per step). SMEM uses the same SwizzledShared
                # layout across BF16 / FP8; FP8 allocates with PFX_SMEM_TY
                # for the prefix, then transitions to BF16 for extend.
                kt_blocked_layout: gl.constexpr = make_serial_kt_blocked(num_warps)
                kt_serial_smem_layout: gl.constexpr = SERIAL_KT_SMEM
                v_serial_smem_layout: gl.constexpr = SERIAL_V_SMEM
                q_smem_layout: gl.constexpr = SERIAL_Q_SMEM

                kt_serial_smem = gl.allocate_shared_memory(
                    layouts.PFX_SMEM_TY,
                    [BLOCK_DMODEL, BLOCK_N],
                    layout=kt_serial_smem_layout,
                )
                v_serial_smem = gl.allocate_shared_memory(
                    layouts.PFX_SMEM_TY,
                    [BLOCK_N, BLOCK_DV],
                    layout=v_serial_smem_layout,
                )
                q_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [BLOCK_M, BLOCK_DMODEL],
                    layout=q_smem_layout,
                )

                q_smem.store(q)
                q_dot = q_smem.load(layouts.q_dot_layout)
                if IS_FP8:
                    pfx_q = gl.convert_layout(q.to(tl.float8e4nv), layouts.pfx_q_dot_layout)
                else:
                    pfx_q = gl.convert_layout(q, layouts.pfx_q_dot_layout)

                if pfx_seq_len > 0:
                    acc, l_i, m_i = attn_fwd_inner_prefix_serial_4w(
                        acc,
                        l_i,
                        m_i,
                        pfx_q,
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
                        ENABLE_PREFIX_UNMASKED,
                        BLOCK_M,
                        BLOCK_N,
                        BLOCK_DMODEL,
                        BLOCK_DV,
                        kt_blocked_layout,
                        layouts.blocked_layout,
                        layouts.pfx_kt_dot_layout,
                        layouts.pfx_p_dot_layout,
                        layouts.pfx_v_dot_layout,
                        layouts.mma_layout,
                        layouts.mma_offs_n_col,
                    )

                # FP8 smem transition (serial path): release the FP8 prefix
                # smem via _keep_alive and re-allocate with BF16 extend dtype.
                # Same SwizzledSharedLayout (path-invariant); only the
                # element type changes. IS_FP8=False skips this.
                if IS_FP8:
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

                # ===---- EXTEND hotloop (4w serial) ----------------------===
                # Same two-phase structure as the pipelined path, but there
                # is only one helper: attn_fwd_inner_extend_serial_4w masks
                # per step, so we reuse it for both the unmasked bulk and
                # the masked tail and pick the two ranges via (start, end):
                #   [0, n_full_blocks)              -- bulk   (IS_MASKED=False)
                #   [masked_start, n_extend_blocks) -- tail   (IS_MASKED=True)
                # swa_skip_n_blocks fast-forwards the tail past any blocks
                # fully outside the SWA window (no-op when SWA is inactive).
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
                        layouts.blocked_layout,
                        layouts.kt_dot_layout,
                        layouts.p_dot_layout,
                        layouts.v_dot_layout,
                        layouts.mma_layout,
                        layouts.mma_offs_n_col,
                        layouts.mma_offs_m_row,
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
                        layouts.blocked_layout,
                        layouts.kt_dot_layout,
                        layouts.p_dot_layout,
                        layouts.v_dot_layout,
                        layouts.mma_layout,
                        layouts.mma_offs_n_col,
                        layouts.mma_offs_m_row,
                    )

        else:
            # ---- 8w pingpong ----
            # 8-warp ping-pong: 4 warps issue async DMA loads while the other
            # 4 warps MFMA on the previous tile, alternating each iteration.
            # Uniform across BLOCK_DMODEL={64, 128, 256}: layout factories
            # below parametrize on BLOCK_DMODEL directly. FP8 path flips
            # prefix dot operands + prefix smem padding via `layouts`; on
            # 8w pingpong EXT_BLOCK_N always equals BLOCK_N (extend reuses
            # the same tile size), so we only need an FP8 DLL swap.
            kt_offset_bases: gl.constexpr = make_kt_offset_bases(BLOCK_DMODEL, BLOCK_N)
            v_offset_bases: gl.constexpr = prefix_v_offset_bases(layouts, num_warps, BLOCK_DV, BLOCK_N)
            kt_async_layout: gl.constexpr = prefix_kt_dll(layouts, num_warps, BLOCK_DMODEL, BLOCK_N)
            v_async_layout: gl.constexpr = prefix_v_dll(layouts, num_warps, BLOCK_DMODEL, BLOCK_DV, BLOCK_N)

            kt_async_smem_layout: gl.constexpr = prefix_kt_smem_layout(layouts, BLOCK_DMODEL, BLOCK_N, kt_offset_bases)
            v_async_smem_layout: gl.constexpr = prefix_v_smem_layout(layouts, BLOCK_N, BLOCK_DV, v_offset_bases)

            # Extend-phase layouts (FP8 only; BF16 reuses prefix layouts).
            # The 8w pingpong extend shares the FP8 offset-bases ladder
            # (so the smem transition doesn't change offset layouts) but
            # swaps the DLLs to BF16-friendly variants.
            if IS_FP8:
                ext_kt_async_layout: gl.constexpr = make_kt_dll(num_warps, BLOCK_DMODEL, BLOCK_N)
                ext_v_async_layout: gl.constexpr = make_fp8_extend_v_dll(num_warps, BLOCK_DV, BLOCK_N)
                ext_kt_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_DMODEL, BLOCK_N], kt_offset_bases, [[512, 16]])
                ext_v_async_smem_layout: gl.constexpr = make_padded_smem([BLOCK_N, BLOCK_DV], v_offset_bases, [[512, 16]])
            else:
                ext_kt_async_layout: gl.constexpr = kt_async_layout
                ext_v_async_layout: gl.constexpr = v_async_layout

            kt_smem = gl.allocate_shared_memory(
                layouts.PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                layout=kt_async_smem_layout,
            )
            v_smem = gl.allocate_shared_memory(
                layouts.PFX_SMEM_TY,
                [NUM_STAGES, BLOCK_N, BLOCK_DV],
                layout=v_async_smem_layout,
            )

            # Zero-fill V smem before the first async load: FP8 always
            # (the prefix -> BF16-smem transition reads all NUM_STAGES
            # slots even on short prefixes), BF16 only for valid tiles
            # (skipped tiles bail before consuming the smem).
            if IS_FP8 or is_valid_tile:
                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DV],
                        dtype=layouts.PFX_SMEM_TY,
                        layout=v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

            q_dot = gl.convert_layout(q, layouts.q_dot_layout)
            if IS_FP8:
                pfx_q = gl.convert_layout(q.to(tl.float8e4nv), layouts.pfx_q_dot_layout)
            else:
                pfx_q = gl.convert_layout(q, layouts.pfx_q_dot_layout)

            # Prefix dispatch (8w pingpong): always pass the full prefix
            # length; the helper floor-aligns to BLOCK_N internally and
            # runs a one-block masked tail when the length isn't a
            # multiple of BLOCK_N. On the persistent path pfx_seq_len is
            # BLOCK_N-aligned (split-K partitioning), so the tail branch
            # is a runtime no-op.
            n_full_prefix = pfx_seq_len // BLOCK_N
            # Scalar-mask prefix on the persistent path (geomean 0.82x,
            # up to 0.47x on big-ragged D=128); pipelined load-gate on
            # basic (scalar and pipelined are within 0.3% on basic --
            # bench_scalar_mask.py Apr 2026).
            _use_scalar_mask: gl.constexpr = IS_PERSISTENT
            if n_full_prefix >= NUM_STAGES:
                acc, l_i, m_i = attn_fwd_inner_prefix_pingpong_8w(
                    acc,
                    l_i,
                    m_i,
                    pfx_q,
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
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DV,
                    NUM_STAGES,
                    kt_async_layout,
                    v_async_layout,
                    layouts.pfx_kt_dot_layout,
                    layouts.pfx_p_dot_layout,
                    layouts.pfx_v_dot_layout,
                    layouts.mma_layout,
                    layouts.mma_offs_n_col,
                    _use_scalar_mask,
                )
            elif pfx_seq_len > 0:
                acc, l_i, m_i = attn_fwd_inner_prefix_unpipelined(
                    acc,
                    l_i,
                    m_i,
                    pfx_q,
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
                    ENABLE_PREFIX_UNMASKED,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_DMODEL,
                    BLOCK_DV,
                    kt_async_layout,
                    v_async_layout,
                    layouts.pfx_kt_dot_layout,
                    layouts.pfx_p_dot_layout,
                    layouts.pfx_v_dot_layout,
                    layouts.mma_layout,
                    layouts.mma_offs_n_col,
                )

            # FP8 smem transition (8w pingpong): release the FP8 prefix
            # smem via _keep_alive and re-allocate with BF16 extend dtype
            # + standard 512-byte padding. IS_FP8=False skips this entirely.
            if IS_FP8:
                kt_smem._keep_alive()
                v_smem._keep_alive()
                kt_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_DMODEL, BLOCK_N],
                    layout=ext_kt_async_smem_layout,
                )
                v_smem = gl.allocate_shared_memory(
                    Q_Extend.dtype.element_ty,
                    [NUM_STAGES, BLOCK_N, BLOCK_DV],
                    layout=ext_v_async_smem_layout,
                )
                for _s in gl.static_range(NUM_STAGES):
                    v_zero = gl.zeros(
                        [BLOCK_N, BLOCK_DV],
                        dtype=Q_Extend.dtype.element_ty,
                        layout=ext_v_async_layout,
                    )
                    v_smem.index(_s).store(v_zero)
                gl.barrier()

            # ===---- EXTEND hotloop (8w pingpong) -----------------------===
            # Same two-phase layout as the 4w sw-pipeline path:
            #   [0, n_full_blocks)                       -- UNMASKED BULK
            #   [max(n_full_blocks, swa_skip_n_blocks),
            #                            n_extend_blocks) -- MASKED TAIL
            # Each phase picks pingpong_8w (>=NUM_STAGES) or unpipelined
            # (<NUM_STAGES), so short runs skip the async DMA setup cost.
            #
            # --- UNMASKED BULK ---
            if n_full_blocks >= NUM_STAGES:
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
                    ext_kt_async_layout,
                    ext_v_async_layout,
                    layouts.kt_dot_layout,
                    layouts.p_dot_layout,
                    layouts.v_dot_layout,
                    layouts.mma_layout,
                    layouts.mma_offs_n_col,
                    layouts.mma_offs_m_row,
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
                    ext_kt_async_layout,
                    ext_v_async_layout,
                    layouts.kt_dot_layout,
                    layouts.p_dot_layout,
                    layouts.v_dot_layout,
                    layouts.mma_layout,
                    layouts.mma_offs_n_col,
                    layouts.mma_offs_m_row,
                )
            # --- MASKED TAIL ---
            masked_start = tl.maximum(n_full_blocks, swa_skip_n_blocks)
            remaining_blocks = n_extend_blocks - masked_start
            if remaining_blocks >= NUM_STAGES:
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
                    ext_kt_async_layout,
                    ext_v_async_layout,
                    layouts.kt_dot_layout,
                    layouts.p_dot_layout,
                    layouts.v_dot_layout,
                    layouts.mma_layout,
                    layouts.mma_offs_n_col,
                    layouts.mma_offs_m_row,
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
                    ext_kt_async_layout,
                    ext_v_async_layout,
                    layouts.kt_dot_layout,
                    layouts.p_dot_layout,
                    layouts.v_dot_layout,
                    layouts.mma_layout,
                    layouts.mma_offs_n_col,
                    layouts.mma_offs_m_row,
                )

        # sinks
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
            po_val = gl.convert_layout(acc_normed, layouts.blocked_layout)
            gl.store(po_ptrs, po_val, mask=po_mask)

            pl_base = partial_lse + split_idx * BLOCK_M
            pl_ptrs = pl_base + offs_m
            pl_mask = (cur_block_m * BLOCK_M + offs_m) < orig_seq_len_extend
            lse_val = gl.convert_layout(lse, layouts.offs_m_layout)
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
                out_r = gl.convert_layout(r_acc, layouts.blocked_layout).to(O_Extend.dtype.element_ty)
                cdna4_buffer_store(out_r, r_o_base, r_o_offsets, mask=r_o_mask)
        else:
            l_recip = 1.0 / l_i
            acc = acc * l_recip[:, None]
            acc = acc * v_scale

            o_base = O_Extend + cur_seq_q_start_idx * stride_obs + cur_head * stride_oh
            o_offsets = ((cur_block_m * BLOCK_M + offs_m[:, None]) * stride_obs + offs_dv[None, :]).to(tl.int32)
            o_mask = (cur_block_m * BLOCK_M + offs_m[:, None]) < seq_len_extend
            out = gl.convert_layout(acc, layouts.blocked_layout).to(O_Extend.dtype.element_ty)
            cdna4_buffer_store(out, o_base, o_offsets, mask=o_mask)

        if IS_PERSISTENT:
            tile_idx += total_programs
        else:
            tile_idx = 1
