import torch
import triton
import triton.language as tl


@triton.jit
def concat_and_cast_mha_k_kernel(
    k_ptr,
    k_nope_ptr,
    k_rope_ptr,
    head_cnt: tl.constexpr,
    k_stride0: tl.constexpr,
    k_stride1: tl.constexpr,
    nope_stride0: tl.constexpr,
    nope_stride1: tl.constexpr,
    rope_stride0: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
):
    pid_loc = tl.program_id(0)
    head_range = tl.arange(0, head_cnt)

    k_head_ptr = k_ptr + pid_loc * k_stride0 + head_range[:, None] * k_stride1

    nope_offs = tl.arange(0, nope_dim)

    src_nope_ptr = (
        k_nope_ptr
        + pid_loc * nope_stride0
        + head_range[:, None] * nope_stride1
        + nope_offs[None, :]
    )
    dst_nope_ptr = k_head_ptr + nope_offs[None, :]

    src_nope = tl.load(src_nope_ptr)
    tl.store(dst_nope_ptr, src_nope)

    rope_offs = tl.arange(0, rope_dim)
    src_rope_ptr = k_rope_ptr + pid_loc * rope_stride0 + rope_offs[None, :]
    dst_rope_ptr = k_head_ptr + nope_dim + rope_offs[None, :]
    src_rope = tl.load(src_rope_ptr)
    tl.store(dst_rope_ptr, src_rope)


def concat_and_cast_mha_k_triton(
    k: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
):
    # The source data type will be implicitly converted to the target data type.
    assert (
        len(k.shape) == 3 and len(k_nope.shape) == 3 and len(k_rope.shape) == 3
    ), f"shape should be 3d, but got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[0] == k_nope.shape[0] and k.shape[0] == k_rope.shape[0]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[1] == k_nope.shape[1] and 1 == k_rope.shape[1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"
    assert (
        k.shape[-1] == k_nope.shape[-1] + k_rope.shape[-1]
    ), f"invalid shape, got {k.shape=}, {k_nope.shape=}, {k_rope.shape=}"

    nope_dim = k_nope.shape[-1]
    rope_dim = k_rope.shape[-1]
    grid = (k.shape[0],)

    concat_and_cast_mha_k_kernel[grid](
        k,
        k_nope,
        k_rope,
        k.shape[1],
        k.stride(0),
        k.stride(1),
        k_nope.stride(0),
        k_nope.stride(1),
        k_rope.stride(0),
        nope_dim,
        rope_dim,
    )


@triton.jit
def reshape_and_cache_flash(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    swa_slot_mapping_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_stride,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SWA: tl.constexpr,
    USE_SCALE: tl.constexpr,
):
    """
    Triton kernel for reshaping per-token K/V tensors into paged KV cache layout.

    Source layout:
        key/value: [num_tokens, num_heads, head_size]

    Target cache layout:
        cache: [num_blocks, block_size, num_heads, head_size]

    Each Triton program instance handles:
        - one token (program_id(0))
        - one block of heads (program_id(1))

    Features:
        - optional SWA slot remapping
        - optional FP8 scale dequantization before cache write

    Args:
        key_ptr: Pointer to source key tensor.
        value_ptr: Pointer to source value tensor.
        key_cache_ptr: Pointer to destination key cache tensor.
        value_cache_ptr: Pointer to destination value cache tensor.
        slot_mapping_ptr: Maps token -> cache slot.
        swa_slot_mapping_ptr: Optional second-stage slot remap for SWA mode.
        k_scale_ptr: Optional key scaling factor pointer.
        v_scale_ptr: Optional value scaling factor pointer.
        block_stride: Stride between cache blocks.
        key_stride: Stride between source key tokens.
        value_stride: Stride between source value tokens.
        num_heads: Number of attention heads.
        head_size: Hidden dimension per head.
        block_size: Number of slots per cache block.
        HEAD_BLOCK: Number of heads processed per program.
        BLOCK_D: Vectorized dimension size (power-of-2 padded).
        HAS_SWA: Enable SWA remapping.
        USE_SCALE: Enable scale division before storing.
    """

    # ----------------------------------
    # program ids
    # pid0 = token
    # pid1 = head block
    # ----------------------------------
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    # ----------------------------------
    # slot mapping
    # ----------------------------------
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    if HAS_SWA:
        slot_idx = tl.load(swa_slot_mapping_ptr + slot_idx)

    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    # ----------------------------------
    # head range
    # ----------------------------------
    head_idx = head_block_idx * HEAD_BLOCK + tl.arange(0, HEAD_BLOCK)

    head_mask = head_idx < num_heads

    dim_idx = tl.arange(0, BLOCK_D)

    # shape = [HEAD_BLOCK, BLOCK_D]
    offs = head_idx[:, None] * head_size + dim_idx[None, :]

    mask = head_mask[:, None] & (dim_idx[None, :] < head_size)

    # ----------------------------------
    # source load
    # ----------------------------------
    src_key = token_idx * key_stride + offs
    src_value = token_idx * value_stride + offs

    k = tl.load(key_ptr + src_key, mask=mask)
    v = tl.load(value_ptr + src_value, mask=mask)

    # ----------------------------------
    # optional scale
    # ----------------------------------
    if USE_SCALE:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)

        k = k / k_scale
        v = v / v_scale

    # ----------------------------------
    # target layout
    # [block_idx, block_offset, head, dim]
    # ----------------------------------
    tgt = block_idx * block_stride + block_offset * num_heads * head_size + offs

    tl.store(key_cache_ptr + tgt, k, mask=mask)
    tl.store(value_cache_ptr + tgt, v, mask=mask)


def launch_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping=None,
    k_scale=None,
    v_scale=None,
):
    """
    Launch wrapper for reshape_and_cache_flash Triton kernel.

    This wrapper prepares launch configuration and dispatches the Triton kernel
    that writes token-major K/V tensors into paged KV cache layout.

    Args:
        key: Source key tensor [num_tokens, num_heads, head_size]
        value: Source value tensor [num_tokens, num_heads, head_size]
        key_cache: Destination key cache [num_blocks, block_size, num_heads, head_size]
        value_cache: Destination value cache [num_blocks, block_size, num_heads, head_size]
        slot_mapping: Token-to-cache slot mapping
        swa_slot_mapping: Optional SWA remapping table
        k_scale: Optional key scaling factor
        v_scale: Optional value scaling factor
    """

    num_tokens = key.shape[0]
    num_heads = key.shape[1]
    head_size = key.shape[2]

    HEAD_BLOCK = 4

    BLOCK_D = triton.next_power_of_2(head_size)

    grid = (
        num_tokens,
        triton.cdiv(num_heads, HEAD_BLOCK),
    )

    reshape_and_cache_flash[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        k_scale if k_scale is not None else key,
        v_scale if v_scale is not None else key,
        key_cache.stride(0),
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        key_cache.shape[1],
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
        HAS_SWA=(swa_slot_mapping is not None),
        USE_SCALE=(k_scale is not None),
    )


@triton.jit
def concat_and_cast_q_fp8_pad_kernel(
    qpad_ptr,  # [num_tokens, pad_heads, NOPE+ROPE] fp8 (dst; only [:, :H, :] written)
    q_nope_ptr,  # [num_tokens, H, NOPE] bf16
    q_rope_ptr,  # [num_tokens, H, ROPE] bf16
    qpad_s0,
    qpad_s1,
    nope_s0,
    nope_s1,
    rope_s0,
    rope_s1,
    H: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
):
    # One program per token: write the H active heads of the padded fp8 q buffer,
    # fusing the bf16->fp8 cast (on store) with the nope/rope concat.  Bit-exact vs the
    # two strided copy_() it replaces; ~3.7x faster because copy_ into the
    # 64-head-padded buffer is strided (~4.5x off memory-bound).  Strides are passed in,
    # so q_nope/q_rope may be views of a [T, H, NOPE+ROPE] q (head-stride != last-dim).
    pid = tl.program_id(0)
    hr = tl.arange(0, H)
    qpad_head = qpad_ptr + pid * qpad_s0 + hr[:, None] * qpad_s1
    no = tl.arange(0, NOPE)
    src_n = tl.load(q_nope_ptr + pid * nope_s0 + hr[:, None] * nope_s1 + no[None, :])
    tl.store(qpad_head + no[None, :], src_n)
    ro = tl.arange(0, ROPE)
    src_r = tl.load(q_rope_ptr + pid * rope_s0 + hr[:, None] * rope_s1 + ro[None, :])
    tl.store(qpad_head + NOPE + ro[None, :], src_r)


def concat_and_cast_q_fp8_pad(q_fp8_pad, q_nope, q_rope, num_heads):
    """fused bf16->fp8 concat-cast of q_nope/q_rope into the active
    [:, :num_heads, :] slice of the padded fp8 q buffer.  Bit-exact replacement for the
    two strided converting copy_() in the Q8KV8 prefill q-prep, ~3.7x faster.  Requires
    num_heads / nope_dim / rope_dim to be powers of two (always true for DeepSeek: 128
    heads / any TP, 512 nope, 64 rope)."""
    num_tokens = q_nope.shape[0]
    nope_dim = q_nope.shape[-1]
    rope_dim = q_rope.shape[-1]
    concat_and_cast_q_fp8_pad_kernel[(num_tokens,)](
        q_fp8_pad,
        q_nope,
        q_rope,
        q_fp8_pad.stride(0),
        q_fp8_pad.stride(1),
        q_nope.stride(0),
        q_nope.stride(1),
        q_rope.stride(0),
        q_rope.stride(1),
        H=num_heads,
        NOPE=nope_dim,
        ROPE=rope_dim,
    )


@triton.jit
def absorbed_bmm_concat_cast_q_fp8_kernel(
    qout_ptr,  # [num_tokens, pad_heads, N+ROPE] fp8 (dst; only [:, :H, :] written)
    a_ptr,  # q_nope (pre-absorb) [num_tokens, H, K] bf16
    b_ptr,  # w_kc [H, K, N] bf16 (any strides; typically N-major)
    rope_ptr,  # q_rope (post-rope) [num_tokens, H, ROPE] bf16
    T,  # num_tokens (runtime; masked)
    qout_s0,
    qout_s1,
    a_s0,
    a_s1,
    b_s0,
    b_s1,
    b_s2,
    rope_s0,
    rope_s1,
    K: tl.constexpr,
    N: tl.constexpr,
    ROPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_MODE: tl.constexpr,
):
    # One program per (token-block, head): q_out[m, h, :N] = fp8(bf16(fp32(
    # q_nope[m, h, :K] @ w_kc[h, :K, :N]))) and q_out[m, h, N:] =
    # fp8(q_rope[m, h, :ROPE]).  This makes q "born fp8": the absorbed bmm,
    # the nope/rope concat, and the bf16->fp8 cast collapse into one kernel,
    # so neither the bf16 q_nope_out ([H, T, N], written by cublas and re-read
    # by the concat-cast) nor the standalone concat-cast launch exist anymore.
    #
    # K handling (K_MODE selects the codegen for the nope-gemm K dimension;
    # every mode keeps the same fp32-accumulator -> bf16 -> fp8 epilogue):
    #   0 "single": BLOCK_K == K.  Preload the whole [BLOCK_M, K] a-tile once,
    #     one tl.dot per N-block — identical codegen to the original
    #     power-of-2-only kernel (DeepSeek K=128).  For non-power-of-2 K this
    #     only compiles if the Triton build allows non-power-of-2 tl.arange
    #     (Triton <= 3.5.x does NOT: "arange's range must be a power of 2").
    #   1 "loop": split-K loop, K % BLOCK_K == 0, BLOCK_K power of 2 >= 16
    #     (e.g. K=192 with BLOCK_K=64 -> 3 iterations).  The a-tile is
    #     re-loaded per (N-block, K-block); slices are L1/L2-resident after
    #     the first N-block, but the load/dot interleave costs bandwidth
    #     (measured ~1372 GB/s vs ~2x that for mode 0 at K=128).
    #   2 "two_dot": K = BLOCK_K + (K - BLOCK_K), both power-of-2 halves
    #     (192 = 128 + 64).  Both a-tiles preload once before the N-loop;
    #     each N-block issues two chained tl.dot into one fp32 accumulator.
    #     No K-loop, no a re-reads — the direct generalization of mode 0.
    #   3 "three_dot": K = 3 * BLOCK_K (192 = 3 x 64).  Same as mode 2 with
    #     three preloaded a-tiles / three chained tl.dot per N-block; the
    #     hoisted-loads analogue of mode 1 (identical fp32 add order).
    #   4 "pad": BLOCK_K = next_pow2(K) > K, k-masked loads (zero fill).
    #     Single tl.dot per N-block; the padded zeros are exact fp32
    #     additive identities so the result matches a K-wide single dot,
    #     at the cost of BLOCK_K/K (e.g. 256/192 = 1.33x) extra MMA work.
    #
    # Rounding contract: the fp32 accumulator is rounded to bf16 first (the
    # same output rounding stage as the cublas bf16 bmm) and then converted
    # bf16->fp8 by the same implicit-store conversion the fused concat-cast
    # kernel uses.  The split-K accumulator stays fp32 across all K-blocks,
    # so the rounding stages are identical in both layouts.  The rope half is
    # a bit-exact copy of that kernel (loads the post-rope bf16, converts on
    # store).  The nope half is NOT guaranteed bit-exact vs the default path:
    # tl.dot accumulates fp32 in a different order than cublas, so last-ulp
    # fp32 differences can occasionally flip the bf16 (and hence fp8)
    # rounding.
    pid_m = tl.program_id(0)
    h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < T
    # token-row offsets in int64: T * row-stride can exceed int32 (e.g. 128
    # heads x 576 dims x tens of thousands of tokens).
    offs_m64 = offs_m.to(tl.int64)
    qout_head = qout_ptr + offs_m64[:, None] * qout_s0 + h * qout_s1
    a_row = a_ptr + offs_m64[:, None] * a_s0 + h * a_s1
    b_head = b_ptr + h * b_s0
    if K_MODE == 0:
        # single-dot path (original kernel): BLOCK_K == K, preload a once.
        offs_k = tl.arange(0, BLOCK_K)
        a = tl.load(a_row + offs_k[None, :], mask=m_mask[:, None], other=0.0)
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            b = tl.load(b_head + offs_k[:, None] * b_s1 + offs_n[None, :] * b_s2)
            acc = tl.dot(a, b)  # fp32 accumulator
            val = acc.to(tl.bfloat16)  # cublas-equivalent bf16 output rounding
            # implicit bf16 -> fp8 conversion on store (same as the concat-cast)
            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    elif K_MODE == 2:
        # two-dot preload: K split as BLOCK_K + (K - BLOCK_K), no K-loop.
        offs_k0 = tl.arange(0, BLOCK_K)
        offs_k1 = BLOCK_K + tl.arange(0, K - BLOCK_K)
        a0 = tl.load(a_row + offs_k0[None, :], mask=m_mask[:, None], other=0.0)
        a1 = tl.load(a_row + offs_k1[None, :], mask=m_mask[:, None], other=0.0)
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            b0 = tl.load(b_head + offs_k0[:, None] * b_s1 + offs_n[None, :] * b_s2)
            b1 = tl.load(b_head + offs_k1[:, None] * b_s1 + offs_n[None, :] * b_s2)
            acc = tl.dot(a0, b0)  # fp32 accumulator
            acc = tl.dot(a1, b1, acc)  # chained: stays fp32 across both dots
            val = acc.to(tl.bfloat16)  # cublas-equivalent bf16 output rounding
            # implicit bf16 -> fp8 conversion on store (same as the concat-cast)
            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    elif K_MODE == 3:
        # three-dot preload: K = 3 * BLOCK_K, a-tiles hoisted out of the N-loop.
        offs_k0 = tl.arange(0, BLOCK_K)
        offs_k1 = BLOCK_K + offs_k0
        offs_k2 = 2 * BLOCK_K + offs_k0
        a0 = tl.load(a_row + offs_k0[None, :], mask=m_mask[:, None], other=0.0)
        a1 = tl.load(a_row + offs_k1[None, :], mask=m_mask[:, None], other=0.0)
        a2 = tl.load(a_row + offs_k2[None, :], mask=m_mask[:, None], other=0.0)
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            b0 = tl.load(b_head + offs_k0[:, None] * b_s1 + offs_n[None, :] * b_s2)
            b1 = tl.load(b_head + offs_k1[:, None] * b_s1 + offs_n[None, :] * b_s2)
            b2 = tl.load(b_head + offs_k2[:, None] * b_s1 + offs_n[None, :] * b_s2)
            acc = tl.dot(a0, b0)  # fp32 accumulator
            acc = tl.dot(a1, b1, acc)
            acc = tl.dot(a2, b2, acc)  # same fp32 add order as the K_MODE=1 loop
            val = acc.to(tl.bfloat16)  # cublas-equivalent bf16 output rounding
            # implicit bf16 -> fp8 conversion on store (same as the concat-cast)
            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    elif K_MODE == 4:
        # padded single dot: BLOCK_K = next_pow2(K), zero-fill the k tail.
        offs_k = tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        a = tl.load(
            a_row + offs_k[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            b = tl.load(
                b_head + offs_k[:, None] * b_s1 + offs_n[None, :] * b_s2,
                mask=k_mask[:, None],
                other=0.0,
            )
            acc = tl.dot(a, b)  # fp32 accumulator (padded zeros add exactly 0)
            val = acc.to(tl.bfloat16)  # cublas-equivalent bf16 output rounding
            # implicit bf16 -> fp8 conversion on store (same as the concat-cast)
            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    else:
        # K_MODE == 1: split-K loop (K % BLOCK_K == 0, e.g. K=192, BLOCK_K=64).
        for nb in tl.static_range(N // BLOCK_N):
            offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for kb in tl.static_range(K // BLOCK_K):
                offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
                a = tl.load(a_row + offs_k[None, :], mask=m_mask[:, None], other=0.0)
                b = tl.load(b_head + offs_k[:, None] * b_s1 + offs_n[None, :] * b_s2)
                acc = tl.dot(a, b, acc)  # fp32 accumulator across K-blocks
            val = acc.to(tl.bfloat16)  # cublas-equivalent bf16 output rounding
            # implicit bf16 -> fp8 conversion on store (same as the concat-cast)
            tl.store(qout_head + offs_n[None, :], val, mask=m_mask[:, None])
    offs_r = tl.arange(0, ROPE)
    r = tl.load(
        rope_ptr + offs_m64[:, None] * rope_s0 + h * rope_s1 + offs_r[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    tl.store(qout_head + N + offs_r[None, :], r, mask=m_mask[:, None])


# Non-power-of-2-K variant used by variant="auto" (power-of-2 K always takes
# the single-dot fast path).  Set to the winner of the K=192 A/B in
# scripts/pr3_qprep_microbench.py; "loop" = the pre-A/B split-K behavior.
_AUTO_NONPOW2_VARIANT = "two_dot"


def absorbed_bmm_concat_cast_q_fp8(
    q_fp8_pad: "torch.Tensor",
    q_nope: "torch.Tensor",
    w_kc: "torch.Tensor",
    q_rope: "torch.Tensor",
    num_heads: int,
    block_m: int = 128,
    block_n: int = 64,
    variant: str = "auto",
    block_k: int = 0,
    num_warps: int = 8,
    num_stages: int = 0,
):
    """Fused absorbed-q bmm + nope/rope concat + bf16->fp8 cast ("born fp8" q).

    Replaces ``torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)``
    followed by ``concat_and_cast_q_fp8_pad`` on the Q8KV8 sparse-prefill
    path, writing the active ``[:, :num_heads, :]`` slice of the padded fp8 q
    buffer directly.  Inputs:

    * ``q_fp8_pad``: [num_tokens, pad_heads, N + ROPE] fp8_e4m3 destination.
    * ``q_nope``: [num_tokens, H, K] bf16 pre-absorb q (strided views OK).
    * ``w_kc``: [H, K, N] bf16 absorbed weight (any strides).
    * ``q_rope``: [num_tokens, H, ROPE] bf16 post-rope q (strided views OK).

    The rope half is bit-exact vs ``concat_and_cast_q_fp8_pad``.  The nope
    half keeps the same rounding stages (fp32 accum -> bf16 -> fp8) but a
    different fp32 accumulation order than cublas, so it is near- but not
    guaranteed bit-exact; keep this path behind
    ``SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q``.

    K (``qk_nope_head_dim``) supports any multiple of 16 in [16, 256].
    Power-of-2 K (DeepSeek 128) always takes the preload-once
    single-``tl.dot`` path.  For other K (GLM 192), ``variant`` selects the
    K-dimension codegen (every variant keeps the identical fp32 -> bf16 ->
    fp8 epilogue):

    * ``"auto"``: the current production choice (see
      ``_AUTO_NONPOW2_VARIANT``).
    * ``"loop"``: split-K accumulator loop, ``BLOCK_K`` = ``block_k`` or the
      largest power-of-2 divisor of K capped at 128 (192 -> 64 x 3).
    * ``"two_dot"``: preload a as two power-of-2 tiles (192 = 128 + 64), two
      chained ``tl.dot`` per N-block, no K-loop.
    * ``"three_dot"``: preload a as three K/3 tiles (192 = 3 x 64), three
      chained ``tl.dot`` per N-block; same fp32 add order as ``"loop"``.
    * ``"pad"``: single ``tl.dot`` with ``BLOCK_K`` = next_pow2(K) (192 ->
      256) and zero-masked k tails.
    * ``"single_k"``: single ``tl.dot`` with ``BLOCK_K`` == K.  Only
      compiles if the Triton build supports non-power-of-2 ``tl.arange``
      (Triton <= 3.5.x raises "arange's range must be a power of 2").

    ``block_m`` / ``block_n`` / ``num_warps`` / ``num_stages`` are tuning
    knobs for the microbench sweep (0 = Triton default for ``num_stages``).
    """
    num_tokens, _, k_dim = q_nope.shape
    n_dim = w_kc.shape[-1]
    rope_dim = q_rope.shape[-1]
    assert q_fp8_pad.dtype == torch.float8_e4m3fn
    assert q_nope.dtype == torch.bfloat16 and w_kc.dtype == torch.bfloat16
    assert q_rope.dtype == torch.bfloat16
    assert q_nope.shape[1] == num_heads and q_rope.shape[1] == num_heads
    assert w_kc.shape[0] == num_heads and w_kc.shape[1] == k_dim
    assert q_fp8_pad.shape[0] >= num_tokens and q_fp8_pad.shape[1] >= num_heads
    assert q_fp8_pad.shape[2] == n_dim + rope_dim
    # tl.arange / tl.dot constraints
    assert (
        k_dim % 16 == 0 and 16 <= k_dim <= 256
    ), "K must be a multiple of 16 in [16, 256]"
    assert (rope_dim & (rope_dim - 1)) == 0, "ROPE must be a power of two"
    assert n_dim % block_n == 0, "N must be a multiple of block_n"
    assert q_nope.stride(2) == 1 and q_rope.stride(2) == 1
    assert q_fp8_pad.stride(2) == 1
    # Resolve (K_MODE, BLOCK_K) from the variant; see the kernel's K-handling
    # comment for what each mode compiles to.
    if k_dim & (k_dim - 1) == 0:
        # power-of-2 K: every variant collapses to the single-dot fast path.
        k_mode, blk_k = 0, k_dim
    else:
        v = _AUTO_NONPOW2_VARIANT if variant == "auto" else variant
        if v == "loop":
            # Largest power-of-2 divisor of K, capped at 128 (K % 16 == 0
            # makes this >= 16), unless the caller pinned block_k.
            blk_k = block_k or min(k_dim & -k_dim, 128)
            assert (
                k_dim % blk_k == 0 and blk_k & (blk_k - 1) == 0 and blk_k >= 16
            ), "loop needs BLOCK_K a power-of-2 divisor of K >= 16"
            k_mode = 1
        elif v == "two_dot":
            blk_k = 1 << (k_dim.bit_length() - 1)  # largest power of 2 < K
            k1 = k_dim - blk_k
            assert (
                k1 & (k1 - 1) == 0 and k1 >= 16
            ), "two_dot needs K = pow2 + pow2 with both halves >= 16"
            k_mode = 2
        elif v == "three_dot":
            blk_k = k_dim // 3
            assert (
                k_dim % 3 == 0 and blk_k & (blk_k - 1) == 0 and blk_k >= 16
            ), "three_dot needs K = 3 * pow2 with pow2 >= 16"
            k_mode = 3
        elif v == "pad":
            blk_k = 1 << k_dim.bit_length()  # next power of 2 above K
            k_mode = 4
        elif v == "single_k":
            # Non-power-of-2 BLOCK_K == K: compiles only on Triton builds
            # that allow non-power-of-2 tl.arange (not 3.5.x).
            blk_k = k_dim
            k_mode = 0
        else:
            raise ValueError(f"unknown absorbed-bmm K variant: {variant!r}")
    extra = {"num_stages": num_stages} if num_stages else {}
    grid = (triton.cdiv(num_tokens, block_m), num_heads)
    absorbed_bmm_concat_cast_q_fp8_kernel[grid](
        q_fp8_pad,
        q_nope,
        w_kc,
        q_rope,
        num_tokens,
        q_fp8_pad.stride(0),
        q_fp8_pad.stride(1),
        q_nope.stride(0),
        q_nope.stride(1),
        w_kc.stride(0),
        w_kc.stride(1),
        w_kc.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        K=k_dim,
        N=n_dim,
        ROPE=rope_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=blk_k,
        K_MODE=k_mode,
        num_warps=num_warps,
        **extra,
    )


@triton.jit
def q8kv8_topk_length_backscan_kernel(
    indices_ptr,
    out_ptr,
    stride_row,
    topk,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = indices_ptr + row * stride_row
    off = topk
    length = 1
    found = 0
    while (found == 0) & (off > 0):
        off -= BLOCK
        idx = off + tl.arange(0, BLOCK)
        vals = tl.load(base + idx)
        pos = tl.max(tl.where(vals >= 0, idx, -1), axis=0)
        found = tl.where(pos >= 0, 1, found)
        length = tl.where(pos >= 0, pos + 1, length)
    tl.store(out_ptr + row, length)


def q8kv8_topk_length_from_indices(indices: torch.Tensor) -> torch.Tensor:
    """Per-row valid-topk count = last non-negative position + 1 (min 1).

    ``indices``: [s_q, topk] int32 topk output whose pad slots are -1.
    Backward block scan per row: the loop exits at the first block holding a
    valid entry, so the cost is proportional to the trailing pad run — one
    block (~topk/4 elements) for rows with a full topk, which dominate long
    contexts. Semantics match the unfused ``(indices >= 0) * ramp).amax``
    derivation exactly, including all-pad rows (length 1: one pad-only block
    keeps the kernel on its clamp+mask path, contributing zero).
    """
    s_q, topk = indices.shape
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    out = torch.empty(s_q, dtype=torch.int32, device=indices.device)
    block = 512 if topk % 512 == 0 else (256 if topk % 256 == 0 else 128)
    q8kv8_topk_length_backscan_kernel[(s_q,)](
        indices,
        out,
        indices.stride(0),
        topk,
        BLOCK=block,
    )
    return out


# ---------------------------------------------------------------------------
# Decode Context Parallel (DCP) helpers.
#
# Not part of upstream main (PR #26000 centralized the other Triton utility
# kernels into triton_ops/*). These three live here because they are DCP-only:
#   - create_triton_kv_indices_for_dcp_triton: per-rank local KV indices
#   - get_dcp_lens: per-rank visible KV length
#   - cp_lse_ag_out_rs: merge DCP partial attention via natural-log LSE
# ---------------------------------------------------------------------------
