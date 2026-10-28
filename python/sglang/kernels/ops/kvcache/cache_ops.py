import triton
import triton.language as tl


@triton.jit
def seg_plan_program(
    pid,
    seq_lens_ptr,
    plan_ptr,
    num_seqs,
    num_work,
    target_wgs,
    seg_max,
    BLOCK_B: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Body of one program of the vattn split-KV segment plan (sequence ``pid``).

    Every program recomputes the (cheap) batch-wide plan and writes its own slice of the
    work list; returns this sequence's segment count.
    """
    b = tl.arange(0, BLOCK_B)
    bm = b < num_seqs
    slen = tl.load(seq_lens_ptr + b, mask=bm, other=0).to(tl.int32)
    nt = (slen + 15) // 16
    total = tl.sum(nt, axis=0)
    mx = tl.max(nt, axis=0)
    # smallest T (tiles per segment) with sum_b ceil(nt_b / T) <= target_wgs and max_b ceil(nt_b / T) <= seg_max
    lo = tl.maximum(
        tl.maximum(
            (total + target_wgs - 1) // target_wgs,
            (mx + seg_max - 1) // seg_max,
        ),
        16,  # min KV tiles per segment (= 256 tokens): shorter segments cost more in fp32 partials than they save
    )
    slack = target_wgs - num_seqs
    hi = tl.where(
        slack > 0,
        (total + tl.maximum(slack, 1) - 1) // tl.maximum(slack, 1),
        lo,
    )
    hi = tl.maximum(hi, lo)
    for _ in range(16):
        mid = (lo + hi) // 2
        fits = tl.sum((nt + mid - 1) // mid, axis=0) <= target_wgs
        hi = tl.where(fits, mid, hi)
        lo = tl.where(fits, lo, mid + 1)
    T = hi
    nseg = (nt + T - 1) // T
    ends = tl.cumsum(nseg, axis=0)
    tot = tl.sum(nseg, axis=0)
    my_n = tl.sum(tl.where(b == pid, nseg, 0), axis=0)
    my_start = tl.sum(tl.where(b == pid, ends, 0), axis=0) - my_n
    if pid == 0:
        tl.store(plan_ptr, T)
    w = tl.arange(0, BLOCK_W)
    tl.store(plan_ptr + 1 + my_start + w, (pid << 16) | w, mask=w < my_n)
    idle = tot + pid + w * num_seqs  # idle tail, strided over programs
    tl.store(
        plan_ptr + 1 + idle,
        tl.full((BLOCK_W,), -1, tl.int32),
        mask=idle < num_work,
    )
    return my_n


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
    seq_lens_ptr,
    plan_ptr,
    tok_nseg_ptr,
    block_stride,
    value_block_stride,
    cache_token_stride: tl.constexpr,
    cache_head_stride: tl.constexpr,
    value_cache_token_stride: tl.constexpr,
    value_cache_head_stride: tl.constexpr,
    key_stride,
    value_stride,
    num_heads,
    head_size,
    block_size,
    num_seqs,
    num_work,
    target_wgs,
    seg_max,
    HEAD_BLOCK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SWA: tl.constexpr,
    USE_SCALE: tl.constexpr,
    DO_PLAN: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """
    Triton kernel for reshaping per-token K/V tensors into paged KV cache layout.

    Source layout:
        key/value: [num_tokens, num_heads, head_size]

    Target cache layout:
        cache: [num_blocks, block_size, num_heads, head_size]
        Token and head strides may describe a view of physical HND storage.

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
        block_stride: Stride between key cache blocks.
        value_block_stride: Stride between value cache blocks.
        cache_token_stride: Stride between key cache tokens.
        cache_head_stride: Stride between key cache heads.
        value_cache_token_stride: Stride between value cache tokens.
        value_cache_head_stride: Stride between value cache heads.
        key_stride: Stride between source key tokens.
        value_stride: Stride between source value tokens.
        num_heads: Number of attention heads.
        head_size: Hidden dimension per head.
        block_size: Number of slots per cache block.
        HEAD_BLOCK: Number of heads processed per program.
        BLOCK_D: Vectorized dimension size (power-of-2 padded).
        HAS_SWA: Enable SWA remapping.
        USE_SCALE: Enable scale division before storing.
        DO_PLAN: Also build the vattn split-KV segment plan (decode only: one
            token per sequence, so program ``token_idx`` plans sequence
            ``token_idx``). Runs before the slot check so padded rows are planned.
        seq_lens_ptr, plan_ptr, tok_nseg_ptr, num_seqs, num_work, target_wgs,
        seg_max, BLOCK_B, BLOCK_W: plan inputs/outputs, see ``seg_plan_program``.
    """

    # ----------------------------------
    # program ids
    # pid0 = token
    # pid1 = head block
    # ----------------------------------
    token_idx = tl.program_id(0)
    head_block_idx = tl.program_id(1)

    if DO_PLAN:
        if head_block_idx == 0:
            my_n = seg_plan_program(
                token_idx,
                seq_lens_ptr,
                plan_ptr,
                num_seqs,
                num_work,
                target_wgs,
                seg_max,
                BLOCK_B,
                BLOCK_W,
            )
            tl.store(tok_nseg_ptr + token_idx, my_n)

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
    tgt_k = (
        block_idx * block_stride
        + block_offset * cache_token_stride
        + head_idx[:, None] * cache_head_stride
        + dim_idx[None, :]
    )
    tgt_v = (
        block_idx * value_block_stride
        + block_offset * value_cache_token_stride
        + head_idx[:, None] * value_cache_head_stride
        + dim_idx[None, :]
    )
    tl.store(key_cache_ptr + tgt_k, k, mask=mask)
    tl.store(value_cache_ptr + tgt_v, v, mask=mask)


def launch_reshape_and_cache_flash(
    key,
    value,
    key_cache,
    value_cache,
    slot_mapping,
    swa_slot_mapping=None,
    k_scale=None,
    v_scale=None,
    seg_plan=None,
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
        seg_plan: Optional ``(seq_lens, plan, tok_nseg, seg_max, target_wgs)`` to
            also build the vattn split-KV segment plan in this launch (decode,
            one token per sequence; see ``vattn_asm_gfx950.alloc_seg_plan``)
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

    if seg_plan is not None:
        seq_lens, plan, tok_nseg, seg_max, target_wgs = seg_plan
        num_seqs = seq_lens.shape[0]
        assert num_tokens == num_seqs
        plan_args = (seq_lens, plan, tok_nseg)
        plan_scalars = (num_seqs, plan.numel() - 1, target_wgs, seg_max)
    else:
        plan_args = (key, key, key)
        plan_scalars = (0, 0, 0, 0)

    reshape_and_cache_flash[grid](
        key,
        value,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        k_scale if k_scale is not None else key,
        v_scale if v_scale is not None else key,
        *plan_args,
        key_cache.stride(0),
        value_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(1),
        value_cache.stride(2),
        key.stride(0),
        value.stride(0),
        num_heads,
        head_size,
        key_cache.shape[1],
        *plan_scalars,
        HEAD_BLOCK=HEAD_BLOCK,
        BLOCK_D=BLOCK_D,
        HAS_SWA=(swa_slot_mapping is not None),
        USE_SCALE=(k_scale is not None),
        DO_PLAN=(seg_plan is not None),
        BLOCK_B=max(16, triton.next_power_of_2(plan_scalars[0])),
        BLOCK_W=max(64, triton.next_power_of_2(plan_scalars[3])),
    )
