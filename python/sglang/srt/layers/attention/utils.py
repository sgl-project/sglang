import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda

_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096
FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON = tl.constexpr(_FLASHMLA_CREATE_KV_BLOCK_SIZE)

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import concat_mla_absorb_q

from sglang.jit_kernel.utils import is_arch_support_pdl


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


def get_num_page_per_block_flashmla(page_size: int = 64) -> int:
    num_page_per_block = _FLASHMLA_CREATE_KV_BLOCK_SIZE // page_size
    return num_page_per_block


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    PAGED_SIZE: tl.constexpr = 64,
):
    NUM_PAGE_PER_BLOCK: tl.constexpr = (
        FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE
    )
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON)

    for i in range(num_pages_loop):
        # index into req_to_token_ptr needs to be int64
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )


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
def pad_sequence_with_mask_kernel(
    input_ptr,  # (total_tokens, hidden)
    offsets_ptr,  # (B,)
    lengths_ptr,  # (B,)
    output_ptr,  # (B, max_len, hidden)
    mask_ptr,  # (B, max_len)
    max_len,
    hidden_dim,
    BLOCK_M: tl.constexpr,  # seq block
    BLOCK_D: tl.constexpr,  # hidden block
):
    b = tl.program_id(0)  # batch index
    m = tl.program_id(1)  # seq block index

    offset = tl.load(offsets_ptr + b)
    length = tl.load(lengths_ptr + b)

    seq_ids = m * BLOCK_M + tl.arange(0, BLOCK_M)
    hid_ids = tl.arange(0, BLOCK_D)

    seq_mask = seq_ids < max_len
    valid_token = seq_ids < length

    # input index
    in_token = offset + seq_ids
    in_ptr = input_ptr + in_token[:, None] * hidden_dim + hid_ids[None, :]

    # output index
    out_ptr = (
        output_ptr
        + b * max_len * hidden_dim
        + seq_ids[:, None] * hidden_dim
        + hid_ids[None, :]
    )

    values = tl.load(
        in_ptr,
        mask=valid_token[:, None] & (hid_ids[None, :] < hidden_dim),
        other=0.0,
    )

    tl.store(
        out_ptr,
        values,
        mask=seq_mask[:, None] & (hid_ids[None, :] < hidden_dim),
    )

    # attention mask
    if tl.program_id(2) == 0:
        mask_out_ptr = mask_ptr + b * max_len + seq_ids
        tl.store(mask_out_ptr, valid_token, mask=seq_mask)


def pad_sequence_with_mask(
    input_emb,  # (total_tokens, hidden)
    offsets,  # (B,)
    lengths,  # (B,)
    max_len,
):
    B = offsets.shape[0]
    hidden_dim = input_emb.shape[1]

    output = torch.zeros(
        (B, max_len, hidden_dim),
        device=input_emb.device,
        dtype=input_emb.dtype,
    )
    attn_mask = torch.empty(
        (B * max_len),
        device=input_emb.device,
        dtype=torch.bool,
    )

    BLOCK_D = triton.next_power_of_2(hidden_dim)
    BLOCK_M = triton.next_power_of_2(max_len)

    grid = (
        B,
        triton.cdiv(max_len, BLOCK_M),
        1,
    )

    pad_sequence_with_mask_kernel[grid](
        input_emb,
        offsets,
        lengths,
        output,
        attn_mask,
        max_len,
        hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )

    return B, output, attn_mask


@triton.jit
def seqlens_expand_kernel(
    extend_seq_lens_ptr,  # [N]
    seq_lens_ptr,  # [N]
    offsets_ptr,  # [N+1]
    output_ptr,  # [sum(extend_seq_lens)]
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid >= N:
        return

    qo_len = tl.load(extend_seq_lens_ptr + pid)
    kv_len = tl.load(seq_lens_ptr + pid)

    start = kv_len - qo_len + 1
    out_offset = tl.load(offsets_ptr + pid)

    offs = tl.arange(0, BLOCK)
    mask = offs < qo_len

    values = start + offs
    tl.store(output_ptr + out_offset + offs, values, mask=mask)


def seqlens_expand_triton(
    extend_seq_lens: torch.Tensor,
    seq_lens: torch.Tensor,
    total_len: int,
    max_q_len: int,
):
    """
    extend_seq_lens: [N], int32, CUDA
    seq_lens:        [N], int32, CUDA
    """
    assert extend_seq_lens.is_cuda
    assert seq_lens.is_cuda

    N = extend_seq_lens.numel()

    offsets = torch.zeros(N + 1, device=extend_seq_lens.device, dtype=torch.int32)
    offsets[1:] = torch.cumsum(extend_seq_lens, dim=0)
    output = torch.empty(total_len, device=extend_seq_lens.device, dtype=torch.int32)

    BLOCK = triton.next_power_of_2(max_q_len)
    grid = (N,)

    seqlens_expand_kernel[grid](
        extend_seq_lens,
        seq_lens,
        offsets,
        output,
        N,
        BLOCK=BLOCK,
    )

    return output


# When num_kv_heads=1, we have tensors with degenerate strides,
# For example, as below, where we have stride[-3] == stride[-2]:
# - shape: [num_pages, 1, 64, 128]
# - stride: [8192, 128, 128, 1]
# This will cause TMA desc validation fail in flashinfer (trtllm-mha backend).
#
# See: https://github.com/flashinfer-ai/flashinfer/issues/2232
def canonicalize_stride(tensor: torch.Tensor) -> torch.Tensor:
    """
    Adjust degenerate strides for a tensor, make it canonical.
    """
    sizes = tensor.size()
    strides = tensor.stride()
    ndim = tensor.dim()

    need_fix = any(
        sizes[i] == 1 and strides[i] == strides[i + 1] for i in range(ndim - 1)
    )

    if not need_fix:
        return tensor

    # canonicalize the stride
    # Example:
    # - shape: [num_pages, 1, 64, 128]
    # - stride: [8192, 128, 128, 1] (wrong!)
    # Gives new stride: [8192, 8192, 128 ,1] (correct!)
    new_strides = [0] * ndim
    new_strides[-1] = 1
    for i in range(ndim - 2, -1, -1):
        new_strides[i] = new_strides[i + 1] * sizes[i + 1]

    return tensor.as_strided(sizes, new_strides)


def mla_quantize_and_rope_for_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    pos_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import flashinfer.rope

    """Quantize and apply RoPE for FP8 attention path.

        This function handles the FP8 quantization and RoPE application for MLA attention.
        It takes separate query/key nope and rope components, applies RoPE to the rope parts,
        quantizes all components to FP8, and merges the query components into a single tensor.

        Args:
            q_nope: Query no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            q_rope: Query RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            k_nope: Key no-position-encoding component [seq_len, num_heads, kv_lora_rank]
                - expected dtype: torch.bfloat16
            k_rope: Key RoPE component [seq_len, num_heads, qk_rope_head_dim]
                - expected dtype: torch.bfloat16
            pos_ids: Position indices for each token
                - expected dtype: torch.int64 or torch.int32
            cos_sin_cache: Precomputed cosine/sine cache for RoPE
                - expected dtype: matches q_/k_ input dtype (torch.bfloat16)
            is_neox: Whether to use NeoX-style RoPE (interleaved) or GPT-style (half rotation)
            kv_lora_rank: Dimension of the no-position-encoding component
            qk_rope_head_dim: Dimension of the RoPE component

        Returns:
            tuple: (merged_q_out, k_nope_out, k_rope_out) quantized to FP8
                - merged_q_out: [seq_len, num_heads, kv_lora_rank + qk_rope_head_dim], dtype=torch.float8_e4m3fn
                - k_nope_out:   [seq_len, num_heads, kv_lora_rank], dtype=torch.float8_e4m3fn
                - k_rope_out:   [seq_len, num_heads, qk_rope_head_dim], dtype=torch.float8_e4m3fn
        """
    attn_dtype = torch.float8_e4m3fn
    q_len, num_heads = q_rope.shape[0], q_rope.shape[1]

    # Allocate output tensors with FP8 dtype
    # Query output will contain merged nope + rope components
    q_out = q_rope.new_empty(
        q_len,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        dtype=attn_dtype,
    )

    # Key outputs maintain original shapes but with FP8 dtype
    k_rope_out = k_rope.new_empty(k_rope.shape, dtype=attn_dtype)
    k_nope_out = k_nope.new_empty(k_nope.shape, dtype=attn_dtype)

    # Apply RoPE and quantize all components in a single fused kernel call
    # This kernel handles:
    # 1. RoPE application to q_rope and k_rope using cos_sin_cache and positions
    # 2. Quantization of all components to FP8 format
    # 3. Output placement into pre-allocated tensors
    flashinfer.rope.mla_rope_quantize_fp8(
        q_rope=q_rope,
        k_rope=k_rope,
        q_nope=q_nope,
        k_nope=k_nope,
        cos_sin_cache=cos_sin_cache,
        pos_ids=pos_ids,
        is_neox=is_neox,
        quantize_dtype=attn_dtype,
        # Output tensor slicing: q_out contains [nope_part, rope_part]
        q_rope_out=q_out[..., kv_lora_rank:],  # RoPE part goes to end
        k_rope_out=k_rope_out,
        q_nope_out=q_out[..., :kv_lora_rank],  # Nope part goes to beginning
        k_nope_out=k_nope_out,
        # Quantization scales (set to 1.0 for no additional scaling)
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
        enable_pdl=is_arch_support_pdl(),
    )

    return q_out, k_nope_out, k_rope_out


def concat_mla_absorb_q_general(q_nope, q_rope):
    if _is_cuda and q_nope.shape[-1] == 512 and q_rope.shape[-1] == 64:
        return concat_mla_absorb_q(q_nope, q_rope)
    else:
        return torch.cat([q_nope, q_rope], dim=-1)


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
def _get_gptj_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    # GPT-J rotary layout:
    # Pair adjacent dimensions and apply:
    # [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]

    # Apply sign inversion on odd positions.
    x_rotated = tl.where(x_rotated_mask, x, -x)
    # Reshape into (D/2, 2) pairs.
    x_rotated = tl.reshape(x_rotated, (BLOCK_D_HALF, 2))
    # Swap each pair.
    x_rotated = tl.flip(x_rotated, 1)
    # Flatten back to original shape.
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    return x_rotated


@triton.jit
def _get_neox_rotated_x(
    x,
    x_rotated_mask,
    BLOCK_D: tl.constexpr,
    BLOCK_D_HALF: tl.constexpr,
):
    # GPT-NeoX rotary layout:
    # Split head dimension into two halves:
    # [x0, x1, x2, x3] -> [-x2, -x3, x0, x1]

    # Keep first half positive, second half negative.
    x_rotated = tl.where(x_rotated_mask, x, -x)
    # Reshape into (2, D/2).
    x_rotated = tl.reshape(x_rotated, (2, BLOCK_D_HALF))
    # Reverse each half.
    x_rotated = tl.flip(x_rotated, 1)
    # Flatten and reverse full vector.
    x_rotated = tl.reshape(x_rotated, (BLOCK_D,))
    x_rotated = tl.flip(x_rotated, 0)
    return x_rotated


@triton.jit
def _unit_rope(
    x_ptrs,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    # Load one full attention head vector.
    x_pe = tl.load(x_ptrs)

    # Stage 1: Build rotated vector according to rotary layout.
    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    # Stage 2: Apply RoPE transform:
    # x' = x*cos + rotate(x)*sin
    x_pe = x_pe * cos + x_pe_rotated * sin

    return x_pe


@triton.jit
def _load_cos_sin(
    cos_sin_ptr,
    pos,
    d_cos_offs,
    stride_t,
    stride_d,
    freq_dim,
):
    base = pos * stride_t
    cos = tl.load(cos_sin_ptr + base + d_cos_offs * stride_d)
    sin = tl.load(cos_sin_ptr + base + (d_cos_offs + freq_dim) * stride_d)
    return cos, sin


@triton.jit
def _fused_qk_rope_reshape_and_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    cos_sin_ptr,
    offs_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    swa_slot_mapping_ptr,
    q_out_ptr,
    k_out_ptr,
    zeros_out_ptr,
    T,
    T_slot,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    cos_sin_stride_t,
    cos_sin_stride_d,
    q_out_stride_t,
    q_out_stride_h,
    q_out_stride_d,
    k_out_stride_t,
    k_out_stride_h,
    k_out_stride_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    key_cache_stride_x,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    value_cache_stride_slot_chunk,
    value_cache_stride_x,
    zeros_out_stride_t,
    zeros_out_stride_h,
    zeros_out_stride_d,
    k_scale_ptr,
    v_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    FLASH_LAYOUT: tl.constexpr,
    VALUE_SHUFFLE_LAYOUT: tl.constexpr = False,
    HAVE_POS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
    HAVE_ZEROS: tl.constexpr = False,
    HAS_SWA: tl.constexpr = False,
):
    # ============================================================
    # Stage 0: Static stride assumptions for Triton compiler
    #
    # These assumptions help Triton optimize pointer arithmetic and
    # simplify generated address calculations.
    # ============================================================

    tl.assume(q_stride_t >= 0)
    tl.assume(q_stride_h >= 0)
    tl.assume(q_stride_d >= 0)
    tl.assume(k_stride_t >= 0)
    tl.assume(k_stride_h >= 0)
    tl.assume(k_stride_d >= 0)
    tl.assume(v_stride_t >= 0)
    tl.assume(v_stride_h >= 0)
    tl.assume(v_stride_d >= 0)
    tl.assume(cos_sin_stride_t >= 0)
    tl.assume(cos_sin_stride_d >= 0)
    tl.assume(q_out_stride_t >= 0)
    tl.assume(q_out_stride_h >= 0)
    tl.assume(q_out_stride_d >= 0)
    tl.assume(k_out_stride_t >= 0)
    tl.assume(k_out_stride_h >= 0)
    tl.assume(k_out_stride_d >= 0)
    tl.assume(key_cache_stride_t >= 0)
    tl.assume(key_cache_stride_h >= 0)
    tl.assume(key_cache_stride_d >= 0)
    tl.assume(key_cache_stride_b >= 0)
    tl.assume(key_cache_stride_x >= 0)
    tl.assume(value_cache_stride_t >= 0)
    tl.assume(value_cache_stride_h >= 0)
    tl.assume(value_cache_stride_d >= 0)
    tl.assume(value_cache_stride_b >= 0)
    tl.assume(value_cache_stride_slot_chunk >= 0)
    tl.assume(value_cache_stride_x >= 0)
    tl.assume(zeros_out_stride_t >= 0)
    tl.assume(zeros_out_stride_h >= 0)
    tl.assume(zeros_out_stride_d >= 0)

    # ============================================================
    # Stage 1: Program instance mapping
    #
    # Each program handles:
    #   - one (token, q_head) for Q path
    #   - selected KV ownership for cache write path
    #
    # pid layout:
    #   [0, T*QH)            -> decode Q path
    #   [T*QH, extra KV)     -> KV-only path
    # ============================================================

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    # ============================================================
    # Stage 2: Main decode path (Q always active)
    # ============================================================

    if pid < T * QH:
        pid_t = pid // QH
        pid_hq = pid % QH

        # --------------------------------------------------------
        # Stage 2.1: Compute rotary frequency offsets
        #
        # RoPE frequencies may be stored as:
        #   D/2 frequencies (shared front-half)
        #   D frequencies (full explicit)
        # --------------------------------------------------------

        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
                # d_cos_mask = d_cos_offs < BLOCK_D_pe
            else:
                d_cos_offs = d_pe_offs // 2
                # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe
        else:
            d_cos_offs = d_pe_offs
            # d_cos_mask = d_cos_offs < BLOCK_D_pe

        # --------------------------------------------------------
        # Stage 2.2: Load token position and optional offset
        #
        # offs_ptr is used by chunked prefill / sliding-window decode.
        # --------------------------------------------------------
        pos = tl.load(pos_ptr + pid_t)
        if HAVE_POS:
            offset = tl.load(offs_ptr + pid_t)
            pos = pos + offset

        # --------------------------------------------------------
        # Stage 2.3: Load cosine / sine table
        # --------------------------------------------------------
        # cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        # cos = tl.load(cos_ptr + cos_offs)
        # sin = tl.load(sin_ptr + cos_offs)

        freq_dim = BLOCK_D_HALF_pe if REUSE_FREQS_FRONT_PART else BLOCK_D_pe

        cos, sin = _load_cos_sin(
            cos_sin_ptr,
            pos,
            d_cos_offs,
            cos_sin_stride_t,
            cos_sin_stride_d,
            freq_dim,
        )

        # --------------------------------------------------------
        # Stage 2.4: Apply RoPE to Q
        # --------------------------------------------------------
        q_ptrs = (
            q_ptr + pid_t * q_stride_t + pid_hq * q_stride_h + d_pe_offs * q_stride_d
        )
        q_pe = _unit_rope(
            q_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )

        # Store rotated Q output.
        q_out_ptrs = (
            q_out_ptr
            + pid_t * q_out_stride_t
            + pid_hq * q_out_stride_h
            + d_pe_offs * q_out_stride_d
        )
        tl.store(q_out_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))

        if HAVE_ZEROS:
            z = tl.zeros((BLOCK_D_pe,), dtype=zeros_out_ptr.dtype.element_ty)
            zeros_out_ptrs = (
                zeros_out_ptr
                + pid_t * zeros_out_stride_t
                + pid_hq * zeros_out_stride_h
                + d_pe_offs * zeros_out_stride_d
            )
            tl.store(zeros_out_ptrs, z)

        # ========================================================
        # Stage 3: KV ownership path
        #
        # Only one Q group leader writes KV:
        #   pid_hq % QH_PER_KH == 0
        #
        # This prevents duplicated KV cache writes.
        # ========================================================

        if pid_hq % QH_PER_KH == 0:
            # ----------------------------------------------------
            # Stage 3.1: Resolve cache slot
            # ----------------------------------------------------
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if HAS_SWA:
                pid_slot = tl.load(swa_slot_mapping_ptr + pid_slot)

            # ------------------------------------------------
            # Stage 3.2: Apply RoPE to K
            # ------------------------------------------------
            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
                pid_hk = pid_hq // QH_PER_KH
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )
                k_pe = _unit_rope(
                    k_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                # ------------------------------------------------
                # Stage 3.3: Optional fp8 scaling before cache
                # ------------------------------------------------

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                # ------------------------------------------------
                # Stage 3.4: Write K cache
                #
                # Two layouts supported:
                #   FLASH_LAYOUT
                #   paged KV layout
                # ------------------------------------------------

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                        + d_pe_offs * key_cache_stride_d
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )

                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                # ------------------------------------------------
                # Stage 3.5: Write V cache
                #
                # Supports:
                #   normal layout
                #   shuffle layout
                # ------------------------------------------------

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                if VALUE_SHUFFLE_LAYOUT:
                    slot_chunk = pid_b // X_SIZE
                    x_off = pid_b % X_SIZE
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + slot_chunk * value_cache_stride_slot_chunk
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d
                        + x_off * value_cache_stride_x
                    )
                else:
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))
    # ============================================================
    # Stage 4: Extra KV-only path
    #
    # Handles tokens that only require cache update:
    #   T_slot > T
    #
    # No Q / no RoPE on Q branch.
    # ============================================================
    else:
        pid = pid - T * QH + T * KH
        if pid < T_slot * KH:
            pid_t = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if HAS_SWA:
                pid_slot = tl.load(swa_slot_mapping_ptr + pid_slot)

            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )

                k_pe = tl.load(k_ptrs)

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + d_pe_offs * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )
                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                if VALUE_SHUFFLE_LAYOUT:
                    slot_chunk = pid_b // X_SIZE
                    x_off = pid_b % X_SIZE
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + slot_chunk * value_cache_stride_slot_chunk
                        + d_pe_offs * value_cache_stride_d
                        + x_off * value_cache_stride_x
                    )
                else:
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_pe_offs * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))


def fused_qk_rope_reshape_and_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    pos: torch.Tensor,
    cos_sin: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    is_neox: bool,
    flash_layout: bool,
    apply_scale: bool = True,
    offs: torch.Tensor = None,
    q_out: torch.Tensor = None,
    k_out: torch.Tensor = None,
    output_zeros: bool = True,
    zeros_out: torch.Tensor = None,
    swa_slot_mapping=None,
):
    """
    Perform RoPE on q and k and along the last dimension and copy k and v in to key_cache and value_cache inplace

    Key parameters:
    - q: shape (T, QH, D).
    - k: shape (T_slot, KH, D).
    - v: shape (T_slot, KH, D).
    - if flash_layout:
    -     key_cache: shape (T_cache, block_size, KH, D).
    -     value_cache: shape (T_cache, block_size, KH, D).
    - else:
    -     key_cache: shape (T_cache, KH, D // x, block_size, x).
    -     value_cache: shape (T_cache, KH, D, block_size).
    - slot_mapping: shape (T_slot, ).

    T is the number of decode tokens, T_cahce * block_size is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - q_out: same shape as input q.
    - k_out: same shape as input k.
    - key_cache: same shape as input key_cache (inplace).
    - value_cache: same shape as input value_cache (inplace).
    - zeros_out: same shape as input q.
    """

    t, qh, d = q.shape
    tk, kh, dk = k.shape
    tv, vh, dv = v.shape
    if flash_layout:
        t_cache, block_size, kh_cache, dk_cache = key_cache.shape
        t_cache_v, block_size_v, vh_cache, dv_cache = value_cache.shape
        value_shuffle_layout = False
    else:
        t_cache, kh_cache, dkx_cache, block_size, x_cache = key_cache.shape
        if value_cache.ndim == 5:
            # value_cache shuffle: (num_blocks, num_kv_heads, block_size // x, head_size, x)
            t_cache_v, vh_cache, slot_chunk_v, dv_cache, x_v = value_cache.shape
            value_shuffle_layout = True
            block_size_v = slot_chunk_v * x_v
            assert block_size_v == block_size and x_v == x_cache, (
                f"value_cache shuffle (T,KH,block_size//x,D,x) must match key: "
                f"{block_size_v=} {block_size=} {x_v=} {x_cache=}"
            )
        else:
            t_cache_v, vh_cache, dv_cache, block_size_v = value_cache.shape
            value_shuffle_layout = False
    (t_slot,) = slot_mapping.shape

    assert (
        t == tk == tv and t_slot <= tk
    ), f"Number of tokens should be identical for q, kand v. The number of tokens of slot_mapping should no more than that of q, k and v, {t=} {tk=} {tv=} {t_slot=}"
    assert (
        block_size == block_size_v
    ), f"block size should be identical for key_cache, and value_cache {block_size} {block_size_v}"
    assert (
        kh == vh == kh_cache == vh_cache
    ), "KV head should be identical for k, v, key_cache, and value_cache"
    assert (
        t_cache == t_cache_v
    ), "Number of tokens should be identical for key_cache, and value_cache"
    if flash_layout:
        assert (
            d == dk == dv == dk_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
    else:
        assert (
            d == dk == dv == dkx_cache * x_cache == dv_cache
        ), "D dimension should be identical for q, k, and v"
        assert x_cache == triton.next_power_of_2(x_cache), "x_size should be power of 2"

    assert d == triton.next_power_of_2(d), "D dimension should be power of 2"
    assert block_size == triton.next_power_of_2(
        block_size
    ), "block_size should be power of 2"
    assert qh % kh == 0, "Q heads must be multiple of H heads"
    d_freq = cos_sin.shape[-1] // 2
    assert (d_freq == d // 2) or (
        d_freq == d
    ), "cos/sin last dim should be the same or half of the qk last dim"
    reuse_freqs_front_part = d_freq == d // 2

    if q_out is None:
        q_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)

    if k_out is None:
        k_out = torch.empty((tk, kh, dk), dtype=k.dtype, device=q.device)

    if zeros_out is not None:
        tz, qhz, dz = zeros_out.shape
        assert (
            t == tz and qh == qhz and d == dz
        ), f"q and zeros shape mismatch {q.shape=} {zeros_out.shape=}"
        output_zeros = True
    elif output_zeros:
        zeros_out = torch.empty((t, qh, d), dtype=q.dtype, device=q.device)
    else:
        zeros_out = None

    n_pid = t * qh + (t_slot - t) * kh if t_slot >= t else t * qh
    grid = (n_pid, 1, 1)
    _fused_qk_rope_reshape_and_cache_kernel[grid](
        q,
        k,
        v,
        pos,
        cos_sin,
        offs,
        key_cache,
        value_cache,
        slot_mapping,
        swa_slot_mapping,
        q_out,
        k_out,
        zeros_out,
        t,
        t_slot,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        cos_sin.stride(0),
        cos_sin.stride(-1),
        *q_out.stride(),
        *k_out.stride(),
        key_cache.stride(0) if not flash_layout else key_cache.stride(0),
        key_cache.stride(1) if not flash_layout else key_cache.stride(2),
        key_cache.stride(2) if not flash_layout else key_cache.stride(3),
        key_cache.stride(3) if not flash_layout else key_cache.stride(1),
        key_cache.stride(4) if not flash_layout else 0,
        value_cache.stride(0) if not flash_layout else value_cache.stride(0),
        value_cache.stride(1) if not flash_layout else value_cache.stride(2),
        (
            value_cache.stride(3)
            if (not flash_layout and value_shuffle_layout)
            else (value_cache.stride(2) if not flash_layout else value_cache.stride(3))
        ),
        (
            0
            if (not flash_layout and value_shuffle_layout)
            else (value_cache.stride(3) if not flash_layout else value_cache.stride(1))
        ),
        value_cache.stride(2) if (not flash_layout and value_shuffle_layout) else 0,
        value_cache.stride(4) if (not flash_layout and value_shuffle_layout) else 0,
        zeros_out.stride(0) if zeros_out is not None else 0,
        zeros_out.stride(1) if zeros_out is not None else 0,
        zeros_out.stride(2) if zeros_out is not None else 0,
        k_scale_ptr=k_scale,
        v_scale_ptr=v_scale,
        QH_PER_KH=qh // kh,
        QH=qh,
        KH=kh,
        REUSE_FREQS_FRONT_PART=reuse_freqs_front_part,
        IS_NEOX=is_neox,
        BLOCK_D_pe=d,
        BLOCK_D_HALF_pe=d // 2,
        BLOCK_SIZE=block_size,
        X_SIZE=x_cache if not flash_layout else 0,
        FLASH_LAYOUT=flash_layout,
        VALUE_SHUFFLE_LAYOUT=value_shuffle_layout,
        HAVE_POS=(offs is not None),
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        HAVE_V_SCALE=(v_scale is not None and apply_scale),
        HAVE_ZEROS=output_zeros,
        HAS_SWA=(swa_slot_mapping is not None),
        num_warps=1,
    )

    if zeros_out is not None:
        return q_out.view(-1, qh * d), k_out, key_cache, value_cache, zeros_out
    return q_out.view(-1, qh * d), k_out, key_cache, value_cache
