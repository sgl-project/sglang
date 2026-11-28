import torch
import triton
import triton.language as tl


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
def pad_draft_extend_query_kernel(
    q_ptr,  # Input query tensor [total_seq_len, num_heads, head_dim]
    padded_q_ptr,  # Output padded query tensor [batch_size, max_seq_len, num_heads, head_dim]
    seq_lens_q_ptr,  # Sequence lengths for each sequence [batch_size]
    cumsum_ptr,  # Cumulative sum of sequence lengths [batch_size + 1]
    batch_size,
    max_seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for padding draft extended query tensor with parallelized head and dim processing."""
    # Use 3D program IDs: (batch_seq, head_block, dim_block)
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // max_seq_len
    seq_pos = batch_seq_pid % max_seq_len

    if batch_id >= batch_size:
        return

    # Load sequence length for this batch
    seq_len = tl.load(seq_lens_q_ptr + batch_id)

    if seq_pos >= seq_len:
        return

    # Load cumulative sum to get start position in input tensor
    input_start = tl.load(cumsum_ptr + batch_id)
    input_pos = input_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, num_heads)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset
    input_offset = (
        input_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        q_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    # Calculate output offset
    output_offset = (
        batch_id * max_seq_len * num_heads * head_dim
        + seq_pos * num_heads * head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        padded_q_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def pad_draft_extend_query(
    q: torch.Tensor,
    padded_q: torch.Tensor,
    seq_lens_q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
) -> torch.Tensor:
    """Pad draft extended query using Triton kernel."""
    batch_size = cu_seqlens_q.shape[0] - 1
    max_seq_len_q = padded_q.shape[1]
    num_heads = padded_q.shape[2]
    head_dim = padded_q.shape[3]

    # Launch Triton kernel with 3D grid for parallelized head and dim processing
    BLOCK_SIZE = 64
    num_head_blocks = triton.cdiv(num_heads, BLOCK_SIZE)
    num_dim_blocks = triton.cdiv(head_dim, BLOCK_SIZE)
    grid = (batch_size * max_seq_len_q, num_head_blocks, num_dim_blocks)

    pad_draft_extend_query_kernel[grid](
        q_ptr=q,
        padded_q_ptr=padded_q,
        seq_lens_q_ptr=seq_lens_q,
        cumsum_ptr=cu_seqlens_q,
        batch_size=batch_size,
        max_seq_len=max_seq_len_q,
        num_heads=num_heads,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return padded_q


@triton.jit
def unpad_draft_extend_output_kernel(
    raw_out_ptr,  # Input raw output tensor (batch_size, token_per_batch, tp_q_head_num, v_head_dim)
    output_ptr,  # Output tensor (-1, tp_q_head_num, v_head_dim)
    num_accept_tokens_ptr,  # Accept lengths for each sequence [batch_size]
    cumsum_ptr,  # Cumulative sum of accept lengths [batch_size + 1]
    batch_size,
    token_per_batch,
    tp_q_head_num,
    v_head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for unpadding draft extended output tensor with parallelized head and dim processing."""
    batch_seq_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    dim_pid = tl.program_id(2)

    batch_id = batch_seq_pid // token_per_batch
    seq_pos = batch_seq_pid % token_per_batch

    if batch_id >= batch_size:
        return

    # Load accept length for this batch
    accept_len = tl.load(num_accept_tokens_ptr + batch_id)

    if seq_pos >= accept_len:
        return

    # Load cumulative sum to get start position in output tensor
    output_start = tl.load(cumsum_ptr + batch_id)
    output_pos = output_start + seq_pos

    # Calculate head and dim block ranges
    head_start = head_pid * BLOCK_SIZE
    head_end = tl.minimum(head_start + BLOCK_SIZE, tp_q_head_num)
    head_mask = tl.arange(0, BLOCK_SIZE) < (head_end - head_start)

    dim_start = dim_pid * BLOCK_SIZE
    dim_end = tl.minimum(dim_start + BLOCK_SIZE, v_head_dim)
    dim_mask = tl.arange(0, BLOCK_SIZE) < (dim_end - dim_start)

    # Calculate input offset: (batch_id, seq_pos, head_id, dim_id)
    input_offset = (
        batch_id * token_per_batch * tp_q_head_num * v_head_dim
        + seq_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Load data
    data = tl.load(
        raw_out_ptr + input_offset,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    output_offset = (
        output_pos * tp_q_head_num * v_head_dim
        + (head_start + tl.arange(0, BLOCK_SIZE))[:, None] * v_head_dim
        + (dim_start + tl.arange(0, BLOCK_SIZE))[None, :]
    )

    # Store data
    tl.store(
        output_ptr + output_offset,
        data,
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def unpad_draft_extend_output(
    raw_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens_q: torch.Tensor,
    sum_seq_lens_q: int,
    unpad_output_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unpad draft extended output using Triton kernel."""
    # raw_out: (batch_size, token_per_batch, layer.tp_q_head_num, layer.v_head_dim)
    batch_size = seq_lens_q.shape[0]
    token_per_batch = raw_out.shape[1]  # max_seq_len
    tp_q_head_num = raw_out.shape[2]  # num_heads
    v_head_dim = raw_out.shape[3]  # head_dim
    total_tokens = sum_seq_lens_q

    # Check if we're in CUDA graph mode (buffers are pre-allocated)
    if unpad_output_buffer is not None:
        # Use pre-allocated buffer for CUDA graph compatibility
        output = unpad_output_buffer[:total_tokens, :, :].to(dtype=raw_out.dtype)
    else:
        # Dynamic allocation for non-CUDA graph mode
        output = torch.empty(
            (total_tokens, tp_q_head_num, v_head_dim),
            dtype=raw_out.dtype,
            device=raw_out.device,
        )

    # Launch Triton kernel with 3D grid for parallelized head and dim processing
    BLOCK_SIZE = 64
    num_head_blocks = triton.cdiv(tp_q_head_num, BLOCK_SIZE)
    num_dim_blocks = triton.cdiv(v_head_dim, BLOCK_SIZE)
    grid = (batch_size * token_per_batch, num_head_blocks, num_dim_blocks)

    unpad_draft_extend_output_kernel[grid](
        raw_out_ptr=raw_out,
        output_ptr=output,
        num_accept_tokens_ptr=seq_lens_q,
        cumsum_ptr=cu_seqlens_q,
        batch_size=batch_size,
        token_per_batch=token_per_batch,
        tp_q_head_num=tp_q_head_num,
        v_head_dim=v_head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output[:total_tokens, :, :]


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

    # Clamp to >= 0: rows with kv_len < qo_len (DP-padded / idle-companion
    # rows whose kv is the CUDA-graph fill value) would otherwise produce
    # negative lengths, which unsigned consumers (e.g. the top-k v2 kernel,
    # which reads lengths as uint32) turn into ~4e9-token lengths and an
    # illegal memory access.
    values = tl.maximum(start + offs, 0)
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
