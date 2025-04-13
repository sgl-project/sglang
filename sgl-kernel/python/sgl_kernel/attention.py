from typing import Tuple

import torch


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    torch.ops.sgl_kernel.lightning_attention_decode.default(
        q, k, v, past_kv, slope, output, new_kv
    )


def merge_state(
    v_a: torch.Tensor, s_a: torch.Tensor, v_b: torch.Tensor, s_b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    v_merged = torch.empty_like(v_a)
    s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def cutlass_mla_decode(
    q_nope_and_q_pe: torch.Tensor,
    kv_c_and_k_pe_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    assert (
        q_nope_and_q_pe.ndim == 3
    ), f"q_nope_and_q_pe must be a 3D tensor, but got {q_nope_and_q_pe.ndim}"
    assert (
        kv_c_and_k_pe_cache.ndim == 3
    ), f"kv_c_and_k_pe_cache must be a 3D tensor, but got {kv_c_and_k_pe_cache.ndim}"
    B_q, H, D_q = q_nope_and_q_pe.shape
    _, PAGE_SIZE, D_ckv = kv_c_and_k_pe_cache.shape

    D_latent = 512
    D_rope = 64
    assert D_q == D_ckv and D_q == D_latent + D_rope, (
        f"D_q must be equal to D_ckv and D_q must be equal to D_latent + D_rope, "
        f"but got D_q = {D_q}, D_ckv = {D_ckv}, D_latent = {D_latent}, D_rope = {D_rope}"
    )
    assert H == 128, f"H must be 128, but got {H}"
    # TODO: There is currently an illegal memory access issue with page size !=
    # 128. Change this when it is fixed.
    assert PAGE_SIZE == 128, f"PAGE_SIZE must be 128, but got {PAGE_SIZE}"

    # TODO(kaixih@nvidia): support fp8
    assert q_nope_and_q_pe.dtype in (
        torch.float16,
        torch.bfloat16,
    ), f"q_nope_and_q_pe.dtype needs to be fp16 or bf16 but got {q_nope_and_q_pe.dtype}."
    assert kv_c_and_k_pe_cache.dtype == q_nope_and_q_pe.dtype, (
        f"kv_c_and_k_pe_cache.dtype needs to be the same as q_nope_and_q_pe.dtype, "
        f"but got {kv_c_and_k_pe_cache.dtype}."
    )
    assert (
        seq_lens.dtype == torch.int32
    ), f"seq_lens.dtype needs to be int32 but got {seq_lens.dtype}."
    assert (
        page_table.dtype == torch.int32
    ), f"page_table.dtype needs to be int32 but got {page_table.dtype}."

    out = torch.empty(
        (B_q, H, D_latent), device=q_nope_and_q_pe.device, dtype=q_nope_and_q_pe.dtype
    )

    torch.ops.sgl_kernel.cutlass_mla_decode.default(
        out, q_nope_and_q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace
    )
    return out


def cutlass_mla_get_workspace_size(
    max_seq_len: int, num_batches: int, sm_count: int = 0
) -> int:
    return torch.ops.sgl_kernel.cutlass_mla_get_workspace_size.default(
        max_seq_len, num_batches, sm_count
    )


def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.zeros(batch_size,
                              num_heads,
                              num_rows,
                              dtype=q_seqlens.dtype,
                              device=q_seqlens.device)
    block_offset = torch.zeros(batch_size,
                               num_heads,
                               num_rows,
                               nnz_slash,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)
    column_count = torch.zeros(batch_size,
                               num_heads,
                               num_rows,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)
    column_index = torch.zeros(batch_size,
                               num_heads,
                               num_rows,
                               nnz_vertical,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)

    torch.ops.sgl_kernel.convert_vertical_slash_indexes(
        block_count, block_offset, column_count, column_index, q_seqlens,
        kv_seqlens, vertical_indexes, slash_indexes, context_size,
        block_size_M, block_size_N, causal)
    return block_count, block_offset, column_count, column_index


def convert_vertical_slash_indexes_mergehead(
    q_seqlens: torch.Tensor,  # [BATCH, ]
    kv_seqlens: torch.Tensor,  # [BATCH, ]
    vertical_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_V]
    slash_indexes: torch.Tensor,  # [BATCH, N_HEADS, NNZ_S]
    # [N_HEADS] : different head use different number of indices
    vertical_indices_count: torch.Tensor,
    slash_indices_count: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = slash_indexes.size(0)
    num_heads = slash_indexes.size(1)
    nnz_slash = slash_indexes.size(2)
    nnz_vertical = vertical_indexes.size(2)
    num_rows = (context_size + block_size_M - 1) // block_size_M

    block_count = torch.empty(batch_size,
                              num_heads,
                              num_rows,
                              dtype=q_seqlens.dtype,
                              device=q_seqlens.device)
    block_offset = torch.empty(batch_size,
                               num_heads,
                               num_rows,
                               nnz_slash,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)
    column_count = torch.empty(batch_size,
                               num_heads,
                               num_rows,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)
    column_index = torch.empty(batch_size,
                               num_heads,
                               num_rows,
                               nnz_vertical,
                               dtype=q_seqlens.dtype,
                               device=q_seqlens.device)

    torch.ops.sgl_kernel.convert_vertical_slash_indexes_mergehead(
        block_count, block_offset, column_count, column_index, q_seqlens,
        kv_seqlens, vertical_indexes, slash_indexes, vertical_indices_count,
        slash_indices_count, context_size, block_size_M, block_size_N, causal)
    return block_count, block_offset, column_count, column_index
