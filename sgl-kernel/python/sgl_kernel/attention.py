import torch


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    torch.ops.sgl_kernel.lightning_attention_decode.default(
        q, k, v, past_kv, slope, output, new_kv
    )


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

    torch.ops.sgl_kernel.cutlass_mla_decode(
        out, q_nope_and_q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace
    )
    return out


def cutlass_mla_get_workspace_size(
    max_seq_len: int, num_batches: int, sm_count: int = 0
) -> int:
    return torch.ops.sgl_kernel.cutlass_mla_get_workspace_size(
        max_seq_len, num_batches, sm_count
    )
