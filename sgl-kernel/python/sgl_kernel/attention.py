from typing import Optional, Tuple

import torch


def lightning_attention_decode(q, k, v, past_kv, slope, output, new_kv):
    torch.ops.sgl_kernel.lightning_attention_decode.default(
        q, k, v, past_kv, slope, output, new_kv
    )


def merge_state(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
    return v_merged, s_merged


def merge_state_v2(
    v_a: torch.Tensor,
    s_a: torch.Tensor,
    v_b: torch.Tensor,
    s_b: torch.Tensor,
    v_merged: Optional[torch.Tensor] = None,
    s_merged: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_a = s_a.to(torch.float32)
    s_b = s_b.to(torch.float32)
    # TODO(DefTruth): Currently, the custom merge_attn_states kernel
    # does not support the FP8 data type and non - CUDA devices.
    # It may be necessary to fall back to using the Triton kernel.

    # Avoid creating new tensors if they are already provided
    if v_merged is None:
        v_merged = torch.empty_like(v_a)
    if s_merged is None:
        s_merged = torch.empty_like(s_a)
    torch.ops.sgl_kernel.merge_state_v2.default(v_a, s_a, v_b, s_b, v_merged, s_merged)
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

    assert len(page_table.shape) == 2
    B_block_table, block_num = page_table.shape
    assert B_block_table == B_q
    assert block_num % (128 / PAGE_SIZE) == 0

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
