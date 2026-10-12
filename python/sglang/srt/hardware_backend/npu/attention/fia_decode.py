"""Isolated non-MLA BSH FIA compile boundary; no workspace approximation."""

from __future__ import annotations
import torch
import torch_npu


@torch.library.custom_op("sglang_npu::fia_decode_bsh", mutates_args=())
def fia_decode_bsh(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    num_heads: int,
    num_key_value_heads: int,
    scale: float,
    actual_seq_lengths_kv: list[int],
    atten_mask: torch.Tensor | None,
    sparse_mode: int,
    output_hidden_size: int,
) -> torch.Tensor:
    workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
        query,
        key_cache,
        value_cache,
        block_table=block_table,
        block_size=block_size,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        input_layout="BSH",
        scale=scale,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        atten_mask=atten_mask,
        sparse_mode=sparse_mode,
    )
    output = torch.empty(
        (query.shape[0], 1, output_hidden_size), dtype=query.dtype, device=query.device
    )
    softmax_lse = torch.empty(1, dtype=query.dtype, device=query.device)
    torch_npu.npu_fused_infer_attention_score.out(
        query,
        key_cache,
        value_cache,
        block_table=block_table,
        block_size=block_size,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        input_layout="BSH",
        scale=scale,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        atten_mask=atten_mask,
        sparse_mode=sparse_mode,
        workspace=workspace,
        out=[output, softmax_lse],
    )
    return output


@fia_decode_bsh.register_fake
def _fia_decode_bsh_fake(
    query,
    key_cache,
    value_cache,
    block_table,
    block_size,
    num_heads,
    num_key_value_heads,
    scale,
    actual_seq_lengths_kv,
    atten_mask,
    sparse_mode,
    output_hidden_size,
):
    return torch.empty(
        (query.shape[0], 1, output_hidden_size), dtype=query.dtype, device=query.device
    )
