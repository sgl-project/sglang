"""NPU DSA adaptation for token-sharded decode context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sgl_kernel_npu  # noqa: F401  Registers torch.ops.npu sparse attention ops.
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from collections.abc import Callable

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def forward_dcp_sparse_attention(
    *,
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    forward_metadata,
    forward_batch: ForwardBatch,
    speculative_num_draft_tokens: int | None,
    scaling: float,
    fp8_attention: Callable | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one rank's sparse partial attention and natural-log LSE."""

    is_target_verify = forward_batch.forward_mode.is_target_verify()
    if is_target_verify:
        block_tables = forward_metadata.dcp_spec_block_tables
        local_lens = forward_metadata.dcp_spec_seq_lens
    else:
        block_tables = forward_metadata.dcp_block_tables
        local_lens = forward_metadata.dcp_seq_lens

    assert block_tables is not None, (
        "NPU DSA+DCP sparse attention requires rank-local paged-KV metadata"
    )
    if is_target_verify:
        # DCP verify presents every speculative query as an independent
        # single-query sequence. Its KV frontier and page-table row therefore
        # stay expanded in request-major [bs * draft_token_num] order.
        num_query_rows = q_nope.shape[0]
        assert block_tables.shape[0] == num_query_rows, (
            "NPU DSA+DCP verify block-table rows must match query rows: "
            f"{block_tables.shape[0]} != {num_query_rows}"
        )
        assert local_lens.shape[0] == num_query_rows, (
            "NPU DSA+DCP verify KV-length rows must match query rows: "
            f"{local_lens.shape[0]} != {num_query_rows}"
        )
        actual_seq_lengths_query = torch.arange(
            1,
            num_query_rows + 1,
            device=q_nope.device,
            dtype=torch.int32,
        )
    else:
        actual_seq_lengths_query = actual_seq_lengths_query.to(
            device=q_nope.device, dtype=torch.int32
        )
    kwargs = dict(
        sparse_indices=topk_indices,
        scale_value=scaling,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=local_lens,
        block_table=block_tables,
    )
    num_heads = q_nope.shape[1]
    if fp8_attention is None:
        # The BF16 kernel accepts 1–64 or 128 MLA query heads. Kimi K3.1
        # gathers 96 heads when the DCP group spans its attention TP group.
        if 64 < num_heads < 128:
            padding = (0, 0, 0, 128 - num_heads)
            q_nope = F.pad(q_nope, padding)
            q_rope = F.pad(q_rope, padding)
        attn_out, softmax_max, softmax_sum = torch.ops.npu.sgl_sparse_flash_attention(
            query=q_nope,
            key=k_nope,
            value=k_nope,
            query_rope=q_rope,
            key_rope=k_rope,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            # Global top-k already enforces causal visibility. Local indices no
            # longer carry enough information for the kernel's causal sparse mode.
            sparse_mode=0,
            attention_mode=2,
            return_softmax_lse=True,
            **kwargs,
        )
    else:
        attn_out, softmax_max, softmax_sum = fp8_attention(
            query=q_nope,
            query_rope=q_rope,
            packed_kv=k_nope.view(torch.uint8),
            **kwargs,
        )
    lse = softmax_max.float() + torch.log(softmax_sum.float())
    lse = lse.permute(1, 0, 2).reshape(lse.shape[1], -1)
    if attn_out.shape[1] != num_heads:
        attn_out = attn_out[:, :num_heads].contiguous()
        lse = lse[:, :num_heads].contiguous()
    return attn_out, lse
