"""NPU DSA adaptation for token-sharded decode context parallelism."""

from __future__ import annotations

from typing import TYPE_CHECKING

import sgl_kernel_npu  # noqa: F401  Registers torch.ops.sgl_kernel_npu.
import torch

if TYPE_CHECKING:
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
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Compute sparse partial attention and return its local softmax stats."""

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
    attn_out, softmax_max, softmax_sum = torch.ops.npu.sgl_sparse_flash_attention(
        query=q_nope,
        key=k_nope,
        value=k_nope,
        query_rope=q_rope,
        key_rope=k_rope,
        sparse_indices=topk_indices,
        scale_value=scaling,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=local_lens,
        block_table=block_tables,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        # Global top-k already enforces causal visibility. Local indices no
        # longer carry enough information for the kernel's causal sparse mode.
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    return attn_out, (softmax_max, softmax_sum)


def pinned_dcp_verify_lens(
    total_kv_lens: torch.Tensor,
    query_len: int,
    dcp_size: int,
    dcp_rank: int,
    page_size: int,
) -> torch.Tensor:
    """Build request-major local KV frontiers directly in pinned CPU memory."""
    cycle = dcp_size * page_size
    rank_offset = dcp_rank * page_size
    values = []
    for length in total_kv_lens.tolist():
        if length < query_len:
            values.extend([0] * query_len)
            continue
        for frontier in range(length - query_len + 1, length + 1):
            whole, tail = divmod(frontier, cycle)
            values.append(
                whole * page_size + min(page_size, max(0, tail - rank_offset))
            )
    return torch.tensor(values, dtype=torch.int32, pin_memory=True)


def try_write_dcp_verify_page_tables(
    slots: torch.Tensor,
    requests: torch.Tensor,
    ordinary: torch.Tensor,
    local: torch.Tensor | None,
    active_pages: int,
    query_len: int,
    page_size: int,
    dcp_size: int,
) -> bool:
    """Fill both captured page tables when the native kernel is available."""
    # Enable the validated GLM-5.2 Verify shape first.
    if (dcp_size, page_size, query_len) != (4, 128, 4):
        return False
    op = getattr(torch.ops.npu, "sgl_dcp_verify_page_tables", None)
    if (
        op is None
        or local is None
        or slots.device.type != "npu"
        or slots.device != requests.device
        or slots.device != ordinary.device
        or slots.device != local.device
        or slots.dtype != torch.int32
        or ordinary.dtype != torch.int32
        or local.dtype != torch.int32
        or requests.dtype not in (torch.int32, torch.int64)
        or not all(t.is_contiguous() for t in (slots, requests, ordinary, local))
        or ordinary.size(0) != requests.numel()
        or local.size(0) != requests.numel() * query_len
        or not 0
        <= active_pages
        <= min(ordinary.size(1), (slots.size(1) + page_size - 1) // page_size)
        or (active_pages + dcp_size - 1) // dcp_size > local.size(1)
    ):
        return False
    op(slots, requests, ordinary, local, active_pages, query_len, page_size, dcp_size)
    return True
