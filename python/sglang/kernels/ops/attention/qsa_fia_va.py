# SPDX-License-Identifier: Apache-2.0
"""Selected-token prefill FIA, reused from the vLLM NPU plugin 24487f4f48.

Adaptations: SG compact512 interface and retain all complete groups plus tail.
"""

import torch
import torch_npu

GBSA_COMPRESS_RATIO, GBSA_PAGE_SIZE, GBSA_RESIDUAL_TOKENS = 4, 128, 3

# Bound temporary mask memory even for long-context prefills.
QSA_FIA_QUERY_TILE = 4096
QSA_FIA_MASK_BUDGET_BYTES = 64 * 1024 * 1024


def build_selected_token_mask(
    selected_token_indices: torch.Tensor,
    effective_kv_length: int | torch.Tensor,
    kv_length: int,
) -> torch.Tensor:
    """Return [B, 1, Tq, KV] allowed positions, discarding invalid indices."""
    if selected_token_indices.ndim != 3 or kv_length <= 0:
        raise ValueError("Selected indices must be [B, Tq, K] and KV length must be positive")
    if isinstance(effective_kv_length, int) and not 0 <= effective_kv_length <= kv_length:
        raise ValueError("effective_kv_length must be between zero and kv_length")
    lengths = torch.as_tensor(effective_kv_length, device=selected_token_indices.device)
    lengths = lengths.reshape(lengths.shape + (1,) * (selected_token_indices.ndim - lengths.ndim))
    valid = (selected_token_indices >= 0) & (selected_token_indices < lengths)
    valid &= selected_token_indices < kv_length
    scatter_indices = torch.where(valid, selected_token_indices, kv_length).long()
    mask = torch.zeros(
        (*selected_token_indices.shape[:-1], kv_length + 1),
        dtype=torch.bool,
        device=selected_token_indices.device,
    )
    mask.scatter_(-1, scatter_indices, True)
    return mask[..., :kv_length].unsqueeze(1)


def expand_qsa_selected_tokens(compact_indices: torch.Tensor, visible_lens: torch.Tensor) -> torch.Tensor:
    """Expand complete groups and reconstruct the implicit GBSA residual group."""
    if compact_indices.ndim != 2 or compact_indices.shape[1] == 0:
        raise ValueError("Expected compact QSA group indices")
    groups = compact_indices
    full_groups = visible_lens.floor_divide(GBSA_COMPRESS_RATIO)
    valid = (groups >= 0) & (groups < full_groups.unsqueeze(-1))
    # SG retains all complete selected groups, including group 512 with tail.
    offsets = torch.arange(GBSA_COMPRESS_RATIO, device=groups.device, dtype=groups.dtype)
    tokens = groups.unsqueeze(-1) * GBSA_COMPRESS_RATIO + offsets
    tokens = torch.where(valid.unsqueeze(-1), tokens, -1).flatten(1)
    tail_offsets = torch.arange(GBSA_RESIDUAL_TOKENS, device=groups.device, dtype=groups.dtype)
    tail = full_groups.unsqueeze(-1) * GBSA_COMPRESS_RATIO + tail_offsets
    tail = torch.where(tail < visible_lens.unsqueeze(-1), tail, -1)
    return torch.cat((tokens, tail), dim=-1)


def prepare_qsa_fia_paged_kv(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    kv_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Expose descriptor-backed KV as zero-copy FIA pages and remap the table."""
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("QSA FIA requires matching 4-D K/V cache shapes")
    if key_cache.stride() != value_cache.stride():
        raise ValueError("QSA FIA requires matching K/V cache page strides")
    blocks, page_size, heads, dim = key_cache.shape
    if min(key_cache.shape) == 0 or page_size != GBSA_PAGE_SIZE:
        raise ValueError("QSA FIA requires nonempty 128-token paged KV cache")
    row_elements = heads * dim
    page_elements = page_size * row_elements
    for cache in (key_cache, value_cache):
        if (
            cache.stride(3) != 1
            or (heads > 1 and cache.stride(2) != dim)
            or cache.stride(1) != row_elements
            or cache.stride(0) < page_elements
            or cache.stride(0) % page_elements
        ):
            raise ValueError("QSA FIA requires dense, non-overlapping pages with whole-page gaps")
    block_stride = key_cache.stride(0) // page_elements
    physical_pages = (blocks - 1) * block_stride + 1
    # Include descriptor padding as unreferenced pages; no KV data copy is needed.
    keys = key_cache.as_strided((physical_pages, page_size, row_elements), (page_elements, row_elements, 1))
    values = value_cache.as_strided((physical_pages, page_size, row_elements), (page_elements, row_elements, 1))
    page_count = (kv_length + GBSA_PAGE_SIZE - 1) // GBSA_PAGE_SIZE
    if kv_length <= 0 or block_table.ndim != 2 or block_table.shape[1] < page_count:
        raise ValueError("QSA FIA block table must cover the positive KV length")
    table = block_table[:, :page_count].clamp_min(0).to(torch.int32)
    if block_stride != 1:
        table = table * block_stride
    return keys, values, table.contiguous(), page_count * GBSA_PAGE_SIZE


def qsa_fia_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compact_indices: torch.Tensor,
    visible_lens: torch.Tensor,
    kv_length: int,
    output: torch.Tensor,
    *,
    block_table: torch.Tensor | None = None,
    kv_lengths: list[int] | None = None,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Attend a prefill request or a batch of single-token paged Decode requests."""
    num_heads, head_dim = query.shape[1:]
    batch_size = len(kv_lengths) if kv_lengths is not None else 1
    if batch_size != 1 and (block_table is None or query.shape[0] != batch_size):
        raise ValueError("Batched QSA FIA requires one paged query per request")
    if query.dtype not in (torch.float16, torch.bfloat16) or key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("QSA FIA requires matching FP16/BF16 Q/K/V")
    if block_table is None:
        num_kv_heads = key.shape[1]
        key = key.reshape(1, kv_length, num_kv_heads * head_dim).contiguous()
        value = value.reshape(1, kv_length, num_kv_heads * head_dim).contiguous()
        mask_kv_length = kv_length
    else:
        num_kv_heads = key.shape[-1] // head_dim
        mask_kv_length = block_table.shape[1] * GBSA_PAGE_SIZE
    if query.device.type == "npu":
        mask_kv_length = (mask_kv_length + 31) // 32 * 32
    if num_heads % num_kv_heads:
        raise ValueError("QSA FIA requires query heads divisible by KV heads")

    query_tile = (
        query.shape[0]
        if batch_size != 1
        else min(QSA_FIA_QUERY_TILE, max(1, QSA_FIA_MASK_BUDGET_BYTES // mask_kv_length))
    )
    mask_workspace = None
    if query.device.type == "npu":
        # One process-local allocation is shared by all QSA layers and reused
        # across every query tile and scheduler step.
        from sglang.kernels.ops.attention.qsa_fia_mask_va import get_qsa_fia_mask_workspace

        mask_workspace = get_qsa_fia_mask_workspace(
            query.device,
            QSA_FIA_QUERY_TILE,
            QSA_FIA_MASK_BUDGET_BYTES,
            (compact_indices.shape[1] + 1 + 7) // 8 * 8,
        )
    for start in range(0, query.shape[0], query_tile):
        end = min(start + query_tile, query.shape[0])
        visible = visible_lens[start:end]
        if query.device.type == "npu":
            # Lazy import keeps model discovery and CPU semantic tests NPU-free.
            from sglang.kernels.ops.attention.qsa_fia_mask_va import build_qsa_fia_mask, copy_qsa_fia_output

            assert mask_workspace is not None
            atten_mask, has_selection = build_qsa_fia_mask(
                compact_indices[start:end], visible, kv_length, mask_kv_length, mask_workspace
            )
        else:
            selected = expand_qsa_selected_tokens(compact_indices[start:end], visible.clamp(0, kv_length))
            allowed = build_selected_token_mask(selected.unsqueeze(0), visible.unsqueeze(0), mask_kv_length)
            has_selection = allowed.any(dim=-1).flatten()
            atten_mask = (~allowed).contiguous()
            atten_mask[..., 0] &= has_selection.view_as(atten_mask[..., 0])
        query_length = (end - start) // batch_size
        atten_mask = atten_mask.reshape(batch_size, 1, query_length, mask_kv_length)
        result, _ = torch_npu.npu_fused_infer_attention_score_v2(
            query=query[start:end].reshape(batch_size, query_length, num_heads * head_dim).contiguous(),
            key=key,
            value=value,
            atten_mask=atten_mask,
            actual_seq_qlen=[query_length] * batch_size,
            actual_seq_kvlen=kv_lengths if kv_lengths is not None else [kv_length],
            num_query_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            softmax_scale=head_dim**-0.5 if softmax_scale is None else softmax_scale,
            input_layout="BSH",
            sparse_mode=0,
            inner_precise=0,
            block_table=block_table,
            block_size=GBSA_PAGE_SIZE if block_table is not None else 0,
            return_softmax_lse=False,
        )
        result = result.reshape(end - start, num_heads, head_dim)
        if query.device.type == "npu":
            copy_qsa_fia_output(result, has_selection, output[start:end])
        else:
            output[start:end].copy_(torch.where(has_selection.reshape(-1, 1, 1), result, 0))
    return output
