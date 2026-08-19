from __future__ import annotations

import torch

_PAGE_SIZE = 64
_HEAD_DIM = 128
_SCALE_BYTES = 4


def torch_paged_mqa_logits(
    q_fp8: torch.Tensor,
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    max_seq_len: int,
    *,
    query_chunk_size: int = 8,
    page_chunk_size: int = 8,
) -> torch.Tensor:
    """Compute DSA paged-MQA logits with ordinary PyTorch operations.

    The index cache stores one packed byte row per physical page. The first
    ``64 * 128`` bytes are E4M3FN keys and the final ``64 * 4`` bytes are
    per-token FP32 scales. Query rows and logical pages are chunked so this
    correctness fallback does not gather the complete paged cache at once.

    Invalid page-table entries and logical positions at or beyond ``seq_lens``
    produce ``0.0``. DSA's Top-K transform applies its own length mask, so this
    matches the existing ``clean_logits=False`` paged-MQA contract while making
    the otherwise unspecified output deterministic.
    """

    if q_fp8.ndim != 3 or q_fp8.shape[-1] != _HEAD_DIM:
        raise ValueError(
            "torch DSA paged-MQA expects q_fp8 shaped [T, H, 128], "
            f"got {tuple(q_fp8.shape)}"
        )
    if q_fp8.element_size() != 1 or not q_fp8.dtype.is_floating_point:
        raise ValueError(
            "torch DSA paged-MQA expects an FP8 query tensor, "
            f"got dtype={q_fp8.dtype}"
        )
    if weights.shape != q_fp8.shape[:2] or weights.dtype != torch.float32:
        raise ValueError(
            "torch DSA paged-MQA expects FP32 weights shaped [T, H], "
            f"got shape={tuple(weights.shape)}, dtype={weights.dtype}"
        )
    if page_table.ndim != 2 or page_table.shape[0] != q_fp8.shape[0]:
        raise ValueError(
            "torch DSA paged-MQA expects page_table shaped [T, L], "
            f"got {tuple(page_table.shape)} for T={q_fp8.shape[0]}"
        )
    if seq_lens.numel() != q_fp8.shape[0]:
        raise ValueError(
            "torch DSA paged-MQA expects one sequence length per query row, "
            f"got {seq_lens.numel()} lengths for T={q_fp8.shape[0]}"
        )
    if max_seq_len < 0:
        raise ValueError(f"max_seq_len must be non-negative, got {max_seq_len}")
    if query_chunk_size <= 0 or page_chunk_size <= 0:
        raise ValueError("query_chunk_size and page_chunk_size must be positive")

    packed_page_bytes = _PAGE_SIZE * (_HEAD_DIM + _SCALE_BYTES)
    if k_cache.ndim < 2 or k_cache.shape[0] == 0:
        raise ValueError("torch DSA paged-MQA expects at least one physical cache page")
    if k_cache[0].numel() != packed_page_bytes:
        raise ValueError(
            "torch DSA paged-MQA expects packed pages with "
            f"{packed_page_bytes} bytes, got {k_cache[0].numel()} elements"
        )
    if k_cache.element_size() != 1:
        raise ValueError(
            "torch DSA paged-MQA expects a byte-addressable packed cache, "
            f"got dtype={k_cache.dtype}"
        )

    num_queries, num_heads, _ = q_fp8.shape
    logits = torch.zeros(
        (num_queries, max_seq_len), dtype=torch.float32, device=q_fp8.device
    )
    if num_queries == 0 or max_seq_len == 0 or page_table.shape[1] == 0:
        return logits

    seq_lens = seq_lens.reshape(-1).to(device=q_fp8.device)
    packed_cache = k_cache.view(torch.uint8).reshape(k_cache.shape[0], -1)
    scale_offset = _PAGE_SIZE * _HEAD_DIM
    max_pages = min(page_table.shape[1], (max_seq_len + _PAGE_SIZE - 1) // _PAGE_SIZE)

    for query_start in range(0, num_queries, query_chunk_size):
        query_end = min(query_start + query_chunk_size, num_queries)
        q = q_fp8[query_start:query_end].float()
        w = weights[query_start:query_end]
        lengths = seq_lens[query_start:query_end]

        for page_start in range(0, max_pages, page_chunk_size):
            page_end = min(page_start + page_chunk_size, max_pages)
            page_ids = page_table[query_start:query_end, page_start:page_end].to(
                torch.long
            )
            valid_pages = (page_ids >= 0) & (page_ids < packed_cache.shape[0])
            gathered = packed_cache[page_ids.clamp(0, packed_cache.shape[0] - 1)]

            key_bytes = gathered[..., :scale_offset].contiguous()
            keys = key_bytes.view(q_fp8.dtype).reshape(
                query_end - query_start,
                page_end - page_start,
                _PAGE_SIZE,
                _HEAD_DIM,
            )
            keys = keys.reshape(
                query_end - query_start,
                (page_end - page_start) * _PAGE_SIZE,
                _HEAD_DIM,
            ).float()

            scale_bytes = gathered[..., scale_offset:].contiguous()
            scales = scale_bytes.view(torch.float32).reshape(
                query_end - query_start,
                (page_end - page_start) * _PAGE_SIZE,
            )

            qk = torch.einsum("qhd,qkd->qhk", q, keys).relu_()
            scores = torch.einsum("qhk,qh->qk", qk, w) * scales

            logical_start = page_start * _PAGE_SIZE
            logical_end = min(page_end * _PAGE_SIZE, max_seq_len)
            positions = torch.arange(
                logical_start,
                page_end * _PAGE_SIZE,
                dtype=lengths.dtype,
                device=q_fp8.device,
            )
            valid_tokens = positions.unsqueeze(0) < lengths.unsqueeze(1)
            valid_tokens &= valid_pages.repeat_interleave(_PAGE_SIZE, dim=1)
            scores.masked_fill_(~valid_tokens, 0.0)
            logits[query_start:query_end, logical_start:logical_end] = scores[
                :, : logical_end - logical_start
            ]

    return logits
