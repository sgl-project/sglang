"""Pure-torch references for the DSA page-table index transform kernels."""

from typing import List, Optional

import torch

from sglang.kernels.ops.attention.dsa.transform_index import _allocate_prefill_result


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.shape == topk_indices.shape
    torch.gather(
        page_table.to(result.dtype),
        dim=1,
        index=topk_indices.clamp(min=0),
        out=result,
    )
    result[topk_indices < 0] = -1
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
    output_num_tokens: Optional[int] = None,
    page_table_is_expanded: bool = False,
) -> torch.Tensor:
    assert page_size == 1
    real_num_tokens = sum(extend_lens_cpu)
    result = _allocate_prefill_result(topk_indices, real_num_tokens, output_num_tokens)

    if page_table_is_expanded:
        if real_num_tokens > 0:
            transform_index_page_table_decode_ref(
                page_table[:real_num_tokens],
                topk_indices[:real_num_tokens],
                result=result[:real_num_tokens],
            )
        return result

    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    return result
