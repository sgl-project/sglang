"""Opt-in Qwen QSA Cube for short eager prefills on A3."""
import logging
from functools import lru_cache

import torch

from sgl_kernel_npu.operators.attention.qsa_cube_attention import (
    is_qsa_cube_attention_supported,
)


@lru_cache(maxsize=1)
def _report_cube():
    logging.getLogger(__name__).info("QSA short prefill uses the bundled A3 Cube kernel")


def try_short_cube(q, k, v, indices, token_table, row_to_req, positions, lengths, scale):
    rows = q.shape[0]
    if (
        not is_qsa_cube_attention_supported()
        or not 2 <= rows < 128 or q.shape[1:] != (3, 256) or scale != 1 / 16
        or q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype
        or k.shape != v.shape or k.shape[1:] != (1, 256) or k.shape[0] % 128
        or not k.is_contiguous() or not v.is_contiguous()
        or indices.shape not in ((rows, 512), (rows, 2051))
        or indices.dtype != torch.int32 or token_table.shape[1] > 65536
        or row_to_req.numel() < rows or token_table.shape[1] == 0
    ):
        return None
    request_ids = row_to_req[:rows].to(torch.int32).contiguous()
    pages = (token_table[:, ::128] // 128).to(torch.int32).contiguous()
    if indices.shape[1] == 512:
        logical_positions = positions[0] if positions.ndim == 2 else positions
        row_lens = lengths.index_select(0, request_ids.long())
        indices = torch.cat((indices, logical_positions[:rows, None].int(), row_lens[:, None].int()), dim=1)
    from sgl_kernel_npu.operators.attention import qsa_cube_attention
    output = qsa_cube_attention(
        q.contiguous(), k.view(-1, 128, 1, 256), v.view(-1, 128, 1, 256),
        indices.contiguous(), pages, request_ids)
    _report_cube()
    return output
