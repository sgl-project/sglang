from __future__ import annotations

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    topk_transform_512_pytorch_vectorized,
)


def _run_topk_transform(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_page_indices = torch.empty(
        (scores.shape[0], 512), device=scores.device, dtype=torch.int32
    )
    out_raw_indices = torch.empty_like(out_page_indices)

    topk_transform_512_pytorch_vectorized(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        out_raw_indices,
    )

    return out_page_indices, out_raw_indices


def test_topk_transform_512_uses_sequential_indices_for_short_sequences():
    page_size = 4
    seq_lens = torch.tensor([3, 6], dtype=torch.int32)
    page_tables = torch.tensor(
        [
            [10, 11],
            [20, 21],
        ],
        dtype=torch.int32,
    )
    scores = torch.arange(12, dtype=torch.float32).reshape(2, 6)

    out_page_indices, out_raw_indices = _run_topk_transform(
        scores, seq_lens, page_tables, page_size
    )

    assert out_raw_indices[0, :5].tolist() == [0, 1, 2, -1, -1]
    assert out_page_indices[0, :5].tolist() == [40, 41, 42, -1, -1]

    assert out_raw_indices[1, :8].tolist() == [0, 1, 2, 3, 4, 5, -1, -1]
    assert out_page_indices[1, :8].tolist() == [80, 81, 82, 83, 84, 85, -1, -1]


def test_topk_transform_512_handles_mixed_short_and_topk_rows():
    page_size = 8
    max_seq_len = 640
    seq_lens = torch.tensor([9, 600], dtype=torch.int32)
    page_tables = torch.arange(200, dtype=torch.int32).reshape(2, 100)
    scores = torch.zeros((2, max_seq_len), dtype=torch.float32)
    scores[1] = -10000
    scores[1, 88:600] = torch.arange(512, dtype=torch.float32)

    out_page_indices, out_raw_indices = _run_topk_transform(
        scores, seq_lens, page_tables, page_size
    )

    assert out_raw_indices[0, :12].tolist() == list(range(9)) + [-1, -1, -1]
    assert out_page_indices[0, :12].tolist() == list(range(9)) + [-1, -1, -1]

    assert (out_raw_indices[1] >= 88).all()
    assert (out_raw_indices[1] < 600).all()
    assert (out_page_indices[1] >= 0).all()
    assert out_raw_indices[1].unique().numel() == 512
