"""Row-slicing equivalence for the DSV4 indexer logits/top-k chunking.

The chunked path computes logits and top-k over query-row slices. This test
pins the property that makes that safe: every row of the top-k transform is a
pure function of its own logits row, sequence length and page-table row, so
slicing the batch cannot change any row's output.

Runs on CPU against the pure-torch reference transform, so it needs no GPU.
"""

import torch

from sglang.srt.layers.attention.dsv4.indexer import (
    _mqa_logits_chunk_rows,
    _mqa_logits_row_bytes,
    topk_transform_512_pytorch_vectorized,
)

PAGE_SIZE = 64


def _run(logits, seq_lens, page_table, rows):
    out = torch.full((rows, 512), -1, dtype=torch.int32)
    topk_transform_512_pytorch_vectorized(
        logits, seq_lens, page_table, out, PAGE_SIZE, None
    )
    return out


def test_chunked_topk_matches_unchunked():
    torch.manual_seed(0)
    rows, width = 37, 2048
    logits = torch.randn(rows, width, dtype=torch.float32)
    seq_lens = torch.randint(1, width, (rows,), dtype=torch.int32)
    page_table = torch.randint(
        0, 4096, (rows, (width + PAGE_SIZE - 1) // PAGE_SIZE), dtype=torch.int32
    )

    expected = _run(logits, seq_lens, page_table, rows)

    for chunk in (1, 7, 16, rows - 1, rows):
        got = torch.full((rows, 512), -1, dtype=torch.int32)
        for start in range(0, rows, chunk):
            end = min(start + chunk, rows)
            topk_transform_512_pytorch_vectorized(
                logits[start:end],
                seq_lens[start:end],
                page_table[start:end],
                got[start:end],
                PAGE_SIZE,
                None,
            )
        assert torch.equal(got, expected), f"chunk={chunk} diverged"


def test_row_bytes_matches_deepgemm_alignment():
    # DeepGEMM pads the fp32 row stride to 256 columns.
    assert _mqa_logits_row_bytes(1) == 256 * 4
    assert _mqa_logits_row_bytes(256) == 256 * 4
    assert _mqa_logits_row_bytes(257) == 512 * 4


def test_chunk_rows_disabled_without_cuda(monkeypatch):
    # No measurable budget (no CUDA) must keep the unchunked fast path.
    monkeypatch.setattr(
        "sglang.srt.layers.attention.dsv4.indexer._mqa_logits_budget_bytes",
        lambda device: None,
    )
    assert _mqa_logits_chunk_rows(1 << 20, 1 << 20, torch.device("cpu")) is None


def test_chunk_rows_slices_when_over_budget(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.layers.attention.dsv4.indexer._mqa_logits_budget_bytes",
        lambda device: 512 << 20,
    )
    # 4096 rows x 93184 cols fp32 = 1.42 GiB > 512 MiB budget -> must slice.
    chunk = _mqa_logits_chunk_rows(4096, 93184, torch.device("cpu"))
    assert chunk is not None and chunk < 4096
    assert chunk * _mqa_logits_row_bytes(93184) <= (512 << 20)
    # Within budget -> no slicing.
    assert _mqa_logits_chunk_rows(64, 93184, torch.device("cpu")) is None
