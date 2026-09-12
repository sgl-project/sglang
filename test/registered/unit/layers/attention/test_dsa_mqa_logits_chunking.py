"""Contract tests for the DSA indexer's MQA-logits chunk budget.

On ROCm the `[num_q x num_k]` fp32 logits tensor goes to aiter's
`fp8_mqa_logits`, which only compiles below 2 GiB, so the budget that decides
chunking is a correctness bound there and not only an out-of-memory guard.

The measured memory budget is stubbed: it is the only input the limit has to
beat, and stubbing it keeps these tests on CPU.
"""

from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa import dsa_indexer_kpool as kpool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

CEILING = dsa_indexer.Indexer._MQA_LOGITS_MAX_BYTES_ROCM
# More than any single logits tensor here needs, so it never decides a case.
HUGE_MEM_BUDGET = 64 * 2**30


def _decide(num_q, num_k, mem_budget=HUGE_MEM_BUDGET, is_hip=True):
    # __new__ skips an __init__ that needs a model config and a device.
    indexer = dsa_indexer.Indexer.__new__(dsa_indexer.Indexer)
    with (
        mock.patch.object(dsa_indexer, "_is_hip", is_hip),
        mock.patch.object(
            dsa_indexer.Indexer,
            "_get_mqa_logits_budget_bytes",
            return_value=mem_budget,
        ),
    ):
        return indexer._should_chunk_mqa_logits(num_q, num_k, 0)


def test_the_ceiling_is_the_largest_logits_aiter_still_takes():
    # 16384 x 32768 x 4 bytes is exactly 2 GiB, and aiter compares `bytes <
    # 2 GiB`, so that shape has to chunk and one KV token less must not.
    assert _decide(16_384, 32_768) == (True, CEILING)
    assert _decide(16_384, 32_767) == (False, CEILING)


def test_a_smaller_memory_budget_still_wins():
    one_gib = 2**30
    assert _decide(16_384, 32_767, mem_budget=one_gib) == (True, one_gib)


def test_off_rocm_the_budget_is_untouched():
    # Elsewhere the logits go to DeepGEMM, which has no such limit.
    assert _decide(16_384, 32_768, is_hip=False) == (False, HUGE_MEM_BUDGET)


# KPool builds on the same budget but caps rows by the padded HIP stride and
# must preserve per-row page/offset metadata across logits chunks.
@pytest.mark.parametrize(
    "rows,keys,expected",
    [
        (0, 65536, 0),
        (4, 0, 4),
        (3, 16, 3),
        (8191, 65536, 8191),
        (8192, 65536, 8191),
        (8192, 65535, 8191),
        (20000, 32768, 16383),
    ],
)
def test_kpool_hip_limit_includes_padded_stride(rows, keys, expected):
    with mock.patch.object(kpool, "is_hip", return_value=True):
        assert kpool.IndexerKPool._mqa_logits_chunk_rows(rows, keys) == expected
    with mock.patch.object(kpool, "is_hip", return_value=False):
        assert kpool.IndexerKPool._mqa_logits_chunk_rows(rows, keys) == rows


@pytest.mark.parametrize(
    "mapping,chunk_rows",
    [
        ("direct", 2),
        ("indexed", 8),
        ("offset", 2),
    ],
)
def test_kpool_chunked_topk_preserves_global_row_metadata(mapping, chunk_rows):
    rows, keys, padded = 5, 16, 7
    q = torch.arange(rows).reshape(rows, 1, 1).float()
    k = torch.empty(keys, 1)
    scales = torch.ones(keys)
    weights = torch.arange(rows).reshape(rows, 1).float()
    starts = torch.tensor([0, 0, 8, 8, 8], dtype=torch.int32)
    ends = torch.tensor([2, 4, 11, 13, 16], dtype=torch.int32)
    lengths = ends - starts
    seq_lens = lengths * 4 + 3
    table = torch.arange(5 * 64).reshape(5, 64)
    row_indices = torch.tensor([2, 2, 0, 0, 1], dtype=torch.int32)
    offsets = torch.tensor([11, 12, 13, 14, 15], dtype=torch.int32)
    calls = []

    def logits(qc, kc, sc, wc, startc, endc):
        begin = int(qc[0, 0, 0])
        stop = begin + qc.shape[0]
        calls.append((begin, stop))
        assert kc is k and sc is scales
        torch.testing.assert_close(wc, weights[begin:stop])
        torch.testing.assert_close(startc, starts[begin:stop])
        torch.testing.assert_close(endc, ends[begin:stop])
        return qc[:, 0, :].expand(-1, keys)

    def topk(lc, lengths_c, **kwargs):
        begin = int(lc[0, 0])
        stop = begin + lc.shape[0]
        torch.testing.assert_close(lengths_c, lengths[begin:stop])
        torch.testing.assert_close(kwargs["seq_lens"], seq_lens[begin:stop])
        torch.testing.assert_close(kwargs["row_starts"], starts[begin:stop])
        if mapping == "direct":
            torch.testing.assert_close(kwargs["page_table"], table[begin:stop])
            assert kwargs["page_table_row_index"] is None
        elif mapping == "indexed":
            assert kwargs["page_table"] is table
            torch.testing.assert_close(
                kwargs["page_table_row_index"], row_indices[begin:stop]
            )
        else:
            assert kwargs["page_table"] is None
            torch.testing.assert_close(kwargs["topk_offsets"], offsets[begin:stop])
        return lc[:, :3].to(torch.int32)

    indexer = mock.MagicMock()
    indexer._mqa_logits_chunk_rows.side_effect = lambda m, n: min(m, chunk_rows)
    indexer._ragged_mqa_logits.side_effect = logits
    indexer._topk_from_kpool_logits.side_effect = topk
    result = kpool.IndexerKPool._topk_from_ragged_kpool(
        indexer,
        q,
        k,
        scales,
        weights,
        starts,
        ends,
        lengths,
        seq_lens,
        page_table=table if mapping != "offset" else None,
        page_table_row_index=row_indices if mapping == "indexed" else None,
        topk_offsets=offsets if mapping == "offset" else None,
        out_rows=padded,
    )
    assert calls == ([(0, 2), (2, 4), (4, 5)] if chunk_rows == 2 else [(0, 5)])
    torch.testing.assert_close(
        result[:rows], torch.arange(rows)[:, None].expand(-1, 3).int()
    )
    assert result.shape == (padded, 3)
    assert bool((result[rows:] == -1).all())


@pytest.mark.parametrize("rows,keys,out_rows", [(0, 16, 0), (0, 16, 3), (4, 0, 4)])
def test_kpool_empty_rows_or_keys_do_not_launch_logits(rows, keys, out_rows):
    def no_logits(*args):
        pytest.fail("Empty queries or keys must not launch AITER")

    indexer = mock.MagicMock()
    indexer._mqa_logits_chunk_rows.side_effect = (
        kpool.IndexerKPool._mqa_logits_chunk_rows
    )
    indexer._ragged_mqa_logits.side_effect = no_logits
    indexer._topk_from_kpool_logits.side_effect = lambda logits, lengths, **kwargs: (
        torch.full((logits.shape[0], 3), -1, dtype=torch.int32)
    )
    lengths = torch.zeros(rows, dtype=torch.int32)
    result = kpool.IndexerKPool._topk_from_ragged_kpool(
        indexer,
        torch.empty(rows, 1, 1),
        torch.empty(keys, 1) if keys else None,
        None,
        torch.empty(rows, 1),
        lengths,
        lengths,
        lengths,
        lengths,
        out_rows=out_rows,
    )
    assert result.shape == (out_rows, 3)
    assert bool((result == -1).all())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
