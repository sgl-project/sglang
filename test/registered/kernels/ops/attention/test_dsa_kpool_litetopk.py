# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the guarded FP32 KPool4 prefill path."""

from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsa.kpool_topk_transform import map_kpool4_groups
from sglang.kernels.ops.attention.dsa.litetopk_prefill.fp8_indexer import (
    fp8_indexer_candidates,
    is_fp8_indexer_supported,
)
from sglang.kernels.ops.attention.dsa.litetopk_prefill.topk_candidates import (
    topk_candidates,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.skipif(not is_fp8_indexer_supported(), reason="SM100 required")
def test_fp32_candidates_match_dense_with_signed_gates_on_two_streams():
    import deep_gemm

    rows_per_request, keys_per_request = 37, 16384
    rows, keys = 2 * rows_per_request, 2 * keys_per_request
    generator = torch.Generator(device="cuda").manual_seed(74)
    q = torch.randn(rows, 32, 128, generator=generator, device="cuda").to(
        torch.float8_e4m3fn
    )
    k = torch.randn(keys, 128, generator=generator, device="cuda").to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(rows, 32, generator=generator, device="cuda")
    scale = torch.rand(keys, generator=generator, device="cuda").add_(0.125)
    starts = torch.tensor(
        [0, keys_per_request], device="cuda", dtype=torch.int32
    ).repeat_interleave(rows_per_request)
    pool_lens = torch.arange(
        keys_per_request - rows_per_request + 1,
        keys_per_request + 1,
        device="cuda",
        dtype=torch.int32,
    ).repeat(2)
    ends = starts + pool_lens
    expected = deep_gemm.fp8_mqa_logits(
        q, (k, scale), weights, starts, ends, clean_logits=True
    )

    current = torch.cuda.current_stream()
    candidates = []
    for tail_chunks in (1, 4):
        stream = torch.cuda.Stream()
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            slab = fp8_indexer_candidates(
                q,
                weights,
                k,
                scale,
                starts,
                ends,
                512,
                cap=4096,
                num_ctas=4,
                tail_chunks=tail_chunks,
                fp32_reduction=True,
            )
        candidates.append((stream, slab))

    for stream, slab in candidates:
        current.wait_stream(stream)
        live = torch.arange(4096, device="cuda")[None, :] < slab.counts[:, None]
        absolute = torch.where(live, starts[:, None] + slab.indices, 0).long()
        assert (slab.indices[live] >= 0).all()
        assert (absolute[live] < ends[:, None].expand_as(absolute)[live]).all()
        torch.testing.assert_close(
            slab.scores[live], expected.gather(1, absolute)[live], atol=0, rtol=0
        )


@pytest.mark.skipif(not is_fp8_indexer_supported(), reason="SM100 required")
def test_guarded_candidate_selector_rejects_ambiguous_rows():
    cap, topk = 4096, 512
    scores = torch.arange(cap, device="cuda", dtype=torch.float32).repeat(8, 1)
    indices = torch.arange(cap, device="cuda", dtype=torch.int32).repeat(8, 1)
    count = torch.tensor(
        [2048, 511, cap + 1, 2048, 2048, 2048, 2048, 2048],
        device="cuda",
        dtype=torch.int32,
    )
    threshold = torch.zeros(8, device="cuda")
    scores[3, 1535] = 1536  # both sides of the cutoff have the same score
    scores[4, 1300:1800] = 1536  # the cutoff bin exceeds the pivot buffer
    scores[5, 7] = torch.nan
    threshold[6] = -torch.inf
    rejected = torch.tensor([0, 0, 0, 0, 0, 0, 0, 2], device="cuda", dtype=torch.int32)
    selected = torch.empty((8, topk), device="cuda", dtype=torch.int32)
    topk_candidates(scores, indices, count, threshold, selected, reject=rejected)
    assert rejected.tolist() == [0, 1, 1, 1, 1, 1, 1, 1]
    assert selected[0].sort().values.tolist() == list(range(1536, 2048))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("paged", [False, True])
def test_kpool_mapping_keeps_tail_positions_with_request_row_indirection(paged):
    groups = torch.arange(511, -1, -1, device="cuda", dtype=torch.int32).repeat(3, 1)
    groups[1, :3] = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    pool_lens = torch.tensor([520, 3, 513], device="cuda", dtype=torch.int32)
    seq_lens = torch.tensor([2083, 13, 2054], device="cuda", dtype=torch.int32)
    reject = torch.tensor([0, 0, 1], device="cuda", dtype=torch.int32)
    page_width = 2100
    page_table = torch.arange(page_width * 4, device="cuda", dtype=torch.int32).reshape(
        4, page_width
    )
    page_rows = torch.tensor([3, 1, 2], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([41, 18, 100], device="cuda", dtype=torch.int32)
    actual = map_kpool4_groups(
        groups,
        reject,
        pool_lens,
        seq_lens,
        page_table if paged else None,
        page_rows if paged else None,
        None if paged else offsets,
    )
    long_history = [group * 4 + i for group in range(511, -1, -1) for i in range(4)]
    short_history = [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]
    bases = [3 * page_width, page_width] if paged else [41, 18]
    expected = [
        [bases[0] + i for i in long_history + [2080, 2081, 2082]],
        [bases[1] + i for i in short_history + [12]] + [-1] * (2051 - 13),
        [-1] * 2051,
    ]
    assert actual.tolist() == expected


def test_dense_fallback_routes_rejected_rows_and_keeps_padding(monkeypatch):
    from sglang.srt.layers.attention.dsa import dsa_indexer_kpool, kpool_litetopk

    q = torch.arange(6, dtype=torch.uint8).reshape(6, 1, 1).view(torch.float8_e4m3fn)
    starts = torch.arange(6, dtype=torch.int32) * 10
    lengths = torch.arange(6, dtype=torch.int32) + 10
    page_table = torch.zeros((5, 3), dtype=torch.int32)
    page_rows = torch.tensor([3, 3, 1, 4, 4, 2], dtype=torch.int32)
    dense_rows = []

    def score(q_fp8, **kwargs):
        dense_rows.extend(q_fp8.view(torch.uint8).flatten().tolist())
        return kwargs["starts"]

    def dense_topk(*, logits, pool_lens, page_table_row_index, **kwargs):
        assert kwargs["page_table"] is page_table
        return torch.stack((logits, pool_lens, page_table_row_index), dim=1)

    def candidate(q_slice, *args, **kwargs):
        if q_slice.view(torch.uint8)[0, 0, 0].item() == 0:
            return torch.full((3, 3), 100, dtype=torch.int32), torch.tensor([1])
        return None

    monkeypatch.setattr(kpool_litetopk, "try_kpool4_lite_topk", candidate)
    monkeypatch.setattr(
        dsa_indexer_kpool,
        "deep_gemm",
        SimpleNamespace(get_num_sms=lambda: 1),
        raising=False,
    )
    indexer = SimpleNamespace(_fp8_mqa_logits=score, _topk_from_kpool_logits=dense_topk)
    actual = dsa_indexer_kpool.IndexerKPool._kpool_topk_by_row_chunks(
        indexer,
        q_fp8=q,
        weights=torch.zeros((6, 1)),
        kv_fp8=(torch.empty((1, 1)), torch.ones(1)),
        logits_starts=starts,
        logits_ends=starts + lengths,
        pool_lens=lengths,
        seq_lens=lengths * 4,
        page_table=page_table,
        page_table_row_index=page_rows,
        topk_offsets=None,
        topk_row_starts=starts,
        row_chunks=(slice(0, 3), slice(3, 6)),
        out_rows=8,
        use_lite_topk=True,
    )
    assert dense_rows == [1, 3, 4, 5]
    assert actual.tolist() == [
        [100, 100, 100],
        [10, 11, 3],
        [100, 100, 100],
        [30, 13, 4],
        [40, 14, 4],
        [50, 15, 2],
        [-1, -1, -1],
        [-1, -1, -1],
    ]
