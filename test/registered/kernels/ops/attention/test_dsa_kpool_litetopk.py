# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the guarded KPool4 prefill path."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
from triton.compiler.errors import CompilationError
from triton.runtime.errors import PTXASError

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
def test_native_fp8_candidates_reject_nonfinite_scores_in_visible_keys():
    generator = torch.Generator(device="cuda").manual_seed(52591)
    keys_per_request = 16384
    q = torch.randn(8, 32, 128, generator=generator, device="cuda").to(
        torch.float8_e4m3fn
    )
    k = torch.randn(2 * keys_per_request, 128, generator=generator, device="cuda").to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(8, 32, generator=generator, device="cuda")
    scale = torch.ones(2 * keys_per_request, device="cuda")
    scale[267] = torch.nan
    starts = torch.tensor(
        [0, keys_per_request], device="cuda", dtype=torch.int32
    ).repeat_interleave(4)

    slab = fp8_indexer_candidates(
        q,
        weights,
        k,
        scale,
        starts,
        starts + keys_per_request,
        512,
        cap=4096,
        num_ctas=4,
        tail_chunks=1,
        track_rejections=True,
    )
    assert (slab.overflow & 2).tolist() == [2] * 4 + [0] * 4


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
@pytest.mark.parametrize("group_topk", [160, 512])
def test_kpool_mapping_keeps_tail_positions_with_request_row_indirection(
    paged, group_topk
):
    groups = torch.arange(
        group_topk - 1, -1, -1, device="cuda", dtype=torch.int32
    ).repeat(6, 1)
    groups[1, :3] = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    groups[3, :2] = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    pool_lens = torch.tensor(
        [group_topk + 8, 3, group_topk + 1, 2, group_topk, 0],
        device="cuda",
        dtype=torch.int32,
    )
    seq_lens = pool_lens * 4 + torch.tensor(
        [3, 1, 2, 3, 0, 2], device="cuda", dtype=torch.int32
    )
    reject = torch.tensor([0, 0, 1, 0, 0, 0], device="cuda", dtype=torch.int32)
    page_width = 4 * group_topk + 64
    page_table = torch.arange(page_width * 4, device="cuda", dtype=torch.int32).reshape(
        4, page_width
    )
    page_rows = torch.tensor([3, 1, 2, 0, 2, 3], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([41, 18, 100, 9, 53, 11], device="cuda", dtype=torch.int32)
    actual = map_kpool4_groups(
        groups,
        reject,
        pool_lens,
        seq_lens,
        page_table if paged else None,
        page_rows if paged else None,
        None if paged else offsets,
    )
    full_history = [
        group * 4 + i for group in range(group_topk - 1, -1, -1) for i in range(4)
    ]
    tail_start = (group_topk + 8) * 4
    visible_tokens = [
        full_history + list(range(tail_start, tail_start + 3)),
        [8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 12],
        [],
        [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10],
        full_history,
        [0, 1],
    ]
    bases = (page_rows * page_width if paged else offsets).tolist()
    width = 4 * group_topk + 3
    expected = [
        [base + token for token in tokens] + [-1] * (width - len(tokens))
        for base, tokens in zip(bases, visible_tokens)
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


@pytest.mark.parametrize(
    "error",
    [
        CompilationError(None, None, "mapper compilation failed"),
        PTXASError("mapper assembly failed"),
        RuntimeError("mapper launch failed"),
    ],
    ids=["compile", "ptxas", "launch"],
)
def test_mapper_falls_back_only_on_compilation_errors(monkeypatch, error):
    from sglang.srt.layers.attention.dsa import kpool_litetopk

    monkeypatch.setenv("SGLANG_OPT_USE_KPOOL4_LITE_TOPK_NATIVE", "0")
    rows = kpool_litetopk._MIN_ROWS
    keys = kpool_litetopk._MIN_KEYS
    q = torch.empty((rows, 32, 128), dtype=torch.float8_e4m3fn)
    k = torch.empty((keys, 128), dtype=torch.float8_e4m3fn)
    weights = torch.empty((rows, 32), dtype=torch.float32)
    scale = torch.empty(keys, dtype=torch.float32)
    metadata = torch.zeros(rows, dtype=torch.int32)
    slab = SimpleNamespace(
        scores=None, indices=None, counts=None, thresholds=None, overflow=metadata
    )

    def fail_mapper(*args, **kwargs):
        raise error

    monkeypatch.setattr(kpool_litetopk, "is_fp8_indexer_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(kpool_litetopk, "_jit_available", lambda native: True)
    monkeypatch.setattr(kpool_litetopk, "fp8_indexer_candidates", lambda *a, **kw: slab)
    monkeypatch.setattr(kpool_litetopk, "topk_candidates", lambda *a, **kw: None)
    monkeypatch.setattr(kpool_litetopk, "map_kpool4_groups", fail_mapper)

    args = (
        q,
        weights,
        k,
        scale,
        metadata,
        metadata,
        metadata,
        metadata,
        None,
        None,
        metadata,
    )
    if isinstance(error, RuntimeError):
        with pytest.raises(RuntimeError) as exc:
            kpool_litetopk.try_kpool4_lite_topk(*args, num_sms=1)
        assert exc.value is error
    else:
        assert kpool_litetopk.try_kpool4_lite_topk(*args, num_sms=1) is None


@pytest.mark.skipif(not is_fp8_indexer_supported(), reason="SM100 required")
@pytest.mark.parametrize("paged", [False, True])
@torch.inference_mode()
def test_exact_indexer_matches_dense_after_partial_recovery(monkeypatch, paged):
    from sglang.srt.layers.attention.dsa import dsa_indexer_kpool, kpool_litetopk

    monkeypatch.setenv("SGLANG_OPT_USE_KPOOL4_LITE_TOPK_NATIVE", "0")
    rows, keys_per_request = 4096, 24576
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(52591)
    q = torch.randn(rows, 32, 128, generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    k = torch.randn(2 * keys_per_request, 128, generator=generator, device=device).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(rows, 32, generator=generator, device=device)
    scale = torch.rand(2 * keys_per_request, generator=generator, device=device).add_(
        0.125
    )
    starts = torch.tensor([0, keys_per_request], device=device, dtype=torch.int32)
    starts = starts.repeat_interleave(rows // 2)
    pool_lens = torch.arange(
        keys_per_request - rows // 2 + 1,
        keys_per_request + 1,
        device=device,
        dtype=torch.int32,
    ).repeat(2)
    pool_lens[0] = 256  # This row cannot fill the candidate top-k.
    seq_lens = 4 * pool_lens + torch.arange(rows, device=device, dtype=torch.int32) % 4
    seq_lens[0] += 2
    if paged:
        width = 4 * keys_per_request + 3
        page_table = torch.arange(3 * width, device=device, dtype=torch.int32).reshape(
            3, width
        )
        page_rows = torch.tensor([2, 0], device=device, dtype=torch.int32)
        page_rows = page_rows.repeat_interleave(rows // 2)
        offsets = None
    else:
        page_table = page_rows = None
        offsets = torch.tensor([137, 100203], device=device, dtype=torch.int32)
        offsets = offsets.repeat_interleave(rows // 2)

    cls = dsa_indexer_kpool.IndexerKPool
    indexer = SimpleNamespace(
        index_kpool=4, index_topk=2048, _fp8_mqa_logits=cls._fp8_mqa_logits
    )
    indexer._topk_from_kpool_logits = partial(cls._topk_from_kpool_logits, indexer)
    run = partial(
        cls._kpool_topk_by_row_chunks,
        indexer,
        q_fp8=q,
        weights=weights,
        kv_fp8=(k, scale),
        logits_starts=starts,
        logits_ends=starts + pool_lens,
        pool_lens=pool_lens,
        seq_lens=seq_lens,
        page_table=page_table,
        page_table_row_index=page_rows,
        topk_offsets=offsets,
        topk_row_starts=starts,
        row_chunks=(slice(0, rows),),
        out_rows=rows + 2,
    )
    original = kpool_litetopk.try_kpool4_lite_topk
    recovered = []

    def record_rejected_rows(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result is not None
        recovered.extend(result[1].tolist())
        return result

    monkeypatch.setattr(kpool_litetopk, "try_kpool4_lite_topk", record_rejected_rows)
    dense = run(use_lite_topk=False)
    candidate = run(use_lite_topk=True)
    assert 0 in recovered and len(recovered) < rows
    assert torch.equal(candidate[rows:], dense[rows:])
    assert (candidate[rows:] == -1).all()
    dense, candidate = dense[:rows], candidate[:rows]
    end = (pool_lens * 4).clamp(max=2048)
    history = torch.arange(dense.shape[1], device=device)[None, :] < end[:, None]
    assert torch.equal(
        torch.where(history, candidate, -1).sort(dim=1).values,
        torch.where(history, dense, -1).sort(dim=1).values,
    )
    assert torch.equal(
        torch.where(history, -1, candidate), torch.where(history, -1, dense)
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
