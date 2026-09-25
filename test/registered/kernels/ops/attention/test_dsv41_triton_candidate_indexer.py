"""Triton candidate-only prefill against an independent dense-and-mask oracle.

Runs on Hopper (no DeepGEMM dependency) and also on CPU for contract tests.
"""

import dataclasses
import sys

import pytest
import torch

from sglang.srt.layers.attention.dsv4.candidate_indexer import (
    PrefillCandidateBlocks,
    TritonPrefillInputs,
)
from sglang.srt.layers.attention.dsv4.triton_candidate_indexer import (
    TritonCandidateIndexer,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _reference_candidate_scores(q, keys, weights, blocks, lens):
    """Independent test-only dense scoring, then select candidate score columns.

    CPU contract tests substitute this scorer; GPU integration calls Triton.
    These tests do not validate the CUDA kernel. GPU integration does not inject
    this scorer. Production never uses this oracle.
    """
    scores = torch.einsum("rhd,nd->rhn", q, keys)
    scores = (scores.relu() * weights[:, :, None]).sum(1).float()
    positions = (
        blocks.long()[:, :, None] * 8 + torch.arange(8, device=q.device)
    ).flatten(1)
    result = scores.gather(1, positions.clamp(0, keys.shape[0] - 1))
    return result.masked_fill(
        (positions < 0) | (positions >= keys.shape[0]) | (positions >= lens[:, None]),
        -torch.inf,
    )


@pytest.fixture(autouse=True)
def contract_scorer(request, monkeypatch):
    if request.node.originalname not in {
        "test_triton_prefill_integration_no_gather",
        "test_unsupported_sparse_rejected",
    }:
        monkeypatch.setattr(
            TritonCandidateIndexer,
            "_candidate_scores",
            staticmethod(_reference_candidate_scores),
        )


def _case(dtype, rows=(5, 0, 4, 2), lengths=(37, 8, 11, 0), *, heads=4, dim=16):
    torch.manual_seed(71)
    total = sum(rows)
    q = torch.randn(total, heads, dim, device=DEVICE).to(dtype)
    weights = torch.rand(total, heads, device=DEVICE).to(dtype)
    keys = [torch.randn(n, dim, device=DEVICE).to(dtype) for n in lengths]
    lens, starts, base = [], [], 0
    for r, n in zip(rows, lengths):
        lens.extend([min(n, i * max(1, n // max(1, r - 1))) for i in range(r)])
        starts.extend([base] * r)
        base += n
    inputs = TritonPrefillInputs(
        q=q,
        weights=weights,
        compress_lens=torch.tensor(lens, device=DEVICE, dtype=torch.int32),
        request_starts=torch.tensor(starts, device=DEVICE, dtype=torch.int32),
        lens_per_request=list(lengths),
        rows_per_request=list(rows),
        get_keys=keys.__getitem__,
    )
    return inputs


def _out(inputs, k=7):
    return torch.full((inputs.num_rows, k), 123, dtype=torch.int32, device=DEVICE)


def _dense(inputs, b, start, rows):
    q, k = inputs.q[start : start + rows], inputs.get_keys(b)
    s = torch.einsum("rhd,nd->rhn", q, k)
    s = (s.relu() * inputs.weights[start : start + rows, :, None]).sum(1).float()
    return s.masked_fill(
        torch.arange(k.shape[0], device=DEVICE)[None]
        >= inputs.compress_lens[start : start + rows, None],
        -torch.inf,
    )


def _check_selection(out, scores, offset, k, *, tolerance):
    # Compare scores at the cutoff rather than requiring arbitrary tie ordering.
    for r in range(scores.shape[0]):
        selected = out[r][out[r] >= 0].long() - offset[r]
        expected_n = min(k, int((scores[r] > -torch.inf).sum()))
        assert selected.numel() == expected_n
        assert selected.unique().numel() == expected_n
        assert ((selected >= 0) & (selected < scores.shape[1])).all()
        assert (out[r][out[r] < 0] == -1).all()
        if expected_n:
            floor = scores[r].topk(expected_n).values[-1]
            assert (scores[r, selected] > -torch.inf).all()
            assert (
                scores[r, selected] >= floor - tolerance * floor.abs().clamp_min(1)
            ).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("budget", [1, 1 << 20])
@torch.inference_mode()
def test_publish_and_candidate_select(dtype, budget):
    inputs = _case(dtype)
    impl = TritonCandidateIndexer(2, 8, budget_bytes=budget)
    own = _out(inputs)
    published = impl.publish_prefill(inputs, own)
    # Consumers must use their own Q and weights, not cached source scores.
    consumer = dataclasses.replace(
        inputs, q=inputs.q.flip(1), weights=inputs.weights.flip(0)
    )
    selected = _out(consumer)
    impl.select_prefill(published, consumer, selected)
    start = 0
    for b, (rows, length) in enumerate(
        zip(inputs.rows_per_request, inputs.lens_per_request)
    ):
        blocks = published.request_blocks[b]
        if not rows or not length:
            assert (own[start : start + rows] == -1).all()
            assert (selected[start : start + rows] == -1).all()
            start += rows
            continue
        scores = _dense(inputs, b, start, rows)
        # Independent block maxima and publication reference.
        maxima = torch.stack(
            [scores[:, j : j + 8].amax(1) for j in range(0, length, 8)], 1
        )
        last = (inputs.compress_lens[start : start + rows] - 1) // 8
        for r in range(rows):
            if last[r] >= 0:
                maxima[r, last[r]] = torch.inf
            got = blocks[r][blocks[r] >= 0].long()
            n = min(2, int((maxima[r] > -torch.inf).sum()))
            assert got.numel() == n and got.unique().numel() == n
            if n:
                assert (maxima[r, got] >= maxima[r].topk(n).values[-1]).all()
                assert (got == last[r]).any()
        _check_selection(
            own[start : start + rows],
            scores,
            inputs.request_starts[start : start + rows],
            7,
            tolerance=0.032 if dtype == torch.bfloat16 else 1e-5,
        )
        # Build a full-width mask without using the implementation's gather map.
        ref = _dense(consumer, b, start, rows)
        mask = torch.zeros_like(ref, dtype=torch.bool)
        for r in range(rows):
            for block in blocks[r].tolist():
                if block >= 0:
                    mask[r, block * 8 : min(length, (block + 1) * 8)] = True
        ref.masked_fill_(~mask, -torch.inf)
        _check_selection(
            selected[start : start + rows],
            ref,
            inputs.request_starts[start : start + rows],
            7,
            tolerance=0.032 if dtype == torch.bfloat16 else 1e-5,
        )
        start += rows


@torch.inference_mode()
def test_tail_empty_rows_and_ties():
    inputs = _case(torch.bfloat16)
    inputs.q.zero_()  # all scores equal; any valid cutoff tie is acceptable
    impl = TritonCandidateIndexer(2, 8, budget_bytes=1)
    published = impl.publish_prefill(inputs, _out(inputs))
    tails = [2, 0, 0, 1]
    trimmed = impl.prefill_tail(published, tails)
    ids, start = [], 0
    for blocks, rows, count in zip(
        published.request_blocks, inputs.rows_per_request, tails
    ):
        ids.extend(range(start + rows - count, start + rows))
        start += rows
    ids = torch.tensor(ids, device=DEVICE, dtype=torch.long)
    sub = dataclasses.replace(
        inputs,
        q=inputs.q[ids],
        weights=inputs.weights[ids],
        compress_lens=inputs.compress_lens[ids],
        request_starts=inputs.request_starts[ids],
        rows_per_request=tails,
    )
    got = _out(sub)
    impl.select_prefill(trimmed, sub, got)
    assert [b.shape[0] for b in trimmed.request_blocks] == tails
    assert (got[-1] == -1).all()
    for r in range(2):
        positions = got[r][got[r] >= 0]
        assert positions.unique().numel() == positions.numel()
        assert (positions < sub.compress_lens[r]).all()
    with pytest.raises(ValueError):
        impl.prefill_tail(published, [6, 0, 0, 0])


@torch.inference_mode()
def test_consumer_scores_only_candidates_and_masks_padding():
    inputs = _case(torch.float32, rows=(3,), lengths=(65,))
    inputs.compress_lens.fill_(65)
    blocks = torch.tensor(
        [[1, -1], [8, -1], [-1, -1]], device=DEVICE, dtype=torch.int32
    )
    published = PrefillCandidateBlocks(request_blocks=[blocks])
    impl = TritonCandidateIndexer(2, 8, budget_bytes=1)
    shapes = []

    def record(q, keys, weights, blocks, lens):
        shapes.append((keys.shape, blocks.shape))
        return _reference_candidate_scores(q, keys, weights, blocks, lens)

    impl._candidate_scores = record
    got = _out(inputs)
    impl.select_prefill(published, inputs, got)
    assert shapes == [(torch.Size([65, 16]), torch.Size([1, 2]))] * 3
    assert ((got[0] >= 8) & (got[0] < 16)).all()
    assert got[1, 0] == 64 and (got[1, 1:] == -1).all()
    assert (got[2] == -1).all()


@torch.inference_mode()
def test_real_candidate_window_and_plain_reference():
    # Cross the actual 2048 * 8 candidate boundary, including a partial block.
    inputs = _case(torch.bfloat16, rows=(3,), lengths=(16393,), heads=32, dim=128)
    inputs.compress_lens[-1] = 16393
    impl = TritonCandidateIndexer(2048, 8, budget_bytes=2 << 20)
    own = _out(inputs, 512)
    published = impl.publish_prefill(inputs, own)
    assert published.request_blocks[0].shape == (3, 2048)
    plain = _out(inputs, 512)
    impl.plain_prefill(inputs, plain)
    assert torch.equal(own, plain)
    selected = _out(inputs, 512)
    impl.select_prefill(published, inputs, selected)
    ref = _dense(inputs, 0, 0, 3)
    for r, blocks in enumerate(published.request_blocks[0]):
        keep = torch.zeros(16393, dtype=torch.bool, device=DEVICE)
        for b in blocks[blocks >= 0].tolist():
            keep[b * 8 : min((b + 1) * 8, 16393)] = True
        ref[r].masked_fill_(~keep, -torch.inf)
    _check_selection(selected, ref, inputs.request_starts, 512, tolerance=2**-5)


@torch.inference_mode()
def test_empty_batch_and_tail_selection_equivalence():
    impl = TritonCandidateIndexer(2, 8, budget_bytes=1)
    empty = _case(torch.float32, rows=(), lengths=())
    table = impl.publish_prefill(empty, _out(empty))
    impl.select_prefill(table, empty, _out(empty))
    assert table.request_blocks == []
    assert impl.prefill_tail(table, []).request_blocks == []

    inputs = _case(torch.float32)
    table = impl.publish_prefill(inputs, _out(inputs))
    full = _out(inputs)
    impl.select_prefill(table, inputs, full)
    tails = [2, 0, 3, 0]
    ids, start = [], 0
    for n, t in zip(inputs.rows_per_request, tails):
        ids.extend(range(start + n - t, start + n))
        start += n
    ids = torch.tensor(ids, device=DEVICE, dtype=torch.long)
    sub = dataclasses.replace(
        inputs,
        q=inputs.q[ids],
        weights=inputs.weights[ids],
        compress_lens=inputs.compress_lens[ids],
        request_starts=inputs.request_starts[ids],
        rows_per_request=tails,
    )
    out = _out(sub)
    impl.select_prefill(impl.prefill_tail(table, tails), sub, out)
    assert torch.equal(full[ids].sort().values, out.sort().values)


@pytest.mark.parametrize(
    "sm,oracle,cp", [(90, False, 1), (100, False, 1), (100, False, 2), (100, True, 1)]
)
def test_factory_prefill_and_decode_selection(monkeypatch, sm, oracle, cp):
    """Test routing without importing or executing any SM100 kernels."""
    import sys
    from types import ModuleType, SimpleNamespace

    from sglang.srt.layers.attention.dsv4.candidate_indexer import (
        make_candidate_indexer,
    )

    class Dense:
        def __init__(self, *args):
            pass

    class DeepGemm:
        def __init__(self, *args, prefill_dense=None):
            self.prefill_dense = prefill_dense

    modules = {
        "sglang.srt.environ": {
            "envs": SimpleNamespace(
                SGLANG_DSV41_TORCH_PREFILL_INDEXER=SimpleNamespace(get=lambda: oracle),
            )
        },
        "sglang.srt.runtime_context": {
            "get_platform": lambda: SimpleNamespace(device_sm=sm),
            "get_parallel": lambda: SimpleNamespace(attn_cp_size=cp),
        },
        "sglang.srt.layers.deep_gemm_wrapper.configurer": {
            "DEEPGEMM_PAGED_SPARSE_MQA_LOGITS": True
        },
        "sglang.srt.layers.attention.dsv4.candidate_indexer_deep_gemm": {
            "DeepGemmCandidateIndexer": DeepGemm
        },
        "sglang.srt.layers.attention.dsv4.dense_prefill_indexer": {
            "DenseCandidateIndexer": Dense
        },
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    assert make_candidate_indexer(0, 8) is None
    indexer = make_candidate_indexer(2048, 8)
    if sm == 90:
        assert isinstance(indexer, TritonCandidateIndexer)
        assert indexer.decode_indexer is None  # Hopper decode remains inline.
    elif oracle:
        assert isinstance(indexer, TritonCandidateIndexer)
        assert isinstance(indexer.decode_indexer, DeepGemm)
    else:
        assert isinstance(indexer, DeepGemm)
        assert isinstance(indexer.prefill_dense, Dense) == (cp > 1)


@pytest.mark.parametrize("length", [15, 16, 17])
def test_candidate_path_for_short_and_pruned_contexts(length, monkeypatch):
    inputs = _case(torch.float32, rows=(4,), lengths=(length,))
    impl = TritonCandidateIndexer(2, 8)
    published = impl.publish_prefill(inputs, _out(inputs))
    calls = []

    def sparse(q, keys, weights, blocks, lens):
        calls.append(keys.shape)
        return _reference_candidate_scores(q, keys, weights, blocks, lens)

    def forbidden(*args):
        raise AssertionError("Consumer must score its candidate blocks")

    monkeypatch.setattr(impl, "_dense_scores", forbidden)
    monkeypatch.setattr(impl, "_candidate_scores", sparse)
    out = _out(inputs)
    impl.select_prefill(published, inputs, out)
    assert calls and all(shape == (length, 16) for shape in calls)
    external = PrefillCandidateBlocks(
        request_blocks=[published.request_blocks[0][:, :1]]
    )
    calls.clear()
    impl.select_prefill(external, inputs, out)
    assert calls


def test_short_context_preserves_source_block_visibility():
    source = _case(torch.float32, rows=(3,), lengths=(15,))
    impl = TritonCandidateIndexer(2, 8)
    published = impl.publish_prefill(source, _out(source))
    # A different consumer visibility must not invent unpublished source blocks.
    consumer = dataclasses.replace(
        source, compress_lens=torch.full_like(source.compress_lens, 15)
    )
    actual = _out(consumer)
    impl.select_prefill(published, consumer, actual)
    ref = _dense(consumer, 0, 0, 3)
    for r, blocks in enumerate(published.request_blocks[0]):
        keep = torch.zeros(15, dtype=torch.bool, device=DEVICE)
        for block in blocks[blocks >= 0].tolist():
            keep[block * 8 : (block + 1) * 8] = True
        ref[r].masked_fill_(~keep, -torch.inf)
    _check_selection(actual, ref, consumer.request_starts, 7, tolerance=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton CUDA required")
@pytest.mark.parametrize("budget", [1, 1 << 20])
@pytest.mark.parametrize("lengths", [(15, 13), (65, 83)])
def test_triton_prefill_integration_no_gather(budget, lengths, monkeypatch):
    import sglang.kernels.ops.attention.dsv4.candidate_bf16_mqa as kernel

    inputs = _case(torch.bfloat16, rows=(5, 4), lengths=lengths, heads=32, dim=128)
    oracle = TritonCandidateIndexer(2, 8, budget_bytes=budget)
    published = oracle.publish_prefill(inputs, _out(inputs))
    expected = _out(inputs)
    oracle._candidate_scores = _reference_candidate_scores
    oracle.select_prefill(published, inputs, expected)
    impl = TritonCandidateIndexer(2, 8, budget_bytes=budget)
    calls = []
    original = kernel.candidate_bf16_mqa_logits

    def record(*args):
        calls.append(args[0].shape[0])
        return original(*args)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Triton sparse scoring must not gather K or call Torch scores"
        )

    monkeypatch.setattr(kernel, "candidate_bf16_mqa_logits", record)
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "index_select", forbidden)
        patch.setattr(impl, "_dense_scores", forbidden)
        actual = _out(inputs)
        impl.select_prefill(published, inputs, actual)
    assert sum(calls) == inputs.num_rows
    # Compare with independently masked dense scores, respecting cutoff ties.
    start = 0
    for b, rows in enumerate(inputs.rows_per_request):
        dense = _dense(inputs, b, start, rows)
        keep = torch.zeros_like(dense, dtype=torch.bool)
        for r, blocks in enumerate(published.request_blocks[b]):
            for block in blocks[blocks >= 0].tolist():
                keep[r, block * 8 : (block + 1) * 8] = True
        dense.masked_fill_(~keep, -torch.inf)
        _check_selection(
            actual[start : start + rows],
            dense,
            inputs.request_starts[start : start + rows],
            7,
            tolerance=0.008,
        )
        start += rows
    assert torch.equal(actual.sort().values, expected.sort().values)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_unsupported_sparse_rejected():
    inputs = _case(torch.float32, rows=(4,), lengths=(65,), heads=32, dim=128)
    impl = TritonCandidateIndexer(2, 8)
    published = impl.publish_prefill(inputs, _out(inputs))
    with pytest.raises(ValueError, match="BF16"):
        impl.select_prefill(published, inputs, _out(inputs))


@pytest.mark.parametrize("length", [0, 7, 35])
@pytest.mark.parametrize("rows", [0, 4])
def test_decode_publish_select_mapping(length, rows, monkeypatch):
    from sglang.srt.layers.attention.dsv4.candidate_indexer import TritonDecodeInputs

    impl = TritonCandidateIndexer(2, 8)
    # Repeated slot rows model verify tokens of one request with different lenses.
    slots = (torch.arange(length, device=DEVICE).flip(0) + 100).expand(rows, -1)
    lens = torch.tensor([0, min(3, length), max(0, length - 1), length], device=DEVICE)[
        :rows
    ]
    inputs = TritonDecodeInputs(
        torch.zeros(rows, 4, 16, device=DEVICE),
        torch.ones(rows, 4, device=DEVICE),
        slots,
        lens,
        torch.empty(0, device=DEVICE),
        64,
        7,
    )
    dense = torch.arange(length, device=DEVICE).float().expand(rows, -1).clone()
    dense.masked_fill_(
        torch.arange(length, device=DEVICE)[None] >= lens[:, None], -torch.inf
    )
    seen = []

    def scorer(inputs, blocks=None, block_size=8):
        seen.append(blocks)
        if blocks is None:
            return dense
        positions = (
            blocks.long()[:, :, None] * block_size
            + torch.arange(block_size, device=DEVICE)
        ).flatten(1)
        if not length:
            return torch.full(positions.shape, -torch.inf, device=DEVICE)
        result = dense.gather(1, positions.clamp(0, length - 1))
        return result.masked_fill(
            (positions < 0) | (positions >= lens[:, None]), -torch.inf
        )

    monkeypatch.setattr(impl, "_decode_scores", scorer)
    page = torch.empty(rows, 7, dtype=torch.int32, device=DEVICE)
    raw = torch.empty_like(page)
    published = impl.publish_decode(inputs, page, raw)
    assert published.blocks.shape == (rows, 2)
    _check_selection(
        raw, dense, torch.zeros(rows, device=DEVICE, dtype=torch.int64), 7, tolerance=0
    )
    # Consumer's own scores change: reference masks dense scores independently.
    dense = torch.where(dense > -torch.inf, -dense, dense)
    impl.select_decode(published, inputs, page, raw)
    assert seen[-1] is published.blocks
    reference = dense.clone()
    for r, blocks in enumerate(published.blocks):
        keep = torch.zeros(length, dtype=torch.bool, device=DEVICE)
        for block in blocks[blocks >= 0].tolist():
            keep[block * 8 : (block + 1) * 8] = True
        reference[r].masked_fill_(~keep, -torch.inf)
    _check_selection(
        raw,
        reference,
        torch.zeros(rows, device=DEVICE, dtype=torch.int64),
        7,
        tolerance=0,
    )
    for r in range(rows):
        valid = raw[r] >= 0
        logical = raw[r, valid].long()
        assert torch.equal(logical, logical.sort().values)
        assert torch.equal(page[r, valid].long(), slots[r, logical])
        assert (page[r, ~valid] == -1).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
@pytest.mark.parametrize("replay", [False, True])
def test_paged_fp4_candidate_scores(replay):
    from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
        fp4_index_logits_decode,
        store_fp4_index_k_cache,
    )

    torch.manual_seed(84)
    rows, heads, length, page_size = 4, 32, 137, 64
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(192, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.zeros(3, page_size * 68, device="cuda", dtype=torch.uint8)
    store_fp4_index_k_cache(
        keys, cache, torch.arange(192, device="cuda"), page_size=page_size
    )
    slots = torch.stack(
        [torch.randperm(192, device="cuda")[:length] for _ in range(rows)]
    )
    lens = torch.tensor([0, 17, 130, 137], device="cuda")
    blocks = torch.tensor(
        [[0, -1, 16], [2, 0, -1], [16, 8, 0], [16, 1, 8]],
        device="cuda",
        dtype=torch.int32,
    )

    def sparse():
        return fp4_index_logits_decode(
            q, weights, slots, lens, cache, page_size, candidate_blocks=blocks
        )

    actual = sparse()
    if replay:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                sparse()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = sparse()
        # Captured code must read new device values, not capture-time lengths/IDs.
        blocks.copy_(blocks.flip(1))
        lens.copy_(torch.tensor([8, 9, 65, 136], device="cuda"))
        graph.replay()
    dense = fp4_index_logits_decode(q, weights, slots, lens, cache, page_size)
    positions = (
        blocks.long()[:, :, None] * 8 + torch.arange(8, device="cuda")
    ).flatten(1)
    expected = dense.gather(1, positions.clamp(0, length - 1))
    expected.masked_fill_(
        (positions < 0) | (positions >= lens[:, None]) | (positions >= length),
        -torch.inf,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
