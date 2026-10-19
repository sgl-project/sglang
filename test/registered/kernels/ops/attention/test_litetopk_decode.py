"""LiteTopK decode selection against an exact (BF16 score, lower slot) reference, with
histograms from a torch model of DeepGEMM's coarse bins; with a DeepGEMM that counts
them, the DSA producer chain and the DeepSeek-V4.1 serving routes too."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.litetopk_decode import (
    BF16_TOP512,
    BF16_TOP2048,
    HISTOGRAM_BINS,
    LiteTopKStorage,
    unsupported_reason,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="LiteTopK targets SM100",
)

CONFIGS = [BF16_TOP2048, BF16_TOP512]
_STATE_BYTES = 16


def _ordered_keys(x: torch.Tensor) -> torch.Tensor:
    """int64 keys ordering the float values of ``x``, -0 below +0."""
    bits = x.float().view(torch.int32).long() & 0xFFFFFFFF
    return torch.where(bits >= 1 << 31, bits ^ 0xFFFFFFFF, bits | 1 << 31)


def _coarse_bins(x: torch.Tensor) -> torch.Tensor:
    """DeepGEMM's coarse histogram bin of each live, non-NaN score (bin 0 holds the
    largest): the FP16-RN magnitude code |h| >> 6 below 16, unit bins above."""
    x = x.float()
    bits = x.view(torch.int32)
    magnitude = bits & 0x7FFFFFFF
    negative = (bits < 0) & (magnitude != 0)
    code = (x.half().view(torch.int16).int() & 0x7FFF) >> 6
    bounded = torch.clamp(magnitude - negative.int(), max=0x435F0000).view(
        torch.float32
    )
    unit = torch.clamp(bounded.floor().int() + 288, min=304)
    code = torch.where(code >= 304, unit, code)
    return torch.where(negative, 512 + code, 511 - code)


def _histogram(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    hist = torch.zeros(
        (scores.shape[0], HISTOGRAM_BINS), dtype=torch.int32, device=scores.device
    )
    for row, length in enumerate(lengths.tolist()):
        x = scores[row, : max(length, 0)].float()
        x = x[~x.isnan()]
        hist[row] = torch.bincount(_coarse_bins(x), minlength=HISTOGRAM_BINS).int()
    return hist


def _slots(table, row, length, page):
    columns = torch.arange(length, device=table.device)
    return table[row, columns // page].long() * page + columns % page


def _check(config, scores, lengths, table, rows_per_table_row, out):
    page = config.page_size
    for row, length in enumerate(lengths.tolist()):
        length = min(max(length, 0), scores.shape[1])
        slots = _slots(table, row // rows_per_table_row, length, page)
        keep = min(length, config.topk)
        # Larger score first; equal scores prefer the lower physical slot.
        by_slot = torch.argsort(slots, stable=True)
        keys = _ordered_keys(scores[row, :length])[by_slot]
        order = torch.argsort(keys, descending=True, stable=True)[:keep]
        want = slots[by_slot][order].sort().values
        got = out[row, :keep].long().sort().values
        assert torch.equal(got, want), f"row {row} of length {length}"
        assert bool((out[row, keep:] == -1).all()), f"row {row} padding"


def _selected_values(config, scores, lengths, table, out):
    """Per row, the sorted scores of the selected slots: equal between two exact
    top-k implementations whatever they pick among scores tied at the threshold."""
    values = []
    for row, length in enumerate(lengths.tolist()):
        slots = _slots(table, row, length, config.page_size)
        column = {s: c for c, s in enumerate(slots.tolist())}
        picked = [column[s] for s in out[row].tolist() if s >= 0]
        values.append(scores[row, picked].float().sort().values)
    return values


def _at_rest(storage):
    assert not bool(storage.histogram.any()), "histogram not returned to zero"
    states = storage.max_rows * _STATE_BYTES
    state = storage.workspace[:states].view(torch.int32).view(-1, 4)
    assert not bool(state[:, :2].any()), "row hand-off words not reset"
    assert torch.equal(state[:, 2], state[:, 3]), "part signals unbalanced"
    assert not bool(storage.workspace[states:].any()), "candidates not cleared"


def _scores(rows, width, kind, gen):
    x = torch.randn((rows, width), device="cuda", generator=gen)
    if kind == "ties":  # many equal scores, signed zeros
        x = torch.randint(-6, 7, (rows, width), device="cuda", generator=gen) / 4.0
        x[:, ::7] = -0.0
    elif kind == "wide":  # unit bins above 16, infinities
        x = x * 3000
        x[:, 11::997] = float("inf")
        x[:, 13::1009] = -float("inf")
    elif kind == "glm":
        # GLM index scores: K-th largest 118..165 (a unit bin holds one BF16 value, so
        # the crossing bin is a tie class of hundreds), or -33..-72 on deeper layers.
        x = 150 + 12 * x
        x[1::2] = 90 - x[1::2]
    return x.bfloat16()


def _table(rows, width, page, gen):
    pages = -(-width // page)
    return (
        torch.randperm(rows * pages + 3, device="cuda", generator=gen)[: rows * pages]
        .int()
        .view(rows, pages)
    )


def _run(config, storage, scores, lengths, table, rows_per_table_row=1, hist=None):
    plan = storage.plan(scores.shape[0])
    plan.histogram.copy_(_histogram(scores, lengths) if hist is None else hist)
    out = torch.full(
        (scores.shape[0], config.topk), -7, dtype=torch.int32, device="cuda"
    )
    plan.select(scores, lengths, table, rows_per_table_row=rows_per_table_row, out=out)
    _check(config, scores, lengths, table, rows_per_table_row, out)
    _at_rest(storage)
    return out


_LONG_ROWS = [1, 511, 512, 513, 2047, 2048, 2049, 10240, 10241, 16384, 16385, 70001]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
@pytest.mark.parametrize("kind", ["normal", "ties", "wide", "glm"])
def test_single_rows(config, kind):
    gen = torch.Generator(device="cuda").manual_seed(1)
    storage = LiteTopKStorage(config, 4, torch.device("cuda"))
    width = 1048576
    for length in _LONG_ROWS + [262147, width]:
        scores = _scores(1, width, kind, gen)
        lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
        table = _table(1, width, config.page_size, gen)
        _run(config, storage, scores, lengths, table)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
def test_launch_shapes(config):
    """One storage, its SM count changed between calls: every B200 shape at 148 SMs,
    up to and past one row per SM at 7 and 63, several waves at 296; mixed lengths."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    storage = LiteTopKStorage(config, 512, torch.device("cuda"))
    width = 49152
    launches = [(148, 5), (148, 20), (148, 70), (148, 160), (148, 300), (148, 500)]
    launches += [(7, 9), (7, 12), (7, 20), (7, 40), (63, 63), (63, 70)]
    for num_sms, rows in launches + [(296, 20), (296, 500)]:
        storage.num_sms = num_sms
        scores = _scores(rows, width, ("normal", "ties", "glm")[rows % 3], gen)
        lengths = torch.randint(0, width + 1, (rows,), device="cuda", generator=gen)
        lengths[: len(_LONG_ROWS)] = torch.tensor(_LONG_ROWS[:rows], device="cuda")
        table = _table(rows, width, config.page_size, gen)
        _run(config, storage, scores, lengths.int(), table)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
def test_shared_table_rows_and_long_rows(config):
    """Rows that share page-table rows (``rows_per_table_row``); rows near 1M keys,
    whose parts span many tiles (16 rows: the 1024-thread kernel's long parts)."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    next_n, width = 4, 1048576
    for requests in (2, 4):
        rows = requests * next_n
        storage = LiteTopKStorage(config, rows, torch.device("cuda"))
        scores = _scores(rows, width, "glm", gen)
        lengths = torch.tensor(
            [width - next_n + 1 + j for _ in range(requests) for j in range(next_n)],
            dtype=torch.int32,
            device="cuda",
        )
        table = _table(requests, width, config.page_size, gen)
        _run(config, storage, scores, lengths, table, rows_per_table_row=next_n)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
def test_fallbacks_stay_exact(config):
    """A histogram that disagrees with the scores (none, another row's, off by one),
    or a crossing bin beyond the candidate capacity or the short-row lists (a tie
    class of thousands), takes the exact whole-row select."""
    gen = torch.Generator(device="cuda").manual_seed(3)
    rows, width = 24, 40000
    lengths = torch.randint(1, width + 1, (rows,), device="cuda", generator=gen).int()
    lengths[:6] = torch.tensor([600, 3000, 9000, 16000, 20000, width], device="cuda")
    table = _table(rows, width, config.page_size, gen)
    storage = LiteTopKStorage(config, rows, torch.device("cuda"))
    scores = _scores(rows, width, "glm", gen)
    for hist in (
        torch.zeros((rows, HISTOGRAM_BINS), dtype=torch.int32, device="cuda"),
        _histogram(_scores(rows, width, "glm", gen), lengths),
        _histogram(scores, lengths)
        + (torch.arange(HISTOGRAM_BINS, device="cuda") == 5),
    ):
        _run(config, storage, scores, lengths, table, hist=hist.int())
    small = LiteTopKStorage(config, rows, torch.device("cuda"), candidate_capacity=4)
    _run(config, small, _scores(rows, width, "ties", gen), lengths, table)
    _run(config, storage, _scores(rows, width, "ties", gen), lengths, table)


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.name)
def test_graph_replay(config):
    """No reset runs between replays: each replay leaves the buffers at rest."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    rows, width = 18, 65536
    storage = LiteTopKStorage(config, 32, torch.device("cuda"))
    scores = _scores(rows, width, "normal", gen)
    lengths = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    table = _table(rows, width, config.page_size, gen)
    out = _run(config, storage, scores, lengths, table)  # warm up the JIT module
    plan = storage.plan(rows)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.select(scores, lengths, table, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    for kind, new_lengths in (
        ("glm", [width] * rows),
        ("ties", [0, 1, 512, 2048, 2049, 9000, 16384, 16385] + [width - 3] * 10),
        ("wide", [width // (r + 1) for r in range(rows)]),
    ):
        scores.copy_(_scores(rows, width, kind, gen))
        lengths.copy_(torch.tensor(new_lengths, dtype=torch.int32))
        plan.histogram.copy_(_histogram(scores, lengths))
        graph.replay()
        torch.cuda.synchronize()
        _check(config, scores, lengths, table, 1, out)
        _at_rest(storage)


# ---- producer chains (need a DeepGEMM whose paged MQA logits count the histogram)


def _require():
    reason = unsupported_reason()
    if reason is not None:
        pytest.skip(reason)
    import deep_gemm

    return deep_gemm


def _fp8(shape, gen):
    """FP8 E4M3 bytes without NaN encodings."""
    magnitude = torch.randint(
        0, 127, shape, dtype=torch.uint8, device="cuda", generator=gen
    )
    sign = torch.randint(0, 2, shape, dtype=torch.uint8, device="cuda", generator=gen)
    return magnitude | (sign << 7)


@pytest.mark.parametrize("next_n", [1, 4])
@pytest.mark.parametrize("batch", [1, 40])
def test_dsa_producer_chain(next_n, batch):
    """The DSA indexer's MXFP8 paged MQA logits (sglang's DeepGEMM adapter) count the
    histogram LiteTopK selects from, and are the logits the default top-k widens."""
    dg = _require()
    from sglang.srt.layers.attention import deep_gemm_mqa_logits as mqa

    gen = torch.Generator(device="cuda").manual_seed(batch * 10 + next_n)
    rows, page, max_len = batch * next_n, 64, 65536
    pages = batch * max_len // page + 5
    q = torch.empty((rows, 1, 32, 128), dtype=torch.float8_e4m3fn, device="cuda")
    q.view(torch.uint8).copy_(_fp8(q.shape, gen))
    weights = torch.randn((rows, 32), device="cuda", generator=gen) * 0.05
    cache = torch.empty((pages, page * 132), dtype=torch.uint8, device="cuda")
    cache[:, : page * 128] = _fp8((pages, page * 128), gen)
    # One power-of-two scale per token, packed UE8M0 for each 32-channel block.
    exponent = torch.randint(115, 122, (pages, page), device="cuda", generator=gen)
    cache[:, page * 128 :] = (exponent.int() * 0x01010101).view(torch.uint8)
    cache = cache.view(pages, page, 1, 132)
    # Verify rows: token j of a request sees j more keys than its first token.
    table = _table(batch, max_len, page, gen).repeat_interleave(next_n, 0)
    lengths = torch.zeros((rows, 1), dtype=torch.int32, device="cuda")
    storage = LiteTopKStorage(BF16_TOP2048, rows, torch.device("cuda"))
    plan = storage.plan(rows)
    out = torch.empty((rows, 2048), dtype=torch.int32, device="cuda")
    # The second call's producer adds into the histogram the first selection cleared.
    for base in (
        [max_len - next_n - r * 997 for r in range(batch)],
        [0, 1, 2047, 2048, 4097][:batch] + [max_len // 3] * max(batch - 5, 0),
    ):
        lengths.copy_(
            torch.tensor(base, device="cuda").repeat_interleave(next_n)[:, None]
        )
        lengths += torch.arange(rows, device="cuda").view(-1, 1) % next_n
        schedule = mqa.get_paged_mqa_logits_metadata(lengths, page, dg.get_num_sms())
        call = (q, cache, weights, lengths, table, schedule, max_len)
        logits = mqa.fp8_paged_mqa_logits(*call, histogram=plan.histogram)
        lens = lengths.view(-1)
        assert logits.dtype == torch.bfloat16
        assert torch.equal(plan.histogram, _histogram(logits, lens))
        live = torch.arange(max_len, device="cuda") < lengths
        widened = mqa.fp8_paged_mqa_logits(*call)
        assert torch.equal(logits.float()[live], widened[live])
        plan.select(logits, lens, table, out=out)
        _check(BF16_TOP2048, logits, lens, table, 1, out)
        _at_rest(storage)


# ---- DeepSeek-V4.1 serving routes (FullTopKIndexer, SparseTableBackend)


def _mxfp4_inputs(requests, next_n, length, gen):
    rows, pages = requests * next_n, -(-length // 128)
    q = torch.randint(0, 256, (rows, 1, 32, 64), device="cuda", generator=gen)
    sf = torch.full((rows, 1, 32), 0x7D7D7D7D, device="cuda", dtype=torch.int32)
    cache = torch.randint(
        0, 256, (requests * pages, 128 * 68), device="cuda", generator=gen
    ).to(torch.uint8)
    cache[:, 128 * 64 :] = 125
    return SimpleNamespace(
        q=(q.to(torch.uint8).view(torch.int8), sf),
        cache=cache.view(requests * pages, 128, 1, 68),
        # BF16 values, as the fused Q packer leaves them in FP32.
        weights=torch.randn((rows, 32), device="cuda", generator=gen).bfloat16(),
        lens=torch.zeros((rows,), dtype=torch.int32, device="cuda"),
        table=_table(requests, length, 128, gen).repeat_interleave(next_n, 0),
    )


@pytest.mark.parametrize("next_n, publish", [(1, True), (4, False), (6, True)])
def test_dsv41_routes_dynamic_graph(monkeypatch, next_n, publish):
    """The hook's per-forward schedule (decode: the forward's own) and the FullTopK /
    SparseTable decode routes, captured once and replayed with new lengths and
    pages; selections hold the default route's values."""
    _require()
    from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
        amax_topk_blocks,
        candidate_row_lens,
    )
    from sglang.kernels.ops.attention.dsv4.candidate_table import (
        sort_candidate_blocks,
    )
    from sglang.srt.layers.attention import litetopk_decode
    from sglang.srt.layers.attention.dsv4.metadata import PagedIndexerMetadata
    from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk, sparse_table
    from sglang.srt.layers.attention.dsv4.v41_indexer.litetopk import Dsv41LiteTopK
    from sglang.srt.layers.attention.dsv4.v41_indexer.scoring import (
        DeepGEMMDecodeData,
    )

    gen = torch.Generator(device="cuda").manual_seed(631 + next_n)
    data = _mxfp4_inputs(2, next_n, 16384, gen)
    rows = data.lens.numel()
    req_rows = torch.arange(rows, device="cuda") // next_n
    out = torch.empty((rows, 512), device="cuda", dtype=torch.int32)
    # Only the query projection / packing is substituted; the routes run as served.
    packed = DeepGEMMDecodeData(*data.q, data.weights.float(), data.cache)
    for module in (full_topk, sparse_table):
        monkeypatch.setattr(module, "get_deep_gemm_decode_data", lambda *_: packed)
    buffers = litetopk_decode.LiteTopKDecode(
        config=BF16_TOP512, device=torch.device("cuda")
    )
    hook = Dsv41LiteTopK(buffers)
    kw = dict(token_to_kv_pool=None, req_to_token=None)
    full = full_topk.FullTopKIndexer(
        **kw, use_deep_gemm_prefill=False, use_deep_gemm_decode=True, litetopk=hook
    )
    default = full_topk.FullTopKIndexer(
        **kw, use_deep_gemm_prefill=False, use_deep_gemm_decode=True
    )
    sparse = sparse_table.SparseTableBackend(
        **kw,
        page_size=256,
        candidate_topk_blocks=512,
        candidate_block_size=8,
        litetopk=hook,
    )

    def forward(hooked=True):
        # Built in the captured region, as decode / verify graphs build it.
        metadata = PagedIndexerMetadata(
            page_size=256,
            compressed_page_size=128,
            page_table=data.table,
            compressed_seq_lens=data.lens,
            use_topk_v2=True,
            force_deep_gemm_metadata=True,
            compress_ratio=2,
        )
        inputs = SimpleNamespace(
            paged_metadata=metadata,
            is_verify=next_n > 1,
            req_rows=req_rows,
            out_page_indices=out,
        )
        if not hooked:
            default.topk_decode(inputs)
            return None, metadata
        hook.prepare_metadata(metadata, next_n)
        if publish:
            table = sparse.publish_decode(inputs)
            torch.cuda.current_stream().wait_event(table.ready)
            return table, metadata
        full.topk_decode(inputs)
        return None, metadata

    def causal(length):  # row j of a request sees length - next_n + 1 + j keys
        j = torch.arange(rows, device="cuda", dtype=torch.int32) % next_n
        return length - next_n + 1 + j

    data.lens.copy_(causal(16384))
    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        table, metadata = forward()
    if next_n == 1:
        assert metadata.litetopk_schedule is metadata.deep_gemm_metadata
    for length in (16384, 513, 8193):
        data.lens.copy_(causal(length))
        # Each request's rows still share one page-table row.
        data.table.copy_(data.table.flip(1))
        graph.replay()
        torch.cuda.synchronize()
        got = out.clone()
        # The default route's logits (DeepGEMM through sglang's adapter, widened).
        logits = full_topk.deep_gemm_fp4_paged_mqa_logits(
            data.q,
            data.cache,
            data.weights,
            data.lens,
            data.table,
            metadata.deep_gemm_metadata,
            metadata.max_compressed_seq_len,
        )
        _check(BF16_TOP512, logits, data.lens, data.table, 1, got)
        _at_rest(buffers._storage)
        forward(hooked=False)
        assert all(
            torch.equal(a, b)
            for a, b in zip(
                _selected_values(BF16_TOP512, logits, data.lens, data.table, got),
                _selected_values(BF16_TOP512, logits, data.lens, data.table, out),
            )
        )
        if publish:
            counts, valid = candidate_row_lens(data.lens, 512)
            expected = amax_topk_blocks(logits, data.lens, counts, 512)
            phys = sort_candidate_blocks(expected, data.lens, data.table, 128)
            assert torch.equal(table.blocks, expected)
            assert torch.equal(table.phys_blocks, phys)
            assert torch.equal(table.valid_lens, valid)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
