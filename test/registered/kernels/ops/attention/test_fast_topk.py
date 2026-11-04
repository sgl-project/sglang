import pytest
import torch
from sglang.kernels.ops.attention.fast_topk import fast_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _check_topk_values(score, lengths, indices, topk, row_starts):
    """fast_topk leaves order and tie-breaking unspecified,
    so compare the sorted top-k values rather than index sets."""
    for b in range(score.shape[0]):
        start = int(row_starts[b]) if row_starts is not None else 0
        length = int(lengths[b])
        section = score[b, start : start + length]
        row = indices[b]
        if length <= topk:
            # naive path: identity indices, then -1 fill
            assert torch.equal(
                row[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (row[length:] == -1).all()
            continue
        assert (row >= 0).all(), "long rows must fill every slot"
        picked = section[row.long()]
        expected = torch.topk(section, topk).values
        assert torch.equal(
            picked.sort(descending=True).values, expected.sort(descending=True).values
        ), f"row {b}: top-{topk} value multiset mismatch"


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "batch,length",
    [
        (1, 4096),
        (7, 3000),
        (33, 32768),
        (128, 2050),
    ],
)
def test_fast_topk_long_rows(topk, batch, length):
    torch.manual_seed(0)
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_short_and_mixed_rows(topk):
    torch.manual_seed(0)
    max_len = topk + 128
    batch = 8
    score = torch.randn(batch, max_len, dtype=torch.float32, device="cuda")
    # rows shorter than k (naive path), exactly k, and longer than k
    lens = [1, topk // 3, topk - 1, topk, topk + 1, topk + 7, 17, max_len]
    lengths = torch.tensor(lens[:batch], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_ragged_with_row_starts(topk):
    torch.manual_seed(0)
    batch, width = 16, 8192
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    row_starts = torch.randint(0, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.randint(1, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.minimum(lengths, width - row_starts).to(torch.int32)
    # ensure some rows are longer than k
    lengths[0] = min(width - int(row_starts[0]), topk + 100)

    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_row_stride(topk):
    torch.manual_seed(0)
    batch, length = 8, 4096
    base = torch.randn(batch, 2 * length, dtype=torch.float32, device="cuda")
    score = base[:, :length]  # stride(0) == 2*length, stride(1) == 1
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "fill",
    [
        "binary",  # only 0s and 1s: extreme duplication at the threshold bin
        "few_levels",  # a handful of distinct levels incl. negatives
        "constant",  # whole rows of one value
    ],
)
def test_fast_topk_duplicate_heavy(topk, fill):
    torch.manual_seed(0)
    batch, length = 16, 8192
    if fill == "binary":
        score = torch.randint(0, 2, (batch, length), dtype=torch.float32, device="cuda")
    elif fill == "few_levels":
        levels = torch.tensor([-5.0, -1.0, 0.0, 0.5, 2.0], device="cuda")
        score = levels[torch.randint(0, 5, (batch, length), device="cuda")]
    else:
        score = torch.full((batch, length), 3.25, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_negative_and_zero(topk):
    torch.manual_seed(0)
    batch, length = 8, 16384
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda") * 100
    score[:, : length // 3] = 0.0  # long zero prefix
    score[:, length // 3 : length // 2] = -1e30  # very negative block
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


def test_fast_topk_unsupported_k():
    score = torch.randn(2, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="topk"):
        fast_topk(score, lengths, 1024)


def _deterministic_selector(name, score, lengths, topk=512, row_starts=None):
    if name == "native":
        return fast_topk(score, lengths, topk, row_starts=row_starts)
    if name == "split":
        from sglang.srt.layers.attention.qsa.split_topk import split_fast_topk

        return split_fast_topk(score, lengths, topk, row_starts=row_starts)
    from sglang.kernels.jit.utils import (
        KERNEL_PATH,
        is_arch_support_pdl,
        load_jit,
        make_cpp_args,
    )

    items = max(1, (score.shape[1] + 4095) // 4096)
    args = make_cpp_args(items, 8, is_arch_support_pdl(), True, 32, True)
    flags = ["-DQ38_PERF_SHORT_SCAN=1"] if name == "mk_scan" else []
    module = load_jit(
        "test_deterministic_mk_topk",
        name,
        *args,
        cuda_files=[str(KERNEL_PATH / "csrc/attention/q38_mk_topk.cuh")],
        cuda_wrappers=[("select_rows", f"q38_mk_topk<{args}>")],
        extra_cuda_cflags=flags,
    )
    out = torch.empty((score.shape[0], topk), device=score.device, dtype=torch.int32)
    module.select_rows(score, lengths, row_starts, out)
    return out


def _stable_id_reference(score, lengths, topk, row_starts=None, selector="native", split_config=None, mk_items=None):
    if selector == "split" and split_config is None:
        from sglang.srt.layers.attention.qsa.split_topk import _config

        split_config = _config(score.shape[0], score.shape[1])
    result = torch.full((score.shape[0], topk), -1, dtype=torch.int32)
    for row in range(score.shape[0]):
        start = int(row_starts[row]) if row_starts is not None else 0
        length = int(lengths[row])
        if length <= topk:
            result[row, :length] = torch.arange(length, dtype=torch.int32)
            continue
        selected = torch.argsort(
            score[row, start : start + length], descending=True, stable=True
        )[:topk]
        if selector == "split":
            threads, cluster, iters = split_config
            span = ((length + cluster - 1) // cluster + 3) & ~3
            rank, local = selected // span, selected % span
            thread = (local // 4) % threads
            order = (rank * threads + thread) * (4 * iters) + (local // (4 * threads)) * 4 + local % 4
        elif length <= 4096:
            threads = 1024 if selector == "native" else 512
            order = selected if length == topk + 1 else (selected % threads) * (4096 // threads) + selected // threads
        elif selector == "native":
            threads = 1024 if selector == "native" else 512
            tile = selected // 131072
            local = selected % 131072
            tile_length = torch.clamp(length - tile * 131072, max=131072)
            rounds = ((tile_length + 31) // 32 + threads - 1) // threads
            word = local // 32
            warp = word // (32 * rounds)
            lane, word_round = word % 32, (word // 32) % rounds
            order = tile * 131072 + (((warp * 32 + lane) * rounds + word_round) * 32 + local % 32)
        else:
            items = mk_items if mk_items is not None else max(1, (score.shape[1] + 4095) // 4096)
            rank, local = selected // (items * 512), selected % (items * 512)
            thread = local % 512
            order = (rank * 512 + thread) * items + local // 512
        result[row] = selected[torch.argsort(order)].to(torch.int32)
    return result


@pytest.mark.parametrize("selector", ["native", "split", "mk", "mk_scan"])
@pytest.mark.parametrize("length", [513, 1024, 4096, 4097, 65536])
@pytest.mark.parametrize("kind", ["cutoff_ties", "overflow_distinct", "constant"])
def test_fast_topk_deterministic_members_and_order(selector, length, kind):
    torch.manual_seed(7838)
    scores = torch.full((4, 65536), float("nan"), device="cuda", dtype=torch.float32)
    if kind == "cutoff_ties":
        for row in range(4):
            scores[row, :length] = 0
            scores[row, torch.randperm(length, device="cuda")[:500]] = 1
    elif kind == "overflow_distinct":
        scores[:, :length] = (
            1 + torch.arange(length, device="cuda", dtype=torch.float32) * 2**-22
        )
    else:
        scores[:, :length] = 0
    lengths = torch.full((4,), length, device="cuda", dtype=torch.int32)
    expected = _stable_id_reference(scores.cpu(), lengths.cpu(), 512, selector=selector)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _deterministic_selector(selector, scores, lengths)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = _deterministic_selector(selector, scores, lengths)
    with torch.cuda.stream(stream):
        copies = []
        for _ in range(32):
            graph.replay()
            copies.append(output.clone())
    stream.synchronize()
    assert torch.equal(
        torch.stack(copies).cpu(), expected.unsqueeze(0).expand(32, -1, -1)
    )
    for _ in range(32):
        assert torch.equal(
            _deterministic_selector(selector, scores, lengths).cpu(), expected
        )
    scores[:, :length].neg_()
    expected_changed = _stable_id_reference(scores.cpu(), lengths.cpu(), 512, selector=selector)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    assert torch.equal(output.cpu(), expected_changed)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_deterministic_offsets_and_signed_zero(topk):
    score = torch.zeros((3, 16384), dtype=torch.float32, device="cuda")[:, :8192]
    score[:, 1::2] = -0.0
    starts = torch.tensor([0, 3, 127], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [topk - 1, topk + 1, 8192 - 127], dtype=torch.int32, device="cuda"
    )
    expected = _stable_id_reference(score.cpu(), lengths.cpu(), topk, starts.cpu())
    for _ in range(32):
        assert torch.equal(fast_topk(score, lengths, topk, starts).cpu(), expected)


@pytest.mark.parametrize("selector", ["mk", "mk_scan"])
def test_mk_topk_deterministic_offsets_and_signed_zero(selector):
    score = torch.zeros((3, 16384), dtype=torch.float32, device="cuda")[:, :8192]
    score[:, 1::2] = -0.0
    starts = torch.tensor([0, 3, 127], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([511, 513, 8065], dtype=torch.int32, device="cuda")
    expected = _stable_id_reference(
        score.cpu(), lengths.cpu(), 512, starts.cpu(), selector=selector
    )
    for _ in range(32):
        actual = _deterministic_selector(selector, score, lengths, row_starts=starts)
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("selector", ["native", "split", "mk", "mk_scan"])
def test_fast_topk_deterministic_special_values(selector):
    lengths = torch.tensor([513, 4096, 4097, 65536], dtype=torch.int32, device="cuda")
    scores = torch.full((4, 65536), float("nan"), dtype=torch.float32, device="cuda")
    scores[0, :513] = -float("inf")
    scores[1, :4096] = float("inf")
    scores[2, :4097] = 0.0
    scores[2, 1:4097:2] = -0.0
    scores[3] = -float("inf")
    scores[3, ::127] = 1.0
    expected = _stable_id_reference(scores.cpu(), lengths.cpu(), 512, selector=selector)
    for _ in range(32):
        actual = _deterministic_selector(selector, scores, lengths)
        assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("kind", ["unique", "cutoff_ties"])
def test_fast_topk_deterministic_tile_boundary(topk, kind):
    length = 131073
    score = torch.full((1, length + 31), float("nan"), device="cuda")
    if kind == "unique":
        score[0, :length] = torch.arange(length, dtype=torch.float32, device="cuda")
    else:
        score[0, :length] = 0
        winners = torch.linspace(64, length - 1, topk - 12, device="cuda").long()
        score[0, winners] = 1
    lengths = torch.tensor([length], dtype=torch.int32, device="cuda")
    expected = _stable_id_reference(score.cpu(), lengths.cpu(), topk)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fast_topk(score, lengths, topk)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fast_topk(score, lengths, topk)
    with torch.cuda.stream(stream):
        copies = []
        for _ in range(32):
            graph.replay()
            copies.append(output.clone())
    stream.synchronize()
    assert torch.equal(torch.stack(copies).cpu(), expected.unsqueeze(0).expand(32, -1, -1))


@pytest.mark.parametrize("selector,topk", [("native", 512), ("native", 2048), ("mk", 512), ("mk_scan", 512)])
@pytest.mark.parametrize("kind", ["close", "cascade", "saturated"])
def test_fast_topk_deterministic_special_low_mantissa_cutoff(selector, topk, kind):
    length = 4096
    raw = torch.arange(length, device="cuda", dtype=torch.int32) & 3
    if kind == "saturated":
        raw &= 1
        raw[-(topk - 1):] = 0x20000000
        raw[-3:] = torch.tensor([1 << 20, 1 << 11, 1 << 2], device="cuda", dtype=torch.int32)
    elif kind == "cascade":
        raw &= 1
        raw[-(topk - 1):] = 0x40000000
        raw[-3:] = torch.tensor([1 << 19, 1 << 10, 1 << 1], device="cuda", dtype=torch.int32)
    else:
        raw |= 0x3F800000
        raw[-(topk - 1):] = 0x3FC00000 + torch.arange(topk - 1, device="cuda", dtype=torch.int32) * 12345
    storage = torch.full((2, 65536), 0x7FC00000, device="cuda", dtype=torch.int32)
    storage[0, :length] = raw
    storage[1, :length] = raw.flip(0)
    if kind == "close":
        storage[:, 0] = 0
    score = storage.view(torch.float32)
    lengths = torch.full((2,), length, dtype=torch.int32, device="cuda")
    # Integer order preserves these nonnegative finite scores, including subnormals.
    reference = storage.cpu().to(torch.float64) if kind != "close" else score.cpu()
    expected = _stable_id_reference(reference, lengths.cpu(), topk, selector=selector)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _deterministic_selector(selector, score, lengths, topk)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = _deterministic_selector(selector, score, lengths, topk)
    with torch.cuda.stream(stream):
        copies = []
        for _ in range(32):
            graph.replay()
            copies.append(out.clone())
    stream.synchronize()
    assert torch.equal(torch.stack(copies).cpu(), expected.unsqueeze(0).expand(32, -1, -1))
    if kind != "close":
        storage[:, :length].copy_(storage[:, :length].flip(1))
    else:
        score.neg_()
    reference = storage.cpu().to(torch.float64) if kind != "close" else score.cpu()
    expected = _stable_id_reference(reference, lengths.cpu(), topk, selector=selector)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("selector", ["mk", "mk_scan"])
@pytest.mark.parametrize("value", [float("inf"), 0.0, -(2**-20)])
def test_fast_topk_deterministic_special_whole_root(selector, value):
    torch.manual_seed(7838)
    length, topk = 2304, 512
    score = torch.full((2, 65536), float("nan"), device="cuda", dtype=torch.float32)
    score[:, :length] = -1
    ids = torch.randperm(length, device="cuda")[:topk]
    score[0, ids] = value
    score[1, length - 1 - ids] = value
    lengths = torch.full((2,), length, device="cuda", dtype=torch.int32)
    expected = _stable_id_reference(score.cpu(), lengths.cpu(), topk, selector=selector)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _deterministic_selector(selector, score, lengths, topk)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = _deterministic_selector(selector, score, lengths, topk)
    with torch.cuda.stream(stream):
        copies = []
        for _ in range(32):
            graph.replay()
            copies.append(out.clone())
    stream.synchronize()
    assert torch.equal(torch.stack(copies).cpu(), expected.unsqueeze(0).expand(32, -1, -1))
    score[:, :length].copy_(score[:, :length].flip(1))
    expected = _stable_id_reference(score.cpu(), lengths.cpu(), topk, selector=selector)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    assert torch.equal(out.cpu(), expected)


@pytest.mark.parametrize("selector", ["mk", "mk_scan"])
@pytest.mark.parametrize("kind", ["uniform", "low_bits", "wide_bits"])
@pytest.mark.parametrize("negative", [False, True])
def test_fast_topk_deterministic_special_known_prefix(selector, kind, negative):
    length, topk = 2208, 512
    raw = torch.arange(length, device="cuda", dtype=torch.int32)
    if kind == "uniform":
        raw.zero_()
    elif kind == "low_bits":
        raw &= 511
    else:
        raw = (raw * 480) & 0xFFFFF
    values = (raw + 0x3F800000).view(torch.float32)
    score = torch.full((2, 65536), float("nan"), device="cuda", dtype=torch.float32)
    score[0, :length] = values
    score[1, :length] = values.flip(0)
    if negative:
        score.neg_()
    lengths = torch.full((2,), length, device="cuda", dtype=torch.int32)
    expected = _stable_id_reference(score.cpu(), lengths.cpu(), topk, selector=selector)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _deterministic_selector(selector, score, lengths, topk)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = _deterministic_selector(selector, score, lengths, topk)
    with torch.cuda.stream(stream):
        copies = []
        for _ in range(32):
            graph.replay()
            copies.append(out.clone())
    stream.synchronize()
    assert torch.equal(torch.stack(copies).cpu(), expected.unsqueeze(0).expand(32, -1, -1))
    score[:, :length].neg_()
    expected = _stable_id_reference(score.cpu(), lengths.cpu(), topk, selector=selector)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    assert torch.equal(out.cpu(), expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
