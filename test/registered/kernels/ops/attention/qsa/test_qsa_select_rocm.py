"""gfx95 QSA block selection through ``QSAIndexer.select_prefill_tokens`` and
``QSAIndexer.select_decode_tokens``.

On gfx95 the blocks come from topk_v2, which selects exactly where ``fast_topk``
keeps at most 4096 candidates of a row's threshold bin and drops the rest. Prefill
in batch-invariant mode gives ties to the smallest block indices.
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsv4 import topk as topk_v2_module
from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.layers.attention.qsa.mqa import (
    HAS_TILELANG,
    qsa_mqa_decode,
    tilelang_qsa_mqa_prefill_relative,
)
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

pytestmark = [
    pytest.mark.skipif(not is_gfx95_supported(), reason="gfx95-only selection path"),
    pytest.mark.skipif(not HAS_TILELANG, reason="tilelang unavailable"),
]

HEAD_DIM = 128
NUM_Q_HEADS = 4
RATIO = 4
BLOCK_TOPK = 512
TOKEN_TOPK = 2048
ROWS_PER_REQUEST = 32
SECOND_REQUEST_KEYS = 1000


def _select(q, k, key_counts):
    """select_prefill_tokens on q and k (bf16, on the GPU) over requests packed back to
    back, one per entry of key_counts (its compressed keys): the last ROWS_PER_REQUEST
    tokens of each request are query rows, and a row sees the blocks completed before
    it. Returns the selected tokens and each row's start and end in the keys."""
    starts, ends, positions, seq_lens, base = [], [], [], [], 0
    for count in key_counts:
        seq_len = RATIO * count + 1
        row_positions = torch.arange(seq_len - ROWS_PER_REQUEST, seq_len)
        starts.append(torch.full((ROWS_PER_REQUEST,), base))
        ends.append(base + (row_positions + 1) // RATIO)
        positions.append(row_positions)
        seq_lens.append(torch.full((ROWS_PER_REQUEST,), seq_len))
        base += count
    device = q.device
    starts = torch.cat(starts).int().to(device)
    ends = torch.cat(ends).int().to(device)
    indexer = SimpleNamespace(
        token_topk=TOKEN_TOPK, compress_ratio=RATIO, block_topk=BLOCK_TOPK
    )
    selected = QSAIndexer.select_prefill_tokens(
        indexer,
        q,
        k,
        starts,
        ends,
        torch.cat(positions).to(device),
        torch.cat(seq_lens).int().to(device),
        max_row_keys=max(key_counts),
    )
    return selected, starts, ends


def _blocks(selected_row, length):
    tokens = selected_row[selected_row >= 0].long()
    blocks = torch.unique(tokens // RATIO)
    return blocks[blocks < length]  # the row's incomplete group is not a block


# 8190 is not a multiple of 4: topk_v2 rejects such a logits width unless it is rounded up.
@pytest.mark.parametrize("num_keys", [8192, 8190])
def test_select_prefill_tokens_exact_on_narrow_band_scores(num_keys):
    """Scores packed into one fp16 coarse bin (all about 22.6 +- 1): a row of the first
    request has about 8,190 members of its threshold bin, twice the 4096 that fast_topk
    stages, so it dropped members of the true top-k. Each row's selected blocks,
    relative to its request, must hold exactly its top-512 scores. A second, shorter
    request packed behind the first starts at a nonzero column of the keys."""
    gen = torch.Generator().manual_seed(7)
    # With q all ones, a head's score of key j is relu(sum_d k[j, d]) = 64 +- 0.6.
    key_counts = (num_keys, SECOND_REQUEST_KEYS)
    k = 0.5 + 0.05 * torch.randn(sum(key_counts), 1, HEAD_DIM, generator=gen)
    q = torch.ones(len(key_counts) * ROWS_PER_REQUEST, NUM_Q_HEADS, HEAD_DIM)
    q, k = q.bfloat16().cuda(), k.bfloat16().cuda()

    selected, starts, ends = _select(q, k, key_counts)

    scores = tilelang_qsa_mqa_prefill_relative(q, k, starts, ends, num_keys)
    for row in range(q.shape[0]):
        length = int(ends[row] - starts[row])
        blocks = _blocks(selected[row], length)
        assert blocks.numel() == BLOCK_TOPK, row
        expected = torch.topk(scores[row, :length], BLOCK_TOPK).values.sort().values
        assert torch.equal(scores[row, blocks].sort().values, expected), row


def test_select_prefill_tokens_ties_go_to_the_smallest_blocks_when_batch_invariant(
    monkeypatch,
):
    """Every key equal, so all of a row's scores tie: in batch-invariant mode each row
    must select blocks 0..511, the smallest indices."""
    monkeypatch.setattr(qsa_indexer, "is_batch_invariant_mode_enabled", lambda: True)
    gen = torch.Generator().manual_seed(7)
    num_keys = 4096
    k = torch.full((num_keys, 1, HEAD_DIM), 0.05)
    q = torch.randn(ROWS_PER_REQUEST, NUM_Q_HEADS, HEAD_DIM, generator=gen)
    q, k = q.bfloat16().cuda(), k.bfloat16().cuda()

    selected, starts, ends = _select(q, k, (num_keys,))

    scores = tilelang_qsa_mqa_prefill_relative(q, k, starts, ends, num_keys)
    for row in range(q.shape[0]):
        length = int(ends[row] - starts[row])
        assert torch.all(scores[row, :length] == scores[row, 0]), row
        blocks = _blocks(selected[row], length)
        assert torch.equal(blocks.cpu(), torch.arange(BLOCK_TOPK)), row


PAGE_SIZE = 16
NUM_KEYS = 8192
VERIFY_ROWS = 4
SHORT_LENGTHS = (0, 1, 300, 512, 513)
# A decode graph's logits width: its page table of 4096 pages.
GRAPH_WIDTH = 4096 * PAGE_SIZE
# Row lengths around k and around topk_v2's thresholds for splitting a row across blocks.
GRAPH_LENGTHS = (GRAPH_WIDTH, 49153, 40961, 60000, 40960, 513, 512, 0)
# Stands in for logits the scorer leaves unset: a selection that reads one shows.
UNSET = 1.0e30


def _assert_top_blocks(selected, scores, lengths):
    """Each row of ``selected`` holds distinct blocks below the row's length, then -1:
    every block of a row that fits BLOCK_TOPK, else exactly a top BLOCK_TOPK of the
    row's scores (a tie at the cut may go either way)."""
    for row, length in enumerate(lengths.tolist()):
        count = min(length, BLOCK_TOPK)
        assert torch.all(selected[row, count:] == -1), row
        blocks = selected[row, :count].long()
        if length <= BLOCK_TOPK:
            assert torch.equal(blocks.sort().values.cpu(), torch.arange(length)), row
            continue
        assert blocks.unique().numel() == count, row
        assert 0 <= int(blocks.min()) and int(blocks.max()) < length, row
        expected = torch.topk(scores[row, :length], BLOCK_TOPK).values.sort().values
        assert torch.equal(scores[row, blocks].sort().values, expected), row


def _select_decode(q, k_cache, page_table, lengths, max_model_len):
    """select_decode_tokens, deferring the expansion of blocks to tokens."""
    indexer = SimpleNamespace(
        token_topk=TOKEN_TOPK, compress_ratio=RATIO, block_topk=BLOCK_TOPK
    )
    return QSAIndexer.select_decode_tokens(
        indexer,
        q,
        k_cache,
        page_table,
        lengths,
        max_model_len,
        lengths,
        lengths,
        defer_expansion=True,
    )


def _stubbed_scorer_operands(rows):
    """Operands of a select_decode_tokens whose scorer is stubbed: q's device only."""
    operand = torch.zeros(rows, 1, 1, device="cuda")
    return operand, operand, operand


@pytest.fixture
def decode_scores(monkeypatch):
    """The scores select_decode_tokens hands its selection, one tensor per call. The
    logits the scorer leaves unset (rows that fit the top-k, columns past a row's
    length) are overwritten with UNSET first."""
    handed = []

    def score(q, k_cache, page_table, lengths, max_model_len, *, min_scored_len, **kw):
        logits = qsa_mqa_decode(
            q,
            k_cache,
            page_table,
            lengths,
            max_model_len,
            min_scored_len=min_scored_len,
            **kw,
        )
        handed.append(logits.clone())
        columns = torch.arange(max_model_len, device=logits.device)
        unset = (columns >= lengths[:, None]) | (lengths <= min_scored_len)[:, None]
        logits.masked_fill_(unset, UNSET)
        return logits

    monkeypatch.setattr(qsa_indexer, "qsa_mqa_decode", score)
    return handed


def _narrow_band_decode_operands():
    """CPU operands of decode rows over one request's NUM_KEYS compressed keys: the
    VERIFY_ROWS draft rows (the last keys short of NUM_KEYS, three of the four lengths
    not a multiple of 4) and rows of SHORT_LENGTHS. With q all ones, a head's score of
    key j is relu(sum_d k[j, d]) = 64 +- 0.6, so every score lies in one fp16 coarse bin
    (all about 22.6 +- 1)."""
    gen = torch.Generator().manual_seed(7)
    num_pages = NUM_KEYS // PAGE_SIZE
    lengths = [NUM_KEYS - VERIFY_ROWS + 1 + j for j in range(VERIFY_ROWS)]
    lengths += SHORT_LENGTHS
    k_cache = 0.5 + 0.05 * torch.randn(num_pages, PAGE_SIZE, 1, HEAD_DIM, generator=gen)
    q = torch.ones(len(lengths), NUM_Q_HEADS, HEAD_DIM)
    # Eight pages more than the keys fill: columns past every row's length.
    page_table = torch.arange(num_pages + 8).clamp(max=num_pages - 1).int()
    return (
        q.bfloat16(),
        k_cache.bfloat16(),
        page_table.repeat(len(lengths), 1),
        torch.tensor(lengths, dtype=torch.int32),
    )


def test_select_decode_tokens_exact_on_narrow_band_scores(decode_scores):
    """Each draft row has about 8,190 members of its fp16 threshold bin, twice the 4096
    that fast_topk stages, so it dropped members of the true top-k, as in the prefill
    test. Each row must select exactly its top-512 scores, and a row of at most 513
    keys all its blocks, from logits the scorer leaves unset."""
    q, k_cache, page_table, lengths = (t.cuda() for t in _narrow_band_decode_operands())

    selected = _select_decode(
        q, k_cache, page_table, lengths, page_table.shape[1] * PAGE_SIZE
    )

    [scores] = decode_scores
    _assert_top_blocks(selected, scores, lengths)


@pytest.mark.parametrize("rows", [16, 256])
def test_select_decode_tokens_exact_under_graph_replay(monkeypatch, rows):
    """The selection of a decode graph at its 65,536-wide logits, captured once and
    replayed on new scores that all sit in one fp16 coarse bin (100 + U(0, 1)), where
    fast_topk is not exact: 16 rows are few enough for topk_v2 to split a long row
    across blocks, 256 rows run one block a row."""
    gen = torch.Generator(device="cuda").manual_seed(rows)
    lengths = torch.tensor(
        GRAPH_LENGTHS * (rows // len(GRAPH_LENGTHS)), dtype=torch.int32
    )
    lengths = lengths.cuda()
    past_length = torch.arange(GRAPH_WIDTH, device="cuda") >= lengths[:, None]
    logits = torch.empty(rows, GRAPH_WIDTH, device="cuda")

    def refill():
        logits.copy_(
            100.0 + torch.rand(rows, GRAPH_WIDTH, generator=gen, device="cuda")
        )
        logits.masked_fill_(past_length, UNSET)

    monkeypatch.setattr(qsa_indexer, "qsa_mqa_decode", lambda *args, **kwargs: logits)
    operands = _stubbed_scorer_operands(rows)
    refill()
    # Two eager runs before the capture, as the graph runners do.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _select_decode(*operands, lengths, GRAPH_WIDTH)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _select_decode(*operands, lengths, GRAPH_WIDTH)

    for _ in range(2):
        refill()
        captured.fill_(-9)
        graph.replay()
        torch.cuda.synchronize()
        _assert_top_blocks(captured, logits, lengths)


@pytest.mark.parametrize("case", ["not_gfx95", "row_stride_not_a_multiple_of_4"])
def test_select_decode_tokens_keeps_fast_topk(monkeypatch, case):
    """topk_v2 does not run, and fast_topk selects as before, off gfx95 and when the
    logits row stride is not a multiple of 4 (topk_v2 rejects it; a scorer that
    accepts any compressed page size can produce one)."""

    def topk_v2_must_not_run(*args, **kwargs):
        raise AssertionError("topk_v2 ran")

    monkeypatch.setattr(topk_v2_module, "topk_transform_paged_v2", topk_v2_must_not_run)
    width = 4096
    if case == "not_gfx95":
        monkeypatch.setattr(qsa_indexer, "is_gfx95_supported", lambda: False)
    else:
        width -= 6
    gen = torch.Generator(device="cuda").manual_seed(3)
    # Continuous scores: a row's top-k set is unique.
    logits = torch.randn(8, width, generator=gen, device="cuda")
    lengths = torch.tensor([width, 3000, 1000, 600, 513, 512, 1, 0], dtype=torch.int32)
    lengths = lengths.cuda()
    monkeypatch.setattr(qsa_indexer, "qsa_mqa_decode", lambda *args, **kwargs: logits)

    selected = _select_decode(*_stubbed_scorer_operands(8), lengths, width)
    _assert_top_blocks(selected, logits, lengths)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
