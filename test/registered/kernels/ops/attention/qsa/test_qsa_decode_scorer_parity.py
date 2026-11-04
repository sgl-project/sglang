"""Bitwise parity of the QSA paged decode scorer across launch configurations.

Every case NaN-poisons the logits the kernel may leave unset, compares the scored
region bit for bit, records which launch ran, and runs a seeded positive control
(one perturbed key element) that the comparison must catch.
"""

import pytest
import torch

from sglang.srt.layers.attention.qsa import mqa
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() < (10, 0)
        or not mqa.HAS_TILELANG,
        reason="needs an SM100+ GPU and TileLang",
    ),
]

PAGE_SIZE = 16
HEAD_DIM = 128
INDEX_HEADS = 4
PADDED_HEADS = 8
MIN_SCORED_LEN = 512
ROWS_PER_REQUEST = 4


def _verify_batch(requests: int, max_len: int, seed: int, dtype=torch.bfloat16):
    """Target-verify scorer inputs: four rows per request share a page-table row,
    with compressed lengths L..L+3; heads 4-7 are the zero padding the indexer adds."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows = requests * ROWS_PER_REQUEST
    base = torch.randint(1, max_len - ROWS_PER_REQUEST, (requests,), generator=gen)
    lens = (base[:, None] + torch.arange(ROWS_PER_REQUEST)).reshape(-1)
    max_pages = -(-max_len // PAGE_SIZE)
    pages = requests * max_pages + 1
    perm = torch.randperm(pages - 1, generator=gen)[: requests * max_pages] + 1
    page_table = perm.view(requests, 1, max_pages).expand(-1, ROWS_PER_REQUEST, -1)
    page_table = page_table.reshape(rows, max_pages).to(torch.int32)
    q = torch.zeros(rows, PADDED_HEADS, HEAD_DIM)
    q[:, :INDEX_HEADS] = torch.randn(rows, INDEX_HEADS, HEAD_DIM, generator=gen)
    k_cache = torch.randn(pages, PAGE_SIZE, 1, HEAD_DIM, generator=gen)
    return (
        q.to(dtype).cuda(),
        k_cache.to(dtype).cuda(),
        page_table.cuda(),
        lens.to(torch.int32).cuda(),
        max_pages * PAGE_SIZE,
    )


def _score(inputs, **kwargs):
    q, k_cache, page_table, lens, max_model_len = inputs
    out = mqa.tilelang_qsa_mqa_decode(
        q,
        k_cache,
        page_table,
        lens,
        max_model_len,
        min_scored_len=MIN_SCORED_LEN,
        **kwargs,
    )
    torch.cuda.synchronize()
    return out


def _scored_mask(lens: torch.Tensor, width: int) -> torch.Tensor:
    positions = torch.arange(width, device=lens.device)[None, :]
    return (positions < lens[:, None]) & (lens[:, None] > MIN_SCORED_LEN)


def _assert_bitwise_equal(actual, expected, lens):
    mask = _scored_mask(lens, actual.shape[1])
    assert mask.any()
    assert not torch.isnan(actual[mask]).any(), "a scored logit was never written"
    assert torch.equal(actual[mask].view(torch.int32), expected[mask].view(torch.int32))


def _positive_control(inputs, reference, seed, **kwargs):
    """Perturb one key element that a scored row reads; the comparison must fire."""
    q, k_cache, page_table, lens, max_model_len = inputs
    gen = torch.Generator(device="cpu").manual_seed(seed)
    scored_rows = torch.nonzero(lens > MIN_SCORED_LEN).flatten().cpu()
    row = int(scored_rows[torch.randint(len(scored_rows), (1,), generator=gen)])
    position = int(torch.randint(int(lens[row]), (1,), generator=gen))
    page = int(page_table[row, position // PAGE_SIZE])
    perturbed = k_cache.clone()
    # Aligning the key with the row's first head makes that head's ReLU term
    # strictly positive, so the score must move.
    perturbed[page, position % PAGE_SIZE, 0] = q[row, 0].to(k_cache.dtype)
    changed = _score((q, perturbed, page_table, lens, max_model_len), **kwargs)
    mask = _scored_mask(lens, changed.shape[1])
    assert not torch.equal(
        changed[mask].view(torch.int32), reference[mask].view(torch.int32)
    ), "the bitwise comparison missed a perturbed key"


@pytest.fixture
def poisoned_logits(monkeypatch):
    # Unset logits start as NaN, so a missing store cannot pass on reused memory.
    monkeypatch.setenv("SGLANG_QSA_SCORER_POISON", "1")


@pytest.mark.parametrize("requests", [1, 4, 16, 32, 64, 128])
def test_decode_ctas_per_row_levels_are_bitwise(
    requests, monkeypatch, poisoned_logits, record_property
):
    inputs = _verify_batch(requests, max_len=4096, seed=requests)
    rows = requests * ROWS_PER_REQUEST
    device = inputs[0].device

    monkeypatch.setattr(mqa, "_DECODE_CTAS_PER_ROW_LEVELS", (1, 16, 256))
    stock_ctas = mqa._decode_ctas_per_row(rows, device)
    stock = _score(inputs)
    monkeypatch.setattr(mqa, "_DECODE_CTAS_PER_ROW_LEVELS", (1, 256))
    coarse_ctas = mqa._decode_ctas_per_row(rows, device)
    coarse = _score(inputs)

    record_property("ctas_per_row", f"rows={rows} stock={stock_ctas} new={coarse_ctas}")
    _assert_bitwise_equal(coarse, stock, inputs[3])
    _positive_control(inputs, coarse, seed=requests)


def _spy_factories(monkeypatch, names):
    """Wrap TileLang kernel factories to record which specialization each call built."""
    built = []
    for name in names:
        factory = getattr(mqa, name)

        def spy(*args, _name=name, _factory=factory, **config):
            built.append((_name, config))
            return _factory(*args, **config)

        monkeypatch.setattr(mqa, name, spy)
    return built


@pytest.fixture
def sm107_dispatch(monkeypatch):
    # The grouped4 path is gated to SM107 for speed only; its arithmetic is
    # arch-independent, so other SM100-class parts exercise it under this override.
    real = torch.cuda.get_device_capability
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device=None: (10, 7) if real(device)[0] == 10 else real(device),
    )


@pytest.mark.parametrize("ctas_per_group", [32, 256])
@pytest.mark.parametrize("requests", [17, 32, 64, 100, 144])
def test_grouped4_verify_matches_the_per_row_scorer(
    requests, ctas_per_group, monkeypatch, poisoned_logits, sm107_dispatch, record_property
):
    inputs = _verify_batch(requests, max_len=8192, seed=1000 + requests)
    monkeypatch.setattr(mqa, "_QSA_GROUPED4_VERIFY", True)
    monkeypatch.setattr(mqa, "_QSA_GROUPED4_VERIFY_CTAS", ctas_per_group)
    built = _spy_factories(
        monkeypatch,
        ["_tilelang_qsa_mqa_decode_grouped4_kernel", "_tilelang_qsa_mqa_decode_kernel"],
    )

    per_row = _score(inputs, rows_per_group=1, num_query_heads=INDEX_HEADS)
    grouped = _score(inputs, rows_per_group=4, num_query_heads=INDEX_HEADS)

    record_property("kernels", repr(built))
    assert [name for name, _ in built] == [
        "_tilelang_qsa_mqa_decode_kernel",
        "_tilelang_qsa_mqa_decode_grouped4_kernel",
    ]
    assert built[1][1]["ctas_per_row"] == ctas_per_group
    _assert_bitwise_equal(grouped, per_row, inputs[3])
    _positive_control(inputs, grouped, seed=requests, rows_per_group=4, num_query_heads=INDEX_HEADS)


def test_grouped4_verify_leaves_fp8_index_on_the_stock_path(
    monkeypatch, poisoned_logits, sm107_dispatch
):
    inputs = _verify_batch(64, max_len=4096, seed=7, dtype=torch.float8_e4m3fn)
    monkeypatch.setattr(mqa, "_QSA_GROUPED4_VERIFY", True)
    built = _spy_factories(monkeypatch, ["_tilelang_qsa_mqa_decode_grouped4_kernel"])
    _score(inputs, rows_per_group=4, num_query_heads=INDEX_HEADS)
    assert built == []


@pytest.mark.parametrize("ctas_per_group", [256, 512])
@pytest.mark.parametrize("requests", [17, 48, 96, 144])
def test_grouped4_chain_matches_the_per_row_scorer(
    requests, ctas_per_group, monkeypatch, poisoned_logits, sm107_dispatch, record_property
):
    inputs = _verify_batch(requests, max_len=16384, seed=2000 + requests)
    monkeypatch.setattr(mqa, "_QSA_GROUPED4_VERIFY", True)
    monkeypatch.setattr(mqa, "_QSA_GROUPED4_VERIFY_CHAIN", True)
    monkeypatch.setattr(mqa, "_QSA_GROUPED4_VERIFY_CHAIN_CTAS", ctas_per_group)
    built = _spy_factories(
        monkeypatch,
        ["_tilelang_qsa_mqa_decode_grouped4_chain_kernel", "_tilelang_qsa_mqa_decode_kernel"],
    )

    per_row = _score(inputs, rows_per_group=1, num_query_heads=INDEX_HEADS)
    chain = _score(inputs, rows_per_group=4, num_query_heads=INDEX_HEADS)

    record_property("kernels", repr(built))
    name, config = built[-1]
    assert name == "_tilelang_qsa_mqa_decode_grouped4_chain_kernel"
    assert config["ctas_per_row"] == ctas_per_group
    _assert_bitwise_equal(chain, per_row, inputs[3])
    _positive_control(inputs, chain, seed=requests, rows_per_group=4, num_query_heads=INDEX_HEADS)


@pytest.mark.parametrize("requests", [8, 16, 32, 144])
def test_draft_extend_grouping_is_bitwise_or_falls_back(
    requests, monkeypatch, poisoned_logits, sm107_dispatch, record_property
):
    inputs = _verify_batch(requests, max_len=8192, seed=3000 + requests)
    monkeypatch.setattr(mqa, "_QSA_GROUPED4_VERIFY", True)
    monkeypatch.setattr(mqa, "_QSA_GROUPED4_VERIFY_CHAIN", True)
    built = _spy_factories(
        monkeypatch,
        ["_tilelang_qsa_mqa_decode_grouped4_chain_kernel", "_tilelang_qsa_mqa_decode_kernel"],
    )

    per_row = _score(inputs, rows_per_group=1, num_query_heads=INDEX_HEADS)
    draft = _score(
        inputs, rows_per_group=4, num_query_heads=INDEX_HEADS, grouped4_only=True
    )

    record_property("kernels", repr(built))
    expected = (
        "_tilelang_qsa_mqa_decode_grouped4_chain_kernel"
        if requests > 16
        else "_tilelang_qsa_mqa_decode_kernel"
    )
    assert built[-1][0] == expected
    _assert_bitwise_equal(draft, per_row, inputs[3])


@pytest.mark.parametrize("bs", [17, 20, 64, 144])
def test_draft_extend_layout_groups_four_rows_per_request(bs):
    """The grouped4 kernel reads a group's pages and length through its last row, so
    each aligned 4-row block of the draft-extend layout must be one request with
    non-decreasing (compressed) lengths, padding included."""
    from sglang.srt.layers.attention.qsa.graph_metadata import _qsa_graph_layout_kernel

    gen = torch.Generator(device="cpu").manual_seed(bs)
    for num_padding in (0, bs // 3):
        real = bs - num_padding
        seq_lens = torch.randint(1, 262144, (bs,), generator=gen)
        seq_lens[:4] = torch.tensor([1, 2, 3, 5])
        seq_lens[real:] = 4
        req_pool = torch.zeros(bs, dtype=torch.int64)
        req_pool[:real] = torch.randperm(4096, generator=gen)[:real] + 1
        extend_lens = torch.full((bs,), ROWS_PER_REQUEST, dtype=torch.int32)
        rows = ROWS_PER_REQUEST * bs
        row_lens = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
        row_prefix = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
        row_req = torch.full((rows,), -1, dtype=torch.int32, device="cuda")
        _qsa_graph_layout_kernel[(bs + 1,)](
            seq_lens.to(torch.int32).cuda(),
            req_pool.cuda(),
            extend_lens.cuda(),
            row_lens,
            row_prefix,
            row_req,
            bs,
            rows,
            num_padding,
            num_warps=1,
        )
        lens = row_lens.view(bs, ROWS_PER_REQUEST).cpu()
        reqs = row_req.view(bs, ROWS_PER_REQUEST).cpu()
        assert (lens >= 1).all(), "a layout row was left unwritten"
        assert (reqs == reqs[:, :1]).all()
        assert (lens[:, 1:] >= lens[:, :-1]).all()
        compressed = lens // 4
        assert (compressed[:, 1:] >= compressed[:, :-1]).all()
