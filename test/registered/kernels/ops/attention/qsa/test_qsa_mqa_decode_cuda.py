"""The CUDA QSA indexer decode kernel must write the TileLang decode kernel's
logits bit for bit (one row per CTA, and verify rows sharing a page-table row),
also under CUDA graph replay. Turns red if either kernel changes its arithmetic
or the CUDA kernel reads a wrong page, row or length."""

import sys

import pytest
import torch

from sglang.srt.layers.attention.qsa.mqa import (
    HAS_TILELANG,
    cuda_qsa_mqa_decode,
    qsa_mqa_decode,
    tilelang_qsa_mqa_decode,
    use_cuda_qsa_mqa_decode,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-small")

HEADS = 4
HEAD_DIM = 128
PAGE = 16

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not HAS_TILELANG
    or not use_cuda_qsa_mqa_decode(
        torch.empty(1, HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
        torch.empty(1, PAGE, 1, HEAD_DIM, dtype=torch.bfloat16, device="cuda"),
    ),
    reason="the CUDA QSA decode kernel does not cover this device",
)


def _make_inputs(lengths, rows_per_request=1, seed=0):
    """Random keys and queries; each request gets distinct random pages, and the
    rows of a request share its page-table row."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    lengths = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    rows = lengths.numel()
    max_pages = max(1, -(-int(lengths.max()) // PAGE))
    requests = rows // rows_per_request
    pages = requests * max_pages + 7
    k_cache = torch.randn(
        pages, PAGE, 1, HEAD_DIM, generator=generator, device="cuda"
    ).to(torch.bfloat16)
    q = torch.randn(rows, HEADS, HEAD_DIM, generator=generator, device="cuda").to(
        torch.bfloat16
    )
    table = torch.randperm(pages, generator=generator, device="cuda")[
        : requests * max_pages
    ].reshape(requests, max_pages)
    page_table = table.repeat_interleave(rows_per_request, dim=0).to(torch.int32)
    return q, k_cache, page_table, lengths


def _written(lengths, max_model_len, min_scored_len):
    positions = torch.arange(max_model_len, device="cuda")[None, :]
    return (positions < lengths[:, None]) & (lengths > min_scored_len)[:, None]


def _assert_bitwise(actual, expected, lengths, min_scored_len):
    if min_scored_len == 0:
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        return
    written = _written(lengths, actual.shape[1], min_scored_len)
    differ = (actual.view(torch.int32) != expected.view(torch.int32)) & written
    assert not differ.any(), f"{int(differ.sum())} logits differ"


@pytest.mark.parametrize("min_scored_len", [0, 512])
@pytest.mark.parametrize(
    "lengths",
    [
        [1, 15, 16, 17, 63, 64, 65, 600],
        [513, 2048, 3001, 0, 512, 1029, 4095, 777] * 8,
    ],
)
def test_one_row_per_cta_matches_tilelang(lengths, min_scored_len):
    q, k_cache, page_table, lengths = _make_inputs(lengths)
    max_model_len = int(lengths.max()) + 37
    expected = tilelang_qsa_mqa_decode(
        q, k_cache, page_table, lengths, max_model_len, min_scored_len=min_scored_len
    )
    actual = cuda_qsa_mqa_decode(
        q, k_cache, page_table, lengths, max_model_len, min_scored_len=min_scored_len
    )
    _assert_bitwise(actual, expected, lengths, min_scored_len)


@pytest.mark.parametrize("min_scored_len", [0, 512])
@pytest.mark.parametrize("rows_per_request", [2, 4])
def test_verify_rows_share_key_loads(rows_per_request, min_scored_len):
    # min_scored_len=0 compares the whole -inf-filled output, so a row writing
    # past its own length (up to the last row's) is caught.
    starts = [5, 511, 530, 1987, 3000, 64, 700, 1290]
    lengths = [s + i for s in starts for i in range(rows_per_request)]
    q, k_cache, page_table, lengths = _make_inputs(lengths, rows_per_request)
    max_model_len = int(lengths.max())
    expected = tilelang_qsa_mqa_decode(
        q, k_cache, page_table, lengths, max_model_len, min_scored_len=min_scored_len
    )
    actual = cuda_qsa_mqa_decode(
        q,
        k_cache,
        page_table,
        lengths,
        max_model_len,
        min_scored_len=min_scored_len,
        rows_per_request=rows_per_request,
    )
    _assert_bitwise(actual, expected, lengths, min_scored_len)


def test_cuda_graph_replay_matches_eager():
    q, k_cache, page_table, lengths = _make_inputs([700, 2500, 1300, 90] * 4, 4)
    max_model_len = 4096

    def run():
        return cuda_qsa_mqa_decode(
            q, k_cache, page_table, lengths, max_model_len, rows_per_request=4
        )

    run()  # compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = run()
    q.copy_(torch.randn_like(q, dtype=torch.float32).to(torch.bfloat16))
    eager = run()
    graph.replay()
    assert torch.equal(replayed.view(torch.int32), eager.view(torch.int32))


def test_dispatch_covers_bf16_and_leaves_fp8_to_tilelang():
    q, k_cache, page_table, lengths = _make_inputs([1000, 2000])
    assert use_cuda_qsa_mqa_decode(q, k_cache)
    assert not use_cuda_qsa_mqa_decode(
        q.to(torch.float8_e4m3fn), k_cache.to(torch.float8_e4m3fn)
    )
    expected = cuda_qsa_mqa_decode(q, k_cache, page_table, lengths, 2048)
    actual = qsa_mqa_decode(q, k_cache, page_table, lengths, 2048)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
