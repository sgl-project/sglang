"""Unit tests for ``combine_topk_swa_indices`` (DSV4 sparse prefill).

Checks the Triton kernel against a per-row torch reference:

1. ``test_trailing_extend_matches_reference``: the V4 layout, where the query
   rows are the trailing extend tokens of each request and every top-k entry
   inside the scanned prefix is valid.
2. ``test_negative_one_holes_are_kept``: ``-1`` entries inside the top-k prefix
   stay ``-1`` instead of being shifted by ``compressed_base``.
3. ``test_non_trailing_query_positions``: absolute ``query_pos`` that are not
   the trailing extend tokens, with a cross-chunk ``query_start_loc`` offset.
4. ``test_swa_only_layer``: ``topk == 0`` writes only the window.
"""

import pytest
import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    combine_topk_swa_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="combine_topk_swa_indices requires CUDA"
)

DEVICE = "cuda"
WINDOW = 128
# flash_mla_sparse_fwd reads rows in 128-wide tiles; the combined width is
# padded to that multiple.
TOPK_ALIGNMENT = 128


def _i32(values):
    return torch.tensor(values, dtype=torch.int32, device=DEVICE)


def _reference(
    topk_indices,
    query_start_loc,
    query_pos,
    seq_lens,
    gather_lens,
    compressed_base,
    swa_base,
    window_size,
    compress_ratio,
    topk,
):
    num_tokens = topk_indices.shape[0]
    width = -(-(topk + window_size) // TOPK_ALIGNMENT) * TOPK_ALIGNMENT
    out = torch.full((num_tokens, width), -1, dtype=torch.int32, device=DEVICE)
    lens = torch.zeros(num_tokens, dtype=torch.int32, device=DEVICE)
    qsl = query_start_loc.tolist()
    base = qsl[0]
    for r in range(seq_lens.shape[0]):
        gather_start = int(seq_lens[r]) - int(gather_lens[r])
        for token_idx in range(qsl[r] - base, qsl[r + 1] - base):
            pos = int(query_pos[token_idx])
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            vals = topk_indices[token_idx, :topk_len]
            out[token_idx, :topk_len] = torch.where(
                vals >= 0, vals + compressed_base[r], torch.full_like(vals, -1)
            )
            window = torch.arange(swa_len, dtype=torch.int32, device=DEVICE)
            out[token_idx, topk_len : topk_len + swa_len] = (
                swa_base[r] + window + pos - swa_len + 1 - gather_start
            )
            lens[token_idx] = topk_len + swa_len
    return out, lens


def _check(**kwargs):
    got_idx, got_lens = combine_topk_swa_indices(**kwargs)
    ref_idx, ref_lens = _reference(**kwargs)
    assert torch.equal(got_lens, ref_lens), (got_lens.tolist(), ref_lens.tolist())
    assert torch.equal(got_idx, ref_idx)


def _trailing_case(seq_lens, extend_lens, topk, compress_ratio, seed=0):
    """Rows are the trailing ``extend_lens[r]`` tokens of request ``r``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    query_pos = []
    for seq_len, extend_len in zip(seq_lens, extend_lens):
        query_pos.extend(range(seq_len - extend_len, seq_len))
    num_tokens = len(query_pos)
    topk_indices = torch.randint(
        0, 1 << 20, (num_tokens, topk), generator=gen, dtype=torch.int32
    ).to(DEVICE)
    starts = [0]
    for extend_len in extend_lens:
        starts.append(starts[-1] + extend_len)
    gather_lens = [min(s, e + WINDOW - 1) for s, e in zip(seq_lens, extend_lens)]
    return dict(
        topk_indices=topk_indices,
        query_start_loc=_i32(starts),
        query_pos=_i32(query_pos),
        seq_lens=_i32(seq_lens),
        gather_lens=_i32(gather_lens),
        compressed_base=_i32([1000 * r for r in range(len(seq_lens))]),
        swa_base=_i32([5000 + 300 * r for r in range(len(seq_lens))]),
        window_size=WINDOW,
        compress_ratio=compress_ratio,
        topk=topk,
    )


@pytest.mark.parametrize("compress_ratio", [4, 128])
@pytest.mark.parametrize(
    "seq_lens, extend_lens",
    [([96, 144], [3, 2]), ([7, 300, 1000], [7, 130, 5]), ([1], [1])],
)
def test_trailing_extend_matches_reference(seq_lens, extend_lens, compress_ratio):
    _check(
        **_trailing_case(seq_lens, extend_lens, topk=64, compress_ratio=compress_ratio)
    )


def test_negative_one_holes_are_kept():
    case = _trailing_case([512, 640], [4, 4], topk=64, compress_ratio=4)
    topk_indices = case["topk_indices"]
    # Holes inside the scanned prefix (every row scans the full top-k here).
    topk_indices[:, 0] = -1
    topk_indices[:, 5] = -1
    topk_indices[3, 10:20] = -1
    _check(**case)
    got_idx, got_lens = combine_topk_swa_indices(**case)
    assert (got_idx[:, 0] == -1).all()
    assert (got_idx[:, 5] == -1).all()
    assert (got_idx[3, 10:20] == -1).all()
    # The scanned prefix still counts the holes.
    assert int(got_lens[0]) == 64 + WINDOW


def test_non_trailing_query_positions():
    # Two requests; each rank of a two-way interleave holds every other row of
    # the extend, so the query positions are not the trailing tokens. The
    # query_start_loc carries a cross-chunk offset that the kernel rebases.
    seq_lens = [96, 144]
    extend_lens = [6, 4]
    query_pos = [90, 92, 94, 140, 142]
    starts = [10, 13, 15]
    num_tokens = len(query_pos)
    topk = 32
    topk_indices = torch.arange(
        num_tokens * topk, dtype=torch.int32, device=DEVICE
    ).view(num_tokens, topk)
    gather_lens = [min(s, e + WINDOW - 1) for s, e in zip(seq_lens, extend_lens)]
    _check(
        topk_indices=topk_indices,
        query_start_loc=_i32(starts),
        query_pos=_i32(query_pos),
        seq_lens=_i32(seq_lens),
        gather_lens=_i32(gather_lens),
        compressed_base=_i32([0, 24]),
        swa_base=_i32([48, 181]),
        window_size=WINDOW,
        compress_ratio=4,
        topk=topk,
    )


def test_swa_only_layer():
    case = _trailing_case([200, 50], [2, 2], topk=0, compress_ratio=4)
    case["topk_indices"] = torch.zeros((4, 1), dtype=torch.int32, device=DEVICE)
    _check(**case)
    _, got_lens = combine_topk_swa_indices(**case)
    assert got_lens.tolist() == [WINDOW, WINDOW, 49, 50]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
