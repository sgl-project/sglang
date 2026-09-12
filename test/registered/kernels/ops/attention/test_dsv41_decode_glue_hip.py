"""Single-launch replacements for the DeepSeek-V4.1 decode glue on HIP must be bitwise the torch chains they replace."""

from __future__ import annotations

import random
import sys

import pytest
import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel
import torch

from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
    expand_index_page_table,
    low_ratio_compression_metadata,
)
from sglang.kernels.ops.attention.dsv4.fp4_indexer import quantize_fp4_indexer_tensor
from sglang.kernels.ops.attention.dsv4.fp4_indexer_hip import (
    pack_fp4_query_flydsl,
    sort_selection_rows,
)
from sglang.srt.layers.attention.deepseek_v4_backend import (
    _expand_index_page_table,
    _low_ratio_compression_metadata,
)
from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
    CandidateBlocks,
    _aot_topk_sorts_output,
    topk_transform_paged_sorted,
    topk_within_candidate_blocks_hip,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()),
    reason="HIP decode glue for the AITER / FlyDSL DeepSeek-V4.1 path (gfx95x).",
)

DEVICE = "cuda"


def _seed(seed: int) -> random.Random:
    torch.manual_seed(seed)
    return random.Random(seed)


def _topk_inputs(bs, width, page_size, lens):
    # a row longer than its logits reads past the tensor (scores[0, len) and
    # page_table[0, len // page_size]), and what it selects there differs by launch
    assert max(lens, default=0) <= width, (lens, width)
    # distinct scores per row: the radix top-k breaks a tie at the threshold in
    # atomic-counter order, so two launches over ties can select different sets
    scores = (
        torch.stack([torch.randperm(width, device=DEVICE).float() for _ in range(bs)])
        * 0.37
    )
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=DEVICE)
    n_pages = (width + page_size - 1) // page_size
    page_table = torch.randint(
        0, 1 << 20, (bs, n_pages), dtype=torch.int32, device=DEVICE
    )
    return scores, seq_lens, page_table


@pytest.mark.skipif(
    not _aot_topk_sorts_output(), reason="sgl_kernel predates sort_output"
)
@pytest.mark.parametrize("topk", [512, 100])
@pytest.mark.parametrize("with_raw", [True, False])
def test_sorted_topk_epilogue_matches_transform_then_sort(topk: int, with_raw: bool):
    rng = _seed(topk)
    for width in (1024, 70000):
        for bs in (1, 33):
            page_size = rng.choice([16, 64])
            # the topk + 1 edge (a radix row of exactly topk picks) only where the
            # logits are that wide; at width == topk it would run past the row
            edges = [
                n
                for n in (0, 1, topk - 1, topk, topk + 1, width // 2, width)
                if n <= width
            ]
            lens = [rng.choice(edges) for _ in range(bs)]
            scores, seq_lens, page_table = _topk_inputs(bs, width, page_size, lens)
            # the unsorted transform, then sort_selection_rows' order (by position with
            # raw indices, by slot without); the Triton sort for a power-of-two k
            ref = torch.empty(bs, topk, dtype=torch.int32, device=DEVICE)
            ref_raw = torch.empty_like(ref) if with_raw else None
            torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
                scores, seq_lens, page_table, ref, page_size, ref_raw
            )
            if topk & (topk - 1) == 0:
                sort_selection_rows(ref, ref_raw)
            else:
                by = ref_raw if with_raw else ref
                key = torch.where(by < 0, torch.iinfo(torch.int32).max, by)
                order = torch.sort(key, dim=1, stable=True).indices
                ref = torch.gather(ref, 1, order)
                if with_raw:
                    ref_raw = torch.gather(ref_raw, 1, order)
            out = torch.empty_like(ref)
            out_raw = torch.empty_like(ref) if with_raw else None
            topk_transform_paged_sorted(
                scores, seq_lens, page_table, out, page_size, out_raw
            )
            assert torch.equal(out, ref), (bs, width, page_size, lens)
            if with_raw:
                assert torch.equal(out_raw, ref_raw)
            # padding last, keys ascending
            for row in (ref_raw if with_raw else ref).tolist():
                n = sum(x >= 0 for x in row)
                assert all(x < 0 for x in row[n:])
                assert row[:n] == sorted(row[:n])


def _candidates(rng, rows, num_blocks, topk_blocks, block_size, seq_lens):
    ids = torch.full((rows, topk_blocks), -1, dtype=torch.int32, device=DEVICE)
    for r in range(rows):
        reach = min(num_blocks, -(-int(seq_lens[r]) // block_size))
        picks = rng.sample(range(reach), min(reach, topk_blocks))
        rng.shuffle(picks)
        if picks:
            ids[r, : len(picks)] = torch.tensor(picks, dtype=torch.int32, device=DEVICE)
    block_lens = (seq_lens + block_size - 1) // block_size
    compact_lens = (torch.clamp(block_lens, max=topk_blocks) * block_size).to(
        torch.int32
    )
    width = topk_blocks * block_size
    return CandidateBlocks(
        ids=ids,
        compact_lens=compact_lens,
        compact_page_table=torch.zeros((rows, 1), dtype=torch.int32, device=DEVICE),
        compact_page_size=1 << (width - 1).bit_length(),
        block_size=block_size,
    )


@pytest.mark.parametrize("topk", [512])
@pytest.mark.parametrize("with_raw", [True, False])
def test_sorted_candidate_mapping_matches_pack_then_sort(topk: int, with_raw: bool):
    rng = _seed(11 + topk)
    block_size, topk_blocks, page_size = 64, 16, 64
    for rows in (1, 7):
        width = 8192
        lens = [
            rng.choice([0, 1, 300, topk, topk + 5, 3000, width]) for _ in range(rows)
        ]
        scores, seq_lens, page_table = _topk_inputs(rows, width, page_size, lens)
        cands = _candidates(
            rng, rows, width // block_size, topk_blocks, block_size, seq_lens
        )
        outs = []
        for sort in (False, True):
            page = torch.empty(rows, topk, dtype=torch.int32, device=DEVICE)
            raw = torch.empty_like(page) if with_raw else None
            topk_within_candidate_blocks_hip(
                scores,
                seq_lens,
                cands,
                page_table=page_table,
                page_size=page_size,
                page_indices=page,
                raw_indices=raw,
                sort_output=sort,
            )
            if not sort:
                sort_selection_rows(page, raw)
            outs.append((page, raw))
        assert torch.equal(outs[0][0], outs[1][0])
        if with_raw:
            assert torch.equal(outs[0][1], outs[1][1])


@pytest.mark.parametrize("bpp", [4])
def test_expand_index_page_table(bpp: int):
    _seed(5)
    for bs, n in ((1, 4608), (3, 17), (0, 10)):
        page_table = torch.randint(
            0, 1 << 20, (bs, n), dtype=torch.int32, device=DEVICE
        )
        ref = _expand_index_page_table(
            page_table, full_page_size=64 * bpp, compress_ratio=1, index_page_size=64
        )
        out = expand_index_page_table(page_table, bpp)
        assert out.dtype is torch.int32 and out.shape == ref.shape
        assert torch.equal(out, ref)
    # a strided (row-sliced) table is read through its strides
    page_table = torch.randint(0, 1 << 20, (8, 33), dtype=torch.int32, device=DEVICE)[
        ::2
    ]
    assert torch.equal(
        expand_index_page_table(page_table, 4),
        _expand_index_page_table(
            page_table, full_page_size=256, compress_ratio=1, index_page_size=64
        ),
    )


@pytest.mark.parametrize("loc_dtype", [torch.int64])
@pytest.mark.parametrize("ratios", [(1, 2)])
def test_low_ratio_compression_metadata(loc_dtype, ratios):
    rng = _seed(7)
    for rows, nw in ((1, 1), (9, 9), (12, 5)):
        seq_lens = torch.tensor(
            [rng.choice([0, 1, 2, 3, 1000, 1001, 65535]) for _ in range(rows)],
            dtype=torch.int32,
            device=DEVICE,
        )
        raw_out_loc = torch.randint(0, 1 << 24, (nw,), dtype=loc_dtype, device=DEVICE)
        out = low_ratio_compression_metadata(seq_lens, raw_out_loc, ratios)
        assert set(out) == {
            f"c{r}_{k}" for r in ratios for k in ("out_loc", "topk_lengths_clamp1")
        }
        for r in ratios:
            ref_loc, ref_clamp1 = _low_ratio_compression_metadata(
                r, seq_lens, raw_out_loc
            )
            assert out[f"c{r}_out_loc"].dtype is ref_loc.dtype
            assert torch.equal(out[f"c{r}_out_loc"], ref_loc)
            assert out[f"c{r}_topk_lengths_clamp1"].dtype is ref_clamp1.dtype
            assert torch.equal(out[f"c{r}_topk_lengths_clamp1"], ref_clamp1)


def pack_fp4_query_flydsl_torch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The three-launch form of ``pack_fp4_query_flydsl``: the shared quantizer, then
    zeros and a permuted copy into the scale layout."""
    num_tokens, heads = q.shape[0], q.shape[1]
    assert heads % 16 == 0 and heads <= 64, heads
    q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
    q_fp4 = q_fp4.view(num_tokens, heads, 64)
    sf_bytes = q_sf.view(torch.uint8).view(num_tokens, heads // 16, 16, 4)
    q_scale = torch.zeros((num_tokens, 1, 4, 16, 4), dtype=torch.uint8, device=q.device)
    q_scale[:, 0, :, :, : heads // 16] = sf_bytes.permute(0, 3, 2, 1)
    return q_fp4, q_scale


@pytest.mark.parametrize("heads", [32])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_pack_fp4_query_flydsl_single_launch(heads: int, dtype):
    _seed(17)
    for tokens in (1, 40):
        q = torch.randn(tokens, heads, 128, device=DEVICE, dtype=dtype) * 4
        # exact fp4 grid points and tie values, zeros and a huge group
        q[0, 0, :32] = torch.tensor(
            [0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] * 4, device=DEVICE, dtype=dtype
        )
        q[0, 0, 32:64] = 0.0
        q[0, 0, 64:96] = 3.0e4
        ref_fp4, ref_scale = pack_fp4_query_flydsl_torch(q)
        fp4, scale = pack_fp4_query_flydsl(q)
        assert fp4.dtype is ref_fp4.dtype and scale.dtype is ref_scale.dtype
        assert fp4.shape == ref_fp4.shape and scale.shape == ref_scale.shape
        assert torch.equal(fp4, ref_fp4)
        assert torch.equal(scale, ref_scale)
    empty = torch.empty(0, heads, 128, device=DEVICE, dtype=dtype)
    fp4, scale = pack_fp4_query_flydsl(empty)
    assert fp4.shape == (0, heads, 64) and scale.shape == (0, 1, 4, 16, 4)


@pytest.mark.parametrize("compressed_kv", [False, True])
def test_rope_fake_quant_gathers_freqs_by_position(compressed_kv: bool):
    from sglang.kernels.ops.attention.dsv4.rope_fake_quant_fp4 import (
        rope_tail_fake_quant_fp4,
    )

    _seed(19)
    table = torch.polar(
        torch.ones(4096, 32, device=DEVICE),
        torch.rand(4096, 32, device=DEVICE) * 6.283,
    )
    for tokens, heads in ((1, 32), (17, 32)):
        x = torch.randn(tokens, heads, 128, device=DEVICE, dtype=torch.bfloat16) * 3
        for pos_dtype in (torch.int64, torch.int32):
            pos = torch.randint(0, 4096, (tokens,), device=DEVICE, dtype=pos_dtype)
            ref = rope_tail_fake_quant_fp4(
                x, table[pos], 64, compressed_kv=compressed_kv
            )
            out = rope_tail_fake_quant_fp4(
                x, table, 64, compressed_kv=compressed_kv, positions=pos
            )
            assert torch.equal(out, ref)


def test_page_table_from_req_to_token_matches_torch():
    from sglang.kernels.ops.attention.dsv4.attn_glue_hip import (
        page_table_from_req_to_token,
    )

    _seed(23)
    req_to_token = torch.randint(
        0, 2**20, (300, 8192), device=DEVICE, dtype=torch.int32
    )
    req_to_token[
        7, :64
    ] = -3  # torch floor-divides; the slot values are never negative in serving
    for bs, max_seq_len, page in (
        (1, 1000, 256),
        (64, 8191, 256),
    ):
        req = torch.randint(0, 300, (bs,), device=DEVICE, dtype=torch.int32)
        req[0] = 7
        ref = (req_to_token[req, :max_seq_len:page] // page).to(torch.int32)
        got = page_table_from_req_to_token(req_to_token, req, max_seq_len, page)
        assert got.shape == ref.shape and got.dtype == torch.int32
        assert torch.equal(got, ref)
    empty = page_table_from_req_to_token(
        req_to_token, torch.empty(0, device=DEVICE, dtype=torch.int32), 1000, 64
    )
    assert empty.shape == (0, 16)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
