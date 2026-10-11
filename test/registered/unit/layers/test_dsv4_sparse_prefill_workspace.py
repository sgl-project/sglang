"""CPU checks for the DeepSeek-V4 sparse prefill workspace layout.

``flash_mla_sparse_fwd`` reads a flat bf16 workspace holding every request's
compressed-cache gather followed by the SWA gather, addressed through per-query
indices. Each request's compressed region must be sized by that request's own
sequence length. Padding every request to the batch's longest one made a single
~820K-token prompt that shares a prefill batch with ~25 short requests ask for
``num_reqs * max_seq_len // ratio`` rows (10.29 GiB) and run out of memory.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.attention.dsv4 import sparse_prefill_utils
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    WORKSPACE_DIM,
    SparsePrefillChunkCache,
    SparsePrefillWorkspace,
    combined_topk_width,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

SWA_WINDOW = 128
# Fake SWA-cache ids encode (request, position) so a resolved index is checkable.
SWA_ID_BASE = 1 << 28
SWA_ID_STRIDE = 1 << 20


def _i32(values):
    return torch.tensor(values, dtype=torch.int32)


def _make_cache(seq_lens, extend_lens, *, host_lens=True):
    """Query rows are the trailing extend tokens of each request."""
    query_pos, starts = [], [0]
    swa_ids, swa_first, swa_lens, swa_offsets = [], [], [], [0]
    for r, (seq_len, extend_len) in enumerate(zip(seq_lens, extend_lens)):
        query_pos.extend(range(seq_len - extend_len, seq_len))
        starts.append(starts[-1] + extend_len)
        gather_len = min(seq_len, extend_len + SWA_WINDOW - 1)
        first = seq_len - gather_len
        swa_ids.extend(
            SWA_ID_BASE + r * SWA_ID_STRIDE + p for p in range(first, seq_len)
        )
        swa_first.append(first)
        swa_lens.append(gather_len)
        swa_offsets.append(swa_offsets[-1] + gather_len)
    return SparsePrefillChunkCache(
        num_reqs=len(seq_lens),
        num_qo_tokens=len(query_pos),
        max_seq_len=max(seq_lens),
        swa_window_size=SWA_WINDOW,
        swa_page_size=SWA_WINDOW,
        seq_lens=_i32(seq_lens),
        query_start_loc=_i32(starts),
        query_pos=_i32(query_pos),
        swa_token_ids=_i32(swa_ids),
        swa_first_pos=_i32(swa_first),
        swa_gather_lens=_i32(swa_lens),
        swa_offsets=_i32(swa_offsets),
        seq_lens_cpu=list(seq_lens) if host_lens else None,
    )


def _page_table(cache, compress_ratio, c_page_size, extra_blocks=3):
    """Per-query page table; rows of a request repeat that request's pages."""
    num_blocks = -(-max(cache.max_seq_len // compress_ratio, 1) // c_page_size)
    num_blocks += extra_blocks  # metadata is usually wider than the live extent
    starts = cache.query_start_loc.tolist()
    table = torch.empty((cache.num_qo_tokens, num_blocks), dtype=torch.int32)
    for r in range(cache.num_reqs):
        pages = 1 + r * 4096 + torch.arange(num_blocks, dtype=torch.int32)
        table[starts[r] : starts[r + 1]] = pages
    return table


def _compressed_token_id(page_table, row, position, c_page_size):
    return int(page_table[row, position // c_page_size]) * c_page_size + (
        position % c_page_size
    )


def _reference_combine(
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
    out_indices=None,
    out_lens=None,
    swa_indices=None,
    swa_lengths=None,
):
    """Per-row torch model of ``combine_topk_swa_indices`` (default SWA layout)."""
    assert swa_indices is None
    num_tokens = topk_indices.shape[0]
    out = torch.full(
        (num_tokens, combined_topk_width(topk, window_size)), -1, dtype=torch.int32
    )
    lens = torch.zeros(num_tokens, dtype=torch.int32)
    qsl = query_start_loc.tolist()
    for r in range(seq_lens.shape[0]):
        gather_start = int(seq_lens[r]) - int(gather_lens[r])
        for t in range(qsl[r] - qsl[0], qsl[r + 1] - qsl[0]):
            pos = int(query_pos[t])
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)
            vals = topk_indices[t, :topk_len]
            out[t, :topk_len] = torch.where(vals >= 0, vals + compressed_base[r], -1)
            out[t, topk_len : topk_len + swa_len] = (
                int(swa_base[r])
                + pos
                - swa_len
                + 1
                - gather_start
                + torch.arange(swa_len, dtype=torch.int32)
            )
            lens[t] = topk_len + swa_len
    return out, lens


class TestDSV4SparsePrefillWorkspace(CustomTestCase):
    def _resolve_and_check(
        self, cache, gather, combined, lens, topk_indices, ratio, pt, c_page_size
    ):
        """Every combined index lands on the row its request and position name."""
        workspace_ids = torch.cat([gather.flat_token_ids, cache.swa_token_ids])
        starts = cache.query_start_loc.tolist()
        for r in range(cache.num_reqs):
            for t in range(starts[r], starts[r + 1]):
                pos = int(cache.query_pos[t])
                topk_len = min((pos + 1) // ratio, topk_indices.shape[1])
                row = combined[t, : int(lens[t])].tolist()
                for j, idx in enumerate(row[:topk_len]):
                    local = int(topk_indices[t, j])
                    if local < 0:
                        self.assertEqual(idx, -1)
                        continue
                    self.assertEqual(
                        int(workspace_ids[idx]),
                        _compressed_token_id(pt, starts[r], local, c_page_size),
                    )
                swa = row[topk_len:]
                expected = [
                    SWA_ID_BASE + r * SWA_ID_STRIDE + p
                    for p in range(pos - len(swa) + 1, pos + 1)
                ]
                self.assertEqual([int(workspace_ids[i]) for i in swa], expected)

    def test_long_prompt_with_short_requests_is_not_padded(self):
        # The production OOM: one ~820K-token prompt mid chunked prefill, batched
        # with 20 short requests.
        seq_lens = [820_000] + [150 + 13 * i for i in range(20)]
        extend_lens = [64] + [min(s, 40) for s in seq_lens[1:]]
        cache = _make_cache(seq_lens, extend_lens)
        total_swa = cache.swa_token_ids.numel()
        for ratio in (1, 2, 4):
            with self.subTest(ratio=ratio):
                c_page_size = 256
                gather = cache.ensure_compressed(
                    ratio, _page_table(cache, ratio, c_page_size), c_page_size
                )
                packed = sum(max(s // ratio, 1) for s in seq_lens)
                self.assertEqual(gather.flat_token_ids.numel(), packed)

                row_bytes = WORKSPACE_DIM * 2  # bf16
                workspace_rows = gather.flat_token_ids.numel() + total_swa
                long_alone_rows = seq_lens[0] // ratio + extend_lens[0] + SWA_WINDOW - 1
                # Each short request costs at most its own context plus a window,
                # never the long request's extent.
                short_rows = sum(
                    s + e + SWA_WINDOW for s, e in zip(seq_lens[1:], extend_lens[1:])
                )
                self.assertLessEqual(
                    workspace_rows * row_bytes,
                    (long_alone_rows + short_rows) * row_bytes,
                )
                # Padding to the longest extent would have been ~num_reqs times this.
                padded_rows = len(seq_lens) * (seq_lens[0] // ratio) + total_swa
                self.assertGreater(padded_rows, 20 * workspace_rows)

    def test_packed_layout_addresses_each_request(self):
        seq_lens = [3, 9, 1030, 4, 517]
        extend_lens = [3, 2, 17, 1, 5]
        gen = torch.Generator().manual_seed(0)
        for ratio, c_page_size in ((1, 64), (2, 64), (4, 64)):
            with self.subTest(ratio=ratio):
                cache = _make_cache(seq_lens, extend_lens)
                pt = _page_table(cache, ratio, c_page_size)
                gather = cache.ensure_compressed(ratio, pt, c_page_size)

                c_lens = [max(s // ratio, 1) for s in seq_lens]
                bases = [sum(c_lens[:r]) for r in range(len(seq_lens))]
                self.assertEqual(gather.compressed_base.tolist(), bases)
                self.assertEqual(
                    gather.swa_base.tolist(),
                    [sum(c_lens) + o for o in cache.swa_offsets[:-1].tolist()],
                )
                starts = cache.query_start_loc.tolist()
                for r, (base, c_len) in enumerate(zip(bases, c_lens)):
                    expected = [
                        max(_compressed_token_id(pt, starts[r], p, c_page_size), 0)
                        for p in range(c_len)
                    ]
                    self.assertEqual(
                        gather.flat_token_ids[base : base + c_len].tolist(), expected
                    )

                # Top-k as the indexer emits it: request-local positions below
                # (pos + 1) // ratio, with -1 holes.
                topk = 16
                topk_indices = torch.full(
                    (cache.num_qo_tokens, topk), -1, dtype=torch.int32
                )
                for t, pos in enumerate(cache.query_pos.tolist()):
                    visible = (pos + 1) // ratio
                    n = min(visible, topk)
                    if n:
                        picks = torch.randperm(visible, generator=gen)[:n]
                        topk_indices[t, :n] = picks.to(torch.int32)
                        if n > 2:
                            topk_indices[t, 0] = -1
                combined, lens = _reference_combine(
                    topk_indices=topk_indices,
                    query_start_loc=cache.query_start_loc,
                    query_pos=cache.query_pos,
                    seq_lens=cache.seq_lens,
                    gather_lens=cache.swa_gather_lens,
                    compressed_base=gather.compressed_base,
                    swa_base=gather.swa_base,
                    window_size=SWA_WINDOW,
                    compress_ratio=ratio,
                    topk=topk,
                )
                self._resolve_and_check(
                    cache, gather, combined, lens, topk_indices, ratio, pt, c_page_size
                )

    def test_c128_packs_per_request(self):
        seq_lens = [127, 5000, 300, 130]
        extend_lens = [7, 9, 3, 2]
        cache = _make_cache(seq_lens, extend_lens)
        c128_max = cache.max_seq_len // 128
        # Production masks a query's row past seq_lens_casual // 128; the last
        # query of a request covers its full extent.
        page_indices = torch.full(
            (cache.num_qo_tokens, c128_max + 5), -1, dtype=torch.int32
        )
        for r, end in enumerate(cache.query_start_loc[1:].tolist()):
            n = seq_lens[r] // 128
            page_indices[end - 1, :n] = 1000 * (r + 1) + torch.arange(
                n, dtype=torch.int32
            )
        with mock.patch.object(
            sparse_prefill_utils, "combine_topk_swa_indices", _reference_combine
        ):
            gather = cache.ensure_c128(page_indices)

        c_lens = [max(s // 128, 1) for s in seq_lens]
        self.assertEqual(gather.flat_token_ids.numel(), sum(c_lens))
        bases = [sum(c_lens[:r]) for r in range(len(seq_lens))]
        self.assertEqual(gather.compressed_base.tolist(), bases)
        workspace_ids = torch.cat([gather.flat_token_ids, cache.swa_token_ids])
        starts = cache.query_start_loc.tolist()
        for r in range(cache.num_reqs):
            for t in range(starts[r], starts[r + 1]):
                pos = int(cache.query_pos[t])
                topk_len = (pos + 1) // 128
                row = gather.combined_indices[t, :topk_len].tolist()
                self.assertEqual(
                    [int(workspace_ids[i]) for i in row],
                    [1000 * (r + 1) + j for j in range(topk_len)],
                )

    def test_host_lens_match_device_fallback(self):
        seq_lens = [70_000, 12, 333]
        extend_lens = [16, 12, 8]
        with_host = _make_cache(seq_lens, extend_lens)
        without_host = _make_cache(seq_lens, extend_lens, host_lens=False)
        pt = _page_table(with_host, 4, 64)
        a = with_host.ensure_compressed(4, pt, 64)
        b = without_host.ensure_compressed(4, pt, 64)
        self.assertTrue(torch.equal(a.flat_token_ids, b.flat_token_ids))
        self.assertTrue(torch.equal(a.compressed_base, b.compressed_base))
        self.assertTrue(torch.equal(a.swa_base, b.swa_base))

    def test_workspace_growth_releases_old_buffer_first(self):
        workspace = SparsePrefillWorkspace(torch.device("cpu"))
        workspace.get(4)
        held_during_alloc = []
        real_empty = torch.empty

        def spy_empty(*args, **kwargs):
            held_during_alloc.append(workspace._buffer)
            return real_empty(*args, **kwargs)

        with mock.patch.object(sparse_prefill_utils.torch, "empty", spy_empty):
            grown = workspace.get(9)
        self.assertEqual(held_during_alloc, [None])
        self.assertEqual(grown.shape, (9, 1, WORKSPACE_DIM))


if __name__ == "__main__":
    unittest.main()
