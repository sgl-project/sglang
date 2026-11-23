"""TOP DeepSeek-V4 sparse attention: per-cell, compact and owner-cell workspaces give the same bytes per query."""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

SM90 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9


def _v4_cache(pages, page_size, generator):
    """A V4 paged cache of finite values: random E4M3 nope bytes (no NaN codes), BF16 rope, UE8M0 scales near 1."""
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN,
    )

    data = page_size * NOPE_ROPE_BYTES
    raw = page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)
    bytes_per_page = -(-raw // NOPE_ROPE_BYTES) * NOPE_ROPE_BYTES
    cache = torch.zeros((pages, bytes_per_page), dtype=torch.uint8)
    tokens = cache[:, :data].view(pages, page_size, NOPE_ROPE_BYTES)
    nope = torch.randint(
        0, 256, (pages, page_size, 448), generator=generator, dtype=torch.uint8
    )
    nope[(nope & 0x7F) == 0x7F] = 0  # no E4M3 NaN codes
    tokens[..., :448] = nope
    rope = (torch.randn(pages, page_size, 64, generator=generator) * 0.1).bfloat16()
    tokens[..., 448:] = rope.view(torch.uint8).view(pages, page_size, 128)
    scales = cache[:, data : data + page_size * PADDED_SCALE_PER_TOKEN].view(
        pages, page_size, PADDED_SCALE_PER_TOKEN
    )
    scales[..., :7] = torch.randint(
        120, 128, (pages, page_size, 7), generator=generator, dtype=torch.uint8
    )
    return cache.cuda()


@unittest.skipUnless(SM90, "SM90 sparse FlashMLA")
class TestDsv4Workspace(unittest.TestCase):
    def _selections(
        self, requests, rows_per_request, width_c4, window, page_size, generator
    ):
        from sglang.srt.true_on_policy.dsv4_attention import KVSelection

        rows = requests * rows_per_request
        c4_slots = 4096
        c4 = torch.full((rows, width_c4), -1, dtype=torch.int32)
        swa = torch.full((rows, window), -1, dtype=torch.int32)
        c4_len = torch.zeros(rows, dtype=torch.int32)
        swa_len = torch.zeros(rows, dtype=torch.int32)
        for q in range(requests):
            base = torch.randperm(c4_slots, generator=generator)[: width_c4 + 8]
            for j in range(rows_per_request):
                r = q * rows_per_request + j
                # Verify rows of a request: mostly the same compressed blocks, a few swapped in.
                pick = torch.cat(
                    [base[: width_c4 - 4 - j], base[width_c4 : width_c4 + 4 + j]]
                )
                n = width_c4 - (r % 3) * 7
                c4[r, :n] = pick[:n].sort().values.int()
                c4_len[r] = n
                # The SWA window slides by one per row, newest first.
                end = 1000 * q + 300 + j
                swa[r, :window] = torch.arange(end, end - window, -1, dtype=torch.int32)
                swa_len[r] = window - (r % 2)
        c4_cache = _v4_cache(c4_slots // page_size, page_size, generator)
        swa_cache = _v4_cache(requests * 1000 // page_size + 4, page_size, generator)
        return rows, [
            KVSelection(
                cache=c4_cache,
                indices=c4.cuda(),
                lengths=c4_len.cuda(),
                page_size=page_size,
                layout="V4",
            ),
            KVSelection(
                cache=swa_cache,
                indices=swa.cuda(),
                lengths=swa_len.cuda(),
                page_size=page_size,
                layout="V4",
            ),
        ]

    def test_modes_match(self):
        from sglang.srt.true_on_policy import dsv4_attention

        generator = torch.Generator().manual_seed(0)
        for requests, per_request in ((1, 1), (3, 4), (16, 4), (5, 7)):
            rows, selections = self._selections(
                requests, per_request, 512, 128, 64, generator
            )
            q = (
                (torch.randn(rows, 64, 512, generator=generator) * 0.5)
                .bfloat16()
                .cuda()
            )
            sink = torch.randn(64, generator=generator).cuda()
            kwargs = dict(attn_sink=sink, scale=512**-0.5, d_v=512)
            per_cell = dsv4_attention.sparse_attention(q, selections, **kwargs)
            owners = dsv4_attention.sparse_attention(
                q, selections, shared_rows=True, **kwargs
            )
            with mock.patch.object(dsv4_attention, "DEDUP_MIN_CELLS", 1):
                compact = dsv4_attention.sparse_attention(q, selections, **kwargs)
            self.assertTrue(torch.isfinite(per_cell.float()).all())
            self.assertTrue(
                torch.equal(owners.view(torch.int16), per_cell.view(torch.int16)),
                (requests, per_request),
            )
            self.assertTrue(
                torch.equal(compact.view(torch.int16), per_cell.view(torch.int16)),
                (requests, per_request),
            )

    def test_owner_mode_dequantizes_each_slot_once(self):
        from sglang.srt.true_on_policy import dsv4_attention

        generator = torch.Generator().manual_seed(1)
        rows, selections = self._selections(4, 4, 512, 128, 64, generator)
        _, packed, lengths = dsv4_attention._gather_chunk(
            selections, 512, shared_rows=True
        )
        # Packed workspace rows: one per distinct (selection, slot), however many rows select it.
        distinct = 0
        for s in selections:
            cols = torch.arange(s.indices.shape[1], device="cuda")[None]
            valid = (s.indices >= 0) & (cols < s.lengths[:, None])
            distinct += torch.unique(s.indices[valid]).numel()
        used = torch.unique(packed[packed >= 0]).numel()
        self.assertEqual(used, distinct)


if __name__ == "__main__":
    unittest.main()
