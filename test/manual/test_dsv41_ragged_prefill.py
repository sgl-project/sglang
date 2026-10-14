"""Standalone CUDA regression for ragged sparse-prefill history gathering."""

import unittest

import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    CompressedGather,
    SparsePrefillChunkCache,
    SparsePrefillWorkspace,
)


def make_cache(lengths, query_lengths, device):
    tensor = lambda values: torch.tensor(values, dtype=torch.int32, device=device)
    starts = [0]
    positions = []
    for length, queries in zip(lengths, query_lengths):
        starts.append(starts[-1] + queries)
        positions.extend(range(length - queries, length))
    return SparsePrefillChunkCache(
        num_reqs=len(lengths),
        num_qo_tokens=sum(query_lengths),
        max_seq_len=max(lengths),
        swa_window_size=128,
        swa_page_size=256,
        seq_lens=tensor(lengths),
        seq_lens_cpu=lengths,
        query_start_loc=tensor(starts),
        query_pos=tensor(positions),
        swa_token_ids=tensor([]),
        swa_first_pos=tensor(lengths),
        swa_gather_lens=tensor([0] * len(lengths)),
        swa_offsets=tensor([0] * (len(lengths) + 1)),
    )


def rectangular_gather(cache, page_table, ratio, page_size):
    width = max(cache.max_seq_len // ratio, 1)
    first = cache.query_start_loc[:-1].clamp_max(cache.num_qo_tokens - 1).long()
    positions = torch.arange(width, device=page_table.device)
    slots = (
        (
            page_table[first][:, positions // page_size] * page_size
            + positions % page_size
        )
        .reshape(-1)
        .clamp_min(0)
        .int()
    )
    bases, swa = cache._workspace_bases(width)
    return CompressedGather(slots, bases, swa)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestRaggedPrefill(unittest.TestCase):
    def test_gather_and_attention(self):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        torch.manual_seed(42)
        device = "cuda"
        for ratio in (1, 2, 4):
            for lengths, queries in (
                ([513, 99, 1, 2049], [3, 2, 1, 2]),
                ([513, 99, 1, 2049], [3, 0, 1, 2]),
                ([513, 513], [2, 2]),
                ([1, 1], [1, 1]),
            ):
                with self.subTest(ratio=ratio, lengths=lengths, queries=queries):
                    cache = make_cache(lengths, queries, device)
                    old_cache = make_cache(lengths, queries, device)
                    page_size = 256
                    blocks = (max(lengths) + page_size - 1) // page_size
                    # Non-contiguous page-table rows, with shuffled physical pages.
                    table = torch.randint(
                        1,
                        40,
                        (sum(queries), blocks * 2),
                        device=device,
                        dtype=torch.int32,
                    )[:, ::2]
                    for start, end in zip(
                        cache.query_start_loc.tolist()[:-1],
                        cache.query_start_loc.tolist()[1:],
                    ):
                        if end > start:
                            table[start:end] = table[start].clone()
                    old = rectangular_gather(old_cache, table, ratio, page_size)
                    old_cache.compressed[ratio] = old
                    new = cache.ensure_compressed(ratio, table, page_size)
                    self.assertEqual(
                        new.flat_token_ids.numel(), sum(x // ratio for x in lengths)
                    )
                    for i, length in enumerate(lengths):
                        n = length // ratio
                        a, b = int(new.compressed_base[i]), int(old.compressed_base[i])
                        torch.testing.assert_close(
                            new.flat_token_ids[a : a + n], old.flat_token_ids[b : b + n]
                        )
                    # Include SWA rows as well as top-k and -1 sentinels.
                    swa_lens = [
                        min(length, query + 127)
                        for length, query in zip(lengths, queries)
                    ]
                    swa_offsets = (
                        torch.tensor([0] + swa_lens, device=device, dtype=torch.int32)
                        .cumsum(0)
                        .int()
                    )
                    for c, g in ((cache, new), (old_cache, old)):
                        c.swa_gather_lens = torch.tensor(
                            swa_lens, device=device, dtype=torch.int32
                        )
                        c.swa_offsets = swa_offsets
                        g.swa_base = g.flat_token_ids.numel() + swa_offsets[:-1]
                    raw = torch.full(
                        (sum(queries), 128), -1, device=device, dtype=torch.int32
                    )
                    for row, pos in enumerate(cache.query_pos.tolist()):
                        n = min((pos + 1) // ratio, 128)
                        if n:
                            raw[row, :n] = torch.randperm(
                                (pos + 1) // ratio, device=device
                            )[:n].int()
                    ni, nl = cache.combine_compressed(ratio, raw)
                    oi, ol = old_cache.combine_compressed(ratio, raw)
                    torch.testing.assert_close(nl, ol)
                    values = (
                        torch.randn(
                            41 * page_size, 1, 512, device=device, dtype=torch.bfloat16
                        )
                        * 0.1
                    )
                    swa = (
                        torch.randn(
                            sum(swa_lens), 1, 512, device=device, dtype=torch.bfloat16
                        )
                        * 0.1
                    )
                    nkv = torch.cat((values[new.flat_token_ids.long()], swa))
                    okv = torch.cat((values[old.flat_token_ids.long()], swa))
                    valid = ni >= 0
                    torch.testing.assert_close(
                        nkv[ni[valid].long()], okv[oi[valid].long()]
                    )
                    q = (
                        torch.randn(
                            sum(queries), 64, 512, device=device, dtype=torch.bfloat16
                        )
                        * 0.1
                    )

                    def attn(kv, indices, lens):
                        return flash_mla_sparse_fwd(
                            q=q,
                            kv=kv,
                            indices=indices.unsqueeze(1),
                            sm_scale=512**-0.5,
                            d_v=512,
                            attn_sink=torch.zeros(64, device=device),
                            topk_length=lens,
                        )[0]

                    torch.testing.assert_close(
                        attn(nkv, ni, nl), attn(okv, oi, ol), rtol=0, atol=0
                    )

    def test_00_mixed_batch_under_memory_budget(self):
        # Run this method alone in a fresh process. The allocator limit applies
        # only to this test process, never to a resident serving process.
        budget = 2 << 30
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(budget / total)
        try:
            cache = make_cache([44198, 577152, 95723, 51427, 6144], [1] * 5, "cuda")
            table = torch.arange(2255, device="cuda", dtype=torch.int32).expand(5, -1)
            with self.assertRaises(torch.OutOfMemoryError):
                SparsePrefillWorkspace(torch.device("cuda")).get(2902511)
            gather = cache.ensure_compressed(1, table, 256)
            workspace = SparsePrefillWorkspace(torch.device("cuda")).get(
                gather.flat_token_ids.numel() + 16751
            )
            self.assertEqual(workspace.shape[0], 791395)
            print("2 GiB process budget: rectangular OOM, ragged PASS", flush=True)
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0)

    def test_reported_workspace_sizes(self):
        for lengths, swa, expected_old, expected_new in (
            ([44198, 577152, 95723, 51427, 6144], 16751, 2902511, 791395),
            ([820552] * 14, 2786, 11490514, 11490514),
        ):
            cache = make_cache(lengths, [1] * len(lengths), "cuda")
            blocks = (max(lengths) + 255) // 256
            table = torch.arange(blocks, device="cuda", dtype=torch.int32).expand(
                len(lengths), -1
            )
            gather = cache.ensure_compressed(1, table, 256)
            old_rows = len(lengths) * max(lengths) + swa
            new_rows = gather.flat_token_ids.numel() + swa
            self.assertEqual(old_rows, expected_old)
            self.assertEqual(new_rows, expected_new)
            ws = SparsePrefillWorkspace(torch.device("cuda"))
            before = torch.cuda.memory_allocated()
            buf = ws.get(new_rows)
            allocated = torch.cuda.memory_allocated() - before
            print(
                f"WORKSPACE old={old_rows * 1024 / 2**30:.6f} GiB new={buf.numel() * buf.element_size() / 2**30:.6f} GiB allocated={allocated / 2**30:.6f} GiB",
                flush=True,
            )
            del ws, buf


if __name__ == "__main__":
    unittest.main(verbosity=2)
