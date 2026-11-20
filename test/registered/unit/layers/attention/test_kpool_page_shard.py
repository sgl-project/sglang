import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.kpool_page_shard import KPoolPageShard, padded_rows
from sglang.srt.mem_cache.index_key_cache import IndexKeyCache
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, stage="base-a", runner_config="cpu")


def _check_distributed(rank, world, init_file):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(init_file).as_uri(), rank=rank, world_size=world
    )
    try:
        page_size = 64
        torch.manual_seed(301)
        physical = torch.randperm(32 * world * page_size).float().view(-1, page_size)
        for size in (2, world):
            teams = [
                dist.new_group(list(range(start, start + size)))
                for start in range(0, world, size)
            ]
            group = teams[rank // size]
            owned_cache = physical[rank % size :: size].contiguous()
            dcp = SimpleNamespace(
                world_size=size, rank_in_group=rank % size, device_group=group
            )
            tp = SimpleNamespace(world_size=world, rank_in_group=rank)
            with (
                get_parallel().override(
                    attn_dcp_size=size,
                    attn_tp_size=world,
                    dcp_group=dcp,
                    attn_tp_group=tp,
                ),
                envs.SGLANG_OPT_DSA_KPOOL_INDEX_CACHE_SHARDS.override(size),
            ):
                actual = KPoolPageShard.create("cpu")
                assert (
                    actual.world_size,
                    actual.world_rank,
                    actual.size,
                    actual.rank,
                ) == (world, rank, size, rank % size)
            for exchange in ("logits", "candidates"):
                layout = KPoolPageShard(size, rank % size, world, rank, group, exchange)
                # Fragmented superpages, a shared prefix and a deliberately
                # incomplete last superpage exercise physical vs logical IDs.
                bases = torch.tensor(
                    [[7, 19, 2], [7, 5, 22], [3, 30, 6], [9, 21, 0], [1, 8, 17]]
                )
                ids = (bases[:, :, None] * world + torch.arange(world)).flatten(1)[
                    :, :-1
                ]
                lengths = torch.tensor(
                    [0, 63, 65, 257, ids.shape[1] * page_size - 1], dtype=torch.int32
                )
                staged, mapping = layout.fetch_pages(owned_cache, ids)
                torch.testing.assert_close(staged[mapping], physical[ids])
                # The common long prefill has aligned request boundaries and
                # fragmented physical superpages. Read two layers through one plan.
                aligned_ids = (
                    torch.tensor([5, 9, 5, 2])[:, None] * size + torch.arange(size)
                ).flatten()[:-1]
                for planned_ids, aligned in (
                    (aligned_ids, True),
                    (ids.flatten(), False),
                ):
                    fetch = layout.plan_fetch(planned_ids, aligned=aligned)
                    for layer in (0, 1):
                        staged, mapping = layout.fetch_pages(
                            owned_cache + layer, planned_ids, fetch
                        )
                        torch.testing.assert_close(
                            staged[mapping], physical[planned_ids] + layer
                        )
                for rows in (1, 3, 5):
                    table = ids[:rows]
                    stride, start, local_lens, local_table = layout.decode_layout(
                        table, lengths[:rows], page_size
                    )
                    local = owned_cache[local_table.long()].flatten(1)
                    output = layout.transpose_scores(
                        local, local_lens, stride, page_size, topk=8
                    )
                    expected = padded_rows(
                        physical[table].flatten(1), rank * stride, stride
                    )
                    lens = padded_rows(lengths[:rows], rank * stride, stride)
                    for i in range(stride):
                        n = int(lens[i])
                        if exchange == "logits":
                            torch.testing.assert_close(output[i, :n], expected[i, :n])
                        else:
                            k = min(n, 8)
                            torch.testing.assert_close(
                                output[i, :n].topk(k).indices,
                                expected[i, :n].topk(k).indices,
                            )
                            assert torch.isneginf(output[i, n:]).all()
    finally:
        dist.destroy_process_group()


class TestKPoolPageShard(CustomTestCase):
    def test_real_distributed_page_reads_and_exchanges(self):
        with tempfile.TemporaryDirectory() as root:
            mp.start_processes(
                _check_distributed,
                args=(4, str(Path(root, "pg"))),
                nprocs=4,
                start_method="fork",
            )

    @staticmethod
    def _pool():
        return SimpleNamespace(
            page_size=256,
            index_kpool=4,
            index_head_dim=128,
            index_bytes_per_token=132,
            quant_block_size=128,
            index_k_with_scale_buffer_dtype=torch.uint8,
            layer_num=3,
            skip_topk_layers=[False, True, False],
            device="cpu",
            custom_mem_pool=None,
            cpu_offloading_chunk_size=512,
            memory_saver_adapter=SimpleNamespace(region=lambda _: nullcontext()),
        )

    def test_single_page_decode_has_unit_column_stride(self):
        table = torch.tensor([[24, 25, 26]], dtype=torch.int32)
        length = torch.tensor([129], dtype=torch.int32)
        for shards in (4, 8):
            for rank in range(shards):
                with self.subTest(shards=shards, rank=rank):
                    layout = KPoolPageShard(shards, rank, 8, rank, None, "logits")
                    _, _, _, local = layout.decode_layout(table, length, 64)
                    self.assertEqual(local.shape, (shards, 1))
                    self.assertEqual(local.stride(1), 1)
                    expected = 24 // shards if rank < 3 else 0
                    self.assertEqual(local[0, 0].item(), expected)

    def test_compact_physical_allocation(self):
        pool = self._pool()
        baseline = IndexKeyCache(pool, index_buf_size=256 * 31)
        for size in (2, 4, 8):
            pool.kpool_page_shard = KPoolPageShard(size, 0, 8, 0, None, "logits")
            compact = IndexKeyCache(pool, index_buf_size=256 * 31)
            self.assertEqual(
                sum(t.nbytes for t in baseline.buffer),
                size * sum(t.nbytes for t in compact.buffer),
            )
            self.assertEqual(compact.buffer[1].shape[0], 0)

    @patch("torch.cuda.synchronize")
    def test_retract_and_restore_into_different_superpages(self, _synchronize):
        pool = self._pool()
        source = torch.tensor([16, 17, 18, 19, 4, 5, 6])
        target = torch.tensor([24, 25, 26, 27, 8, 9, 10])

        def token_slots(pages):
            # Retract in the middle of the final page.
            return (
                pages[:, None] * pool.page_size + torch.arange(pool.page_size)
            ).flatten()[:-31]

        for rank in range(4):
            pool.kpool_page_shard = KPoolPageShard(4, rank, 4, rank, None, "logits")
            cache = IndexKeyCache(pool, index_buf_size=256 * 31)
            from_pages = pool.kpool_page_shard.owned_pages(source)
            to_pages = pool.kpool_page_shard.owned_pages(target)
            expected = []
            for i in (0, 2):
                buf = cache.buffer[i]
                data = (
                    torch.arange(len(from_pages) * buf.shape[1])
                    .view(len(from_pages), -1)
                    .to(torch.uint8)
                )
                data = data + (from_pages[:, None] * 17).to(torch.uint8) + rank + i
                buf[from_pages] = data
                expected.append(data)
            saved = cache.cpu_copy(token_slots(source))
            for buf in cache.buffer:
                buf.zero_()
            cache.load_cpu_copy(saved, token_slots(target))
            for i, data in zip((0, 2), expected):
                torch.testing.assert_close(cache.buffer[i][to_pages], data)
                self.assertEqual(
                    torch.count_nonzero(cache.buffer[i][from_pages]).item(), 0
                )


if __name__ == "__main__":
    unittest.main()
