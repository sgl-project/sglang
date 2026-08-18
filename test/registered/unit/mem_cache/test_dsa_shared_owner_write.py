import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(_HAS_CUDA, "Triton kernels require CUDA")
class TestDSASharedOwnerWrite(unittest.TestCase):
    def test_materialize_dynamic_validation_is_eager_only(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            _validate_materialize_indexer_pages_dynamic,
        )

        seq_len = torch.tensor([64], dtype=torch.int32)
        source_pages = torch.tensor([[0]], dtype=torch.int32)
        target_pages = torch.tensor([[0]], dtype=torch.int32)

        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=True),
            patch("torch._assert_async") as assert_async,
        ):
            _validate_materialize_indexer_pages_dynamic(
                source_pages,
                target_pages,
                seq_len,
                page_size=64,
                source_page_capacity=1,
                target_page_capacity=1,
            )
            assert_async.assert_not_called()

        with (
            patch("torch.cuda.is_current_stream_capturing", return_value=False),
            patch("torch._assert_async") as assert_async,
        ):
            _validate_materialize_indexer_pages_dynamic(
                source_pages,
                target_pages,
                seq_len,
                page_size=64,
                source_page_capacity=1,
                target_page_capacity=1,
            )
            self.assertEqual(assert_async.call_count, 3)

    def test_materialize_indexer_pages_uses_bounded_graph_workers(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            _materialize_indexer_pages_grid,
        )

        self.assertEqual(_materialize_indexer_pages_grid(1, 16_386), (16_386,))
        self.assertEqual(_materialize_indexer_pages_grid(2, 16_386), (32_772,))
        self.assertEqual(_materialize_indexer_pages_grid(4, 16_386), (65_544,))
        self.assertEqual(_materialize_indexer_pages_grid(7, 16_386), (114_702,))
        self.assertEqual(_materialize_indexer_pages_grid(8, 16_386), (1_024,))
        self.assertEqual(_materialize_indexer_pages_grid(32, 16_386), (4_096,))

    def test_materialize_indexer_pages_copies_vmm_sources_to_pool_pages(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            materialize_indexer_pages_triton,
        )

        source = torch.arange(12 * 64, dtype=torch.uint8, device="cuda").view(12, 64)
        source_pages = torch.tensor(
            [[10, 7, 11, 8], [6, 9, 8, 7]], dtype=torch.int32, device="cuda"
        )
        pool_pages = torch.tensor(
            [[4, 1, 5, 2], [0, 3, 2, 1]], dtype=torch.int32, device="cuda"
        )
        seq_len = torch.tensor([130, 65], dtype=torch.int32, device="cuda")
        target = torch.full((8, 64), 0xFF, dtype=torch.uint8, device="cuda")

        materialize_indexer_pages_triton(
            target,
            source,
            source_pages,
            pool_pages,
            seq_len,
            page_size=64,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(target[4], source[10]))
        self.assertTrue(torch.equal(target[1], source[7]))
        self.assertTrue(torch.equal(target[5], source[11]))
        self.assertTrue(torch.equal(target[0], source[6]))
        self.assertTrue(torch.equal(target[3], source[9]))
        self.assertTrue(torch.all(target[2] == 0xFF))
        self.assertTrue(torch.all(target[6:] == 0xFF))

    def test_materialize_indexer_pages_supports_independent_table_strides(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            materialize_indexer_pages_triton,
        )

        source = torch.arange(12 * 64, dtype=torch.uint8, device="cuda").view(12, 64)
        source_pages = torch.tensor(
            [[10, 7, 11, 8], [6, 9, 8, 7]], dtype=torch.int32, device="cuda"
        )
        pool_page_storage = torch.tensor(
            [[4, 0], [1, 3], [5, 2], [2, 1]], dtype=torch.int32, device="cuda"
        )
        pool_pages = pool_page_storage.transpose(0, 1)
        self.assertNotEqual(source_pages.stride(0), pool_pages.stride(0))
        seq_len = torch.tensor([130, 65], dtype=torch.int32, device="cuda")
        target = torch.full((8, 64), 0xFF, dtype=torch.uint8, device="cuda")

        materialize_indexer_pages_triton(
            target,
            source,
            source_pages,
            pool_pages,
            seq_len,
            page_size=64,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(target[4], source[10]))
        self.assertTrue(torch.equal(target[1], source[7]))
        self.assertTrue(torch.equal(target[5], source[11]))
        self.assertTrue(torch.equal(target[0], source[6]))
        self.assertTrue(torch.equal(target[3], source[9]))

    def test_materialize_indexer_pages_replays_with_updated_length_and_table(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            materialize_indexer_pages_triton,
        )

        source = torch.arange(6 * 64, dtype=torch.uint8, device="cuda").view(6, 64)
        page_table = torch.tensor(
            [[4, 1, 5, 2], [0, 3, 2, 1]], dtype=torch.int32, device="cuda"
        )
        seq_len = torch.tensor([130, 65], dtype=torch.int32, device="cuda")
        target = torch.zeros((8, 64), dtype=torch.uint8, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            materialize_indexer_pages_triton(
                target,
                source,
                page_table,
                page_table,
                seq_len,
                page_size=64,
            )

        target.fill_(0xFF)
        page_table.copy_(torch.tensor([[3, 0, 2, 1], [5, 4, 1, 0]], device="cuda"))
        seq_len.copy_(torch.tensor([65, 1], device="cuda"))
        graph.replay()
        torch.cuda.synchronize()

        active = torch.tensor([3, 0, 5], device="cuda")
        self.assertTrue(torch.equal(target[active], source[active]))
        self.assertTrue(torch.all(target[1:3] == 0xFF))
        self.assertTrue(torch.all(target[4] == 0xFF))
        self.assertTrue(torch.all(target[6:] == 0xFF))

    def test_materialize_worker_replays_with_updated_length_and_table(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            materialize_indexer_pages_triton,
        )

        batch_size = 8
        pages_per_request = 4
        total_pages = batch_size * pages_per_request
        source = torch.arange(total_pages * 64, dtype=torch.int64, device="cuda").view(
            total_pages, 64
        )
        target_pages = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(
            batch_size, pages_per_request
        )
        source_pages = target_pages.clone()
        seq_len = torch.full((batch_size,), 130, dtype=torch.int32, device="cuda")
        target = torch.zeros_like(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            materialize_indexer_pages_triton(
                target,
                source,
                source_pages,
                target_pages,
                seq_len,
                page_size=64,
            )

        target.fill_(-1)
        source_pages.copy_(torch.flip(target_pages, dims=[1]))
        replay_lengths = torch.tensor(
            [1, 65, 130, 193, 1, 65, 130, 193],
            dtype=torch.int32,
            device="cuda",
        )
        seq_len.copy_(replay_lengths)
        graph.replay()
        torch.cuda.synchronize()

        for batch in range(batch_size):
            active_pages = (int(replay_lengths[batch]) + 63) // 64
            for page in range(pages_per_request):
                target_page = int(target_pages[batch, page])
                if page < active_pages:
                    source_page = int(source_pages[batch, page])
                    self.assertTrue(
                        torch.equal(target[target_page], source[source_page])
                    )
                else:
                    self.assertTrue(torch.all(target[target_page] == -1))

    def test_persistent_pool_cache_reuses_history_and_refreshes_tail(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            materialize_indexer_pages_triton,
        )

        source = torch.arange(4 * 64, dtype=torch.uint8, device="cuda").view(4, 64)
        page_table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32, device="cuda")
        seq_len = torch.tensor([130], dtype=torch.int32, device="cuda")
        target = torch.full((4, 64), 0xFF, dtype=torch.uint8, device="cuda")
        tags = torch.zeros(4, dtype=torch.int64, device="cuda")
        epoch = torch.ones((), dtype=torch.int32, device="cuda")

        materialize_indexer_pages_triton(
            target,
            source,
            page_table,
            page_table,
            seq_len,
            page_size=64,
            tags=tags,
            epoch=epoch,
        )
        first = target.clone()

        source[2].add_(7)
        source[0].add_(11)
        source[3].add_(13)
        materialize_indexer_pages_triton(
            target,
            source,
            page_table,
            page_table,
            seq_len,
            page_size=64,
            tags=tags,
            epoch=epoch,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(target[2], first[2]))
        self.assertTrue(torch.equal(target[0], first[0]))
        self.assertTrue(torch.equal(target[3], source[3]))

        epoch.add_(1)
        materialize_indexer_pages_triton(
            target,
            source,
            page_table,
            page_table,
            seq_len,
            page_size=64,
            tags=tags,
            epoch=epoch,
        )
        torch.cuda.synchronize()
        active = page_table[0, :3].long()
        self.assertTrue(torch.equal(target[active], source[active]))

    def test_materialize_worker_crosses_page_rounds_before_publishing_tags(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            materialize_indexer_pages_triton,
        )

        batch_size = 8
        active_pages = 130
        page_bytes = 8_448
        source = torch.randint(
            0,
            256,
            (batch_size * active_pages, page_bytes),
            dtype=torch.uint8,
            device="cuda",
        )
        page_table = torch.full((batch_size, 256), -1, dtype=torch.int32, device="cuda")
        page_table[:, :active_pages] = torch.arange(
            batch_size * active_pages, dtype=torch.int32, device="cuda"
        ).view(batch_size, active_pages)
        target = torch.zeros_like(source)
        tags = torch.zeros(batch_size * active_pages, dtype=torch.int64, device="cuda")
        epoch = torch.tensor(9, dtype=torch.int32, device="cuda")

        materialize_indexer_pages_triton(
            target,
            source,
            page_table,
            page_table,
            torch.full(
                (batch_size,),
                active_pages * 64,
                dtype=torch.int32,
                device="cuda",
            ),
            page_size=64,
            tags=tags,
            epoch=epoch,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(target, source))
        expected_tags = (9 << 32) | (
            torch.arange(batch_size * active_pages, dtype=torch.int64, device="cuda")
            + 1
        )
        self.assertTrue(torch.equal(tags, expected_tags))

    def test_materialize_allows_identical_shared_prefix_page_aliases(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            materialize_indexer_pages_triton,
        )

        source = torch.arange(4 * 64, dtype=torch.uint8, device="cuda").view(4, 64)
        source_pages = torch.tensor([[2], [2]], dtype=torch.int32, device="cuda")
        target_pages = torch.tensor([[0], [0]], dtype=torch.int32, device="cuda")
        target = torch.zeros((1, 64), dtype=torch.uint8, device="cuda")
        tags = torch.zeros(1, dtype=torch.int64, device="cuda")
        epoch = torch.tensor(3, dtype=torch.int32, device="cuda")

        materialize_indexer_pages_triton(
            target,
            source,
            source_pages,
            target_pages,
            torch.tensor([1, 1], dtype=torch.int32, device="cuda"),
            page_size=64,
            tags=tags,
            epoch=epoch,
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(target[0], source[2]))
        self.assertEqual(tags.item(), (3 << 32) | 3)

    def _run_case(self, rank: int, locations: torch.Tensor) -> None:
        from sglang.kernels.ops.kvcache.dsa_shared import (
            set_mla_kv_buffer_owner_triton,
        )

        cp_size = 4
        page_size = 4
        nope = torch.arange(
            locations.numel() * 5, dtype=torch.uint8, device="cuda"
        ).view(-1, 5)
        rope = (
            torch.arange(locations.numel() * 3, dtype=torch.uint8, device="cuda") + 91
        ).view(-1, 3)
        output = torch.zeros((16, 8), dtype=torch.uint8, device="cuda")
        expected = torch.zeros_like(output)

        set_mla_kv_buffer_owner_triton(
            output,
            locations,
            nope,
            rope,
            owner_rank=rank,
            owner_size=cp_size,
            page_size=page_size,
        )

        valid = locations >= 0
        pages = torch.div(locations.clamp_min(0), page_size, rounding_mode="floor")
        owned = valid & ((pages % cp_size) == rank)
        rows = torch.nonzero(owned, as_tuple=True)[0]
        owned_locations = locations.index_select(0, rows)
        local_locations = (
            torch.div(
                torch.div(owned_locations, page_size, rounding_mode="floor"),
                cp_size,
                rounding_mode="floor",
            )
            * page_size
            + owned_locations % page_size
        )
        expected[local_locations] = torch.cat(
            (nope.index_select(0, rows), rope.index_select(0, rows)), dim=-1
        )
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(output, expected))

    def test_owner_write_matches_reference_for_every_rank(self):
        locations = torch.tensor(
            [0, 3, 4, 7, 8, 12, 16, 20, 31, -1],
            dtype=torch.int64,
            device="cuda",
        )
        for rank in range(4):
            with self.subTest(rank=rank):
                self._run_case(rank, locations)

    def test_owner_write_replays_with_updated_static_inputs(self):
        from sglang.kernels.ops.kvcache.dsa_shared import (
            set_mla_kv_buffer_owner_triton,
        )

        locations = torch.tensor([0, 4, 8, 12], dtype=torch.int64, device="cuda")
        nope = torch.ones((4, 5), dtype=torch.uint8, device="cuda")
        rope = torch.full((4, 3), 2, dtype=torch.uint8, device="cuda")
        output = torch.zeros((8, 8), dtype=torch.uint8, device="cuda")

        set_mla_kv_buffer_owner_triton(
            output, locations, nope, rope, owner_rank=1, owner_size=4, page_size=4
        )
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            set_mla_kv_buffer_owner_triton(
                output,
                locations,
                nope,
                rope,
                owner_rank=1,
                owner_size=4,
                page_size=4,
            )

        output.zero_()
        locations.copy_(torch.tensor([4, 20, 0, -1], device="cuda"))
        nope.fill_(7)
        rope.fill_(9)
        graph.replay()
        torch.cuda.synchronize()

        expected = torch.tensor([7] * 5 + [9] * 3, dtype=torch.uint8, device="cuda")
        self.assertTrue(torch.equal(output[0], expected))
        self.assertTrue(torch.equal(output[4], expected))
        self.assertEqual(torch.count_nonzero(output[1:4]).item(), 0)
        self.assertEqual(torch.count_nonzero(output[5:]).item(), 0)

    def test_indexer_owner_write_matches_full_buffer(self):
        from types import SimpleNamespace

        from sglang.kernels.ops.attention.dsa.index_buf_accessor import SetKAndS

        cp_size = 4
        page_size = 64
        pages_per_rank = 2
        loc = torch.arange(cp_size * pages_per_rank, dtype=torch.int64, device="cuda")
        loc = loc * page_size + 3
        index_k = (
            torch.arange(loc.numel() * 128, dtype=torch.float32, device="cuda")
            .remainder_(31)
            .view(loc.numel(), 128)
            .to(torch.float8_e4m3fn)
        )
        index_k_scale = torch.arange(
            1, loc.numel() + 1, dtype=torch.float32, device="cuda"
        )
        pool = SimpleNamespace(page_size=page_size)
        full = torch.zeros(
            cp_size * pages_per_rank,
            page_size * 132,
            dtype=torch.uint8,
            device="cuda",
        )

        SetKAndS.execute(
            pool=pool,
            buf=full,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )
        for rank in range(cp_size):
            owner = torch.zeros_like(full[rank::cp_size])
            SetKAndS.execute(
                pool=pool,
                buf=owner,
                loc=loc,
                index_k=index_k,
                index_k_scale=index_k_scale,
                owner_rank=rank,
                owner_size=cp_size,
            )
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(owner, full[rank::cp_size]))


if __name__ == "__main__":
    unittest.main()
