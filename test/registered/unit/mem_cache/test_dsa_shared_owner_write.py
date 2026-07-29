import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

_HAS_CUDA = torch.cuda.is_available()

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(_HAS_CUDA, "Triton kernels require CUDA")
class TestDSASharedOwnerWrite(unittest.TestCase):
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
