import unittest

import torch

from sglang.kernels.ops.kvcache.batched_store_cache import batched_store_cache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class TestBatchedStoreCache(CustomTestCase):
    def _run(self, dtype, prefix, graph=False, tokens=21):
        # Capacity-based source layer strides and padded cache rows must not
        # leak into adjacent layers, slots, or uncommitted verify columns.
        layers, heads, dim, capacity = 3, 2, 96, tokens + 19
        k = torch.randn(layers, tokens + 9, heads, dim, device="cuda", dtype=dtype)[
            :, :tokens
        ]
        v = torch.randn_like(k)
        backing_k = [
            torch.full((capacity, heads * dim + 32), -5, device="cuda", dtype=dtype)
            for _ in range(layers)
        ]
        backing_v = [
            torch.full((capacity, heads * dim + 64), -7, device="cuda", dtype=dtype)
            for _ in range(layers)
        ]
        kc = [b[:, : heads * dim].view(capacity, heads, dim) for b in backing_k]
        vc = [b[:, : heads * dim].view(capacity, heads, dim) for b in backing_v]
        locs = [
            (torch.randperm(capacity - 2, device="cuda")[:tokens] + 1).to(
                torch.int32 if i % 2 else torch.int64
            )
            for i in range(layers)
        ]
        for loc in locs:
            loc[0] = -1
            loc[1] = 0
        lens = (
            torch.tensor([7, 3, 0], device="cuda", dtype=torch.int32)
            if prefix
            else None
        )
        live = (
            torch.arange(tokens, device="cuda") % 7 < lens.repeat_interleave(7)
            if prefix
            else torch.ones(tokens, device="cuda", dtype=torch.bool)
        )

        def run():
            batched_store_cache(k, v, kc, vc, locs, lens, 7)

        run()
        if graph:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                run()
            # Replays must read updated inputs, locations, and commit lengths.
            k.add_(1)
            v.sub_(1)
            for loc in locs:
                loc[2] = capacity - 1
            lens.copy_(torch.tensor([2, 7, 0], device="cuda", dtype=torch.int32))
            live = torch.arange(tokens, device="cuda") % 7 < lens.repeat_interleave(7)
            for b in backing_k:
                b.fill_(-5)
            for b in backing_v:
                b.fill_(-7)
            g.replay()
        for i in range(layers):
            expected_k = torch.full_like(backing_k[i], -5)
            expected_v = torch.full_like(backing_v[i], -7)
            valid = live & (locs[i] >= 0)
            if not prefix:
                valid &= locs[i] != 0
            # Duplicate active physical slots are outside the store contract.
            selected = locs[i][valid].long()
            self.assertEqual(selected.unique().numel(), selected.numel())
            expected_k[selected, : heads * dim] = k[i, valid].flatten(1)
            expected_v[selected, : heads * dim] = v[i, valid].flatten(1)
            torch.testing.assert_close(backing_k[i], expected_k, rtol=0, atol=0)
            torch.testing.assert_close(backing_v[i], expected_v, rtol=0, atol=0)

    def test_layer_locations_and_padded_strides(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self._run(dtype, prefix=False)

    def test_only_accepted_prefix_is_written(self):
        self._run(torch.bfloat16, prefix=True)

    def test_large_non_tile_multiple(self):
        self._run(torch.bfloat16, prefix=False, tokens=8193)

    def test_graph_replay_reads_current_commit_and_locations(self):
        self._run(torch.bfloat16, prefix=True, graph=True)

    def test_source_and_cache_offsets_above_int32(self):
        k = torch.empty_strided(
            (2, 1, 1, 96),
            (2**31 + 96, 96, 96, 1),
            device="cuda",
            dtype=torch.float16,
        )
        k[0].fill_(3)
        k[1].fill_(7)
        v = torch.ones((2, 1, 1, 96), device="cuda", dtype=k.dtype)
        wide_cache = torch.empty_strided(
            (3, 1, 96), (2**30 + 96, 96, 1), device="cuda", dtype=k.dtype
        )
        wide_cache.zero_()
        kc = (wide_cache, torch.zeros((3, 1, 96), device="cuda", dtype=k.dtype))
        vc = tuple(
            torch.zeros((3, 1, 96), device="cuda", dtype=k.dtype) for _ in range(2)
        )
        loc = torch.full((1,), 2, device="cuda", dtype=torch.int32)
        batched_store_cache(k, v, kc, vc, (loc, loc))
        for i in range(2):
            torch.testing.assert_close(kc[i][2], k[i, 0], rtol=0, atol=0)
            torch.testing.assert_close(vc[i][2], v[i, 0], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
