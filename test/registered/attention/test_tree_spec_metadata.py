"""Stable suffix-slot construction from packed verify masks."""

import unittest

import torch

from sglang.kernels.ops.attention.metadata import prepare_tree_spec_metadata
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTreeSpecMetadata(CustomTestCase):
    def _case(self, batch, width, mask_dtype, pool_dtype):
        torch.manual_seed(42)
        capacity = 130
        pool = (
            torch.randperm((batch + 2) * (capacity + width))
            .reshape(batch + 2, capacity + width)
            .to(pool_dtype)
        )
        gpu_pool = pool.cuda()
        req_indices = torch.empty(batch, dtype=torch.int64, device="cuda")
        seq_lens = torch.empty(batch, dtype=torch.int64, device="cuda")
        cu_seqlens = torch.empty(batch + 1, dtype=torch.int32, device="cuda")
        mask = torch.empty(
            batch * (capacity + width) * width, dtype=mask_dtype, device="cuda"
        )
        table = torch.empty((batch * width, width), dtype=pool_dtype, device="cuda")
        lengths = torch.empty(batch * width, dtype=torch.int32, device="cuda")
        # Match graph replay: the final request may be padding, whose mask
        # indices are redirected to element zero of the packed mask.
        live_rows = (batch - 1) * width if batch > 1 else width
        generation = 0

        def update_inputs():
            nonlocal generation
            generation += 1
            reqs = torch.randperm(batch + 2)[:batch]
            lens = torch.randint(0, capacity + 1, (batch,))
            lens[0] = 0
            chunks, expected, expected_lengths = [], [], []
            for b, length in enumerate(lens.tolist()):
                tree = torch.rand(width, width) > 0.5
                tree[0].fill_(generation % 2 == 0)
                tree[-1].fill_(True)
                chunks.append(torch.cat((torch.ones(width, length).bool(), tree), 1))
            packed = torch.cat([chunk.flatten() for chunk in chunks])
            for b, chunk in enumerate(chunks):
                length = lens[b].item()
                slots = pool[reqs[b], length : length + width]
                for t in range(width):
                    visible = chunk[t, length:]
                    if b * width + t >= live_rows:
                        visible = packed[0].expand(width)
                    expected.append(torch.cat((slots[visible], slots[~visible])))
                    expected_lengths.append(int(visible.sum()))
            req_indices.copy_(reqs)
            seq_lens.copy_(lens)
            cu_seqlens[0] = 0
            cu_seqlens[1:].copy_(lens.cumsum(0))
            mask[: packed.numel()].copy_(packed)
            return torch.stack(expected), torch.tensor(
                expected_lengths, dtype=torch.int32
            )

        def run():
            prepare_tree_spec_metadata(
                gpu_pool,
                req_indices,
                seq_lens,
                cu_seqlens,
                mask,
                table,
                lengths,
                live_rows,
            )

        expected_table, expected_lengths = update_inputs()
        run()
        torch.testing.assert_close(table.cpu(), expected_table, rtol=0, atol=0)
        torch.testing.assert_close(lengths.cpu(), expected_lengths, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for _ in range(2):
            expected_table, expected_lengths = update_inputs()
            graph.replay()
            torch.testing.assert_close(table.cpu(), expected_table, rtol=0, atol=0)
            torch.testing.assert_close(lengths.cpu(), expected_lengths, rtol=0, atol=0)

    def test_shapes_and_replay(self):
        for batch, width, mask_dtype, pool_dtype in (
            (1, 1, torch.bool, torch.int32),
            (3, 17, torch.uint8, torch.int64),
            (5, 65, torch.bool, torch.int32),
            (3, 128, torch.bool, torch.int32),
            (2, 257, torch.uint8, torch.int64),
            (1, 1024, torch.bool, torch.int32),
        ):
            with self.subTest(batch=batch, width=width):
                self._case(batch, width, mask_dtype, pool_dtype)


if __name__ == "__main__":
    unittest.main()
