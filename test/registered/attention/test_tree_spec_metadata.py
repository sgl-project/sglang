"""Stable suffix-slot construction from packed verify masks."""

import unittest

import torch

from sglang.kernels.ops.attention.metadata import prepare_tree_spec_metadata
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestTreeSpecMetadata(CustomTestCase):
    def _case(self, batch, width, mask_dtype, pool_dtype, *, qlen_only: bool):
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
        # Match graph replay: the final request may be padding, whose mask
        # indices are redirected to element zero of the packed mask.
        live_rows = (batch - 1) * width if batch > 1 else width
        live_reqs = live_rows // width
        mask_size = (
            live_rows * width if qlen_only else batch * (capacity + width) * width
        )
        mask = torch.empty(mask_size, dtype=mask_dtype, device="cuda")
        table = torch.empty((batch * width, width), dtype=pool_dtype, device="cuda")
        lengths = torch.empty(batch * width, dtype=torch.int32, device="cuda")
        generation = 0

        def update_inputs():
            nonlocal generation
            generation += 1
            reqs = torch.randperm(batch + 2)[:batch]
            lens = torch.randint(0, capacity + 1, (batch,))
            lens[0] = 0
            chunks, trees, expected, expected_lengths = [], [], [], []
            for b, length in enumerate(lens.tolist()):
                tree = torch.rand(width, width) > 0.5
                tree[0].fill_(generation % 2 == 0)
                tree[-1].fill_(True)
                trees.append(tree)
                chunks.append(torch.cat((torch.ones(width, length).bool(), tree), 1))
            packed = torch.cat(
                [tree.flatten() for tree in trees[:live_reqs]]
                if qlen_only
                else [chunk.flatten() for chunk in chunks]
            )
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
            if qlen_only:
                # Sentinel values verify QLEN_ONLY never reads cu_seqlens.
                cu_seqlens.fill_(-999999)
            else:
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
                qlen_only=qlen_only,
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
        for qlen_only in (False, True):
            for batch, width, mask_dtype, pool_dtype in (
                (1, 1, torch.bool, torch.int32),
                (3, 17, torch.uint8, torch.int64),
                (5, 65, torch.bool, torch.int32),
                (3, 128, torch.bool, torch.int32),
                (2, 257, torch.uint8, torch.int64),
                (1, 1024, torch.bool, torch.int32),
            ):
                with self.subTest(qlen_only=qlen_only, batch=batch, width=width):
                    self._case(
                        batch, width, mask_dtype, pool_dtype, qlen_only=qlen_only
                    )

    def test_qlen_only_rejects_undersized_mask(self):
        width = 8
        pool = torch.zeros((2, 32), dtype=torch.int32, device="cuda")
        req_indices = torch.zeros(2, dtype=torch.int64, device="cuda")
        seq_lens = torch.zeros(2, dtype=torch.int64, device="cuda")
        cu_seqlens = torch.zeros(3, dtype=torch.int32, device="cuda")
        table = torch.empty((2 * width, width), dtype=torch.int32, device="cuda")
        lengths = torch.empty(2 * width, dtype=torch.int32, device="cuda")
        undersized = torch.ones(width * width - 1, dtype=torch.bool, device="cuda")
        with self.assertRaises(AssertionError):
            prepare_tree_spec_metadata(
                pool,
                req_indices,
                seq_lens,
                cu_seqlens,
                undersized,
                table,
                lengths,
                width,
                qlen_only=True,
            )


if __name__ == "__main__":
    unittest.main()
