"""Regression tests for scatter prefix offsets and padded expert indices."""

import unittest

import torch
import triton

from sglang.kernels.ops.moe.ep_moe_kernels import _fwd_kernel_ep_scatter_1, ep_scatter
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "ep_scatter needs CUDA")
class TestEpScatter(CustomTestCase):
    def _check_prefix(self, counts, padded, block_e, num_warps, use_graph=False):
        starts = padded.cumsum(0) - padded
        total = padded.sum().item()
        # Build an independent reference for live rows and alignment/slack rows.
        expected = torch.full((total,), -1, dtype=torch.int32)
        for expert, (start, valid) in enumerate(zip(starts.tolist(), counts.tolist())):
            expected[start : start + valid] = expert

        counts_gpu, padded_gpu = counts.cuda(), padded.cuda()
        guard, poison = 128, 0x3F3F3F3F
        prefix_storage = torch.empty(
            counts.numel() + 2 * guard, device="cuda", dtype=torch.int32
        )
        index_storage = torch.empty(total + 2 * guard, device="cuda", dtype=torch.int32)

        def launch():
            # Poison the offsets every iteration, including inside graph replay.
            prefix_storage.fill_(poison)
            index_storage.fill_(poison)
            _fwd_kernel_ep_scatter_1[(counts.numel(),)](
                padded_gpu,
                counts_gpu,
                prefix_storage[guard:-guard],
                index_storage[guard:-guard],
                num_experts=counts.numel(),
                BLOCK_E=block_e,
                BLOCK_EXPERT_NUM=triton.next_power_of_2(counts.numel()),
                num_warps=num_warps,
            )

        launch()
        if use_graph:
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            run = graph.replay
        else:
            run = launch

        expected_prefix = torch.cat(
            [torch.full((guard,), poison), starts, torch.full((guard,), poison)]
        ).to(torch.int32)
        expected_indices = torch.cat(
            [torch.full((guard,), poison), expected, torch.full((guard,), poison)]
        ).to(torch.int32)
        for _ in range(10):
            run()
            torch.testing.assert_close(prefix_storage.cpu(), expected_prefix)
            torch.testing.assert_close(index_storage.cpu(), expected_indices)

    def test_prefix_and_padding(self):
        for experts in (1, 4, 64, 256, 288, 289):
            generator = torch.Generator().manual_seed(experts)
            cases = (
                torch.zeros(experts, dtype=torch.int32),
                torch.ones(experts, dtype=torch.int32),
                torch.randint(
                    0, 258, (experts,), generator=generator, dtype=torch.int32
                ),
            )
            for block_e in (64, 128):
                for num_warps in (4, 8):
                    for counts in cases:
                        with self.subTest(
                            experts=experts,
                            block_e=block_e,
                            num_warps=num_warps,
                            all_zero=not counts.any().item(),
                        ):
                            padded = triton.cdiv(counts, block_e) * block_e
                            self._check_prefix(counts, padded, block_e, num_warps)
                            # The compact layout puts graph-static slack in the
                            # last expert, even if that expert has no live rows.
                            padded = padded.clone()
                            padded[-1] += 2 * block_e
                            self._check_prefix(counts, padded, block_e, num_warps)

    def test_prefix_tile_boundaries(self):
        for block_e in (64, 128):
            counts = torch.tensor(
                [0, 1, block_e - 1, block_e, block_e + 1], dtype=torch.int32
            )
            padded = triton.cdiv(counts, block_e) * block_e
            with self.subTest(block_e=block_e):
                self._check_prefix(counts, padded, block_e, num_warps=8)

    def test_prefix_cuda_graph(self):
        block_e = 128
        counts = torch.arange(289, dtype=torch.int32) % 17
        padded = triton.cdiv(counts, block_e) * block_e
        padded[-1] += block_e
        self._check_prefix(counts, padded, block_e, num_warps=8, use_graph=True)

    def test_scatter_consumes_prefix(self):
        # Exercise experts across warp boundaries and non-power-of-two counts.
        experts, hidden, block_e = 289, 128, 128
        routes = torch.tensor(
            [[0, 32, 288], [31, 64, -1], [63, 256, 288], [255, 289, -1]],
            dtype=torch.int64,
        )
        valid_routes = routes[(routes >= 0) & (routes < experts)]
        counts = torch.bincount(valid_routes, minlength=experts).to(torch.int32)
        padded = triton.cdiv(counts, block_e) * block_e
        padded[-1] += block_e
        starts = padded.cumsum(0) - padded
        total = padded.sum().item()
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                x = torch.arange(routes.shape[0] * hidden).reshape(-1, hidden) % 16
                x = x.to(device="cuda", dtype=dtype)
                scales = torch.arange(1, routes.shape[0] + 1, device="cuda").float()
                scales = scales[:, None]
                output = torch.empty((total, hidden), device="cuda", dtype=dtype)
                output_scales = torch.empty((total, 1), device="cuda")
                indices = torch.empty(total, device="cuda", dtype=torch.int32)
                output_index = torch.empty_like(
                    routes, device="cuda", dtype=torch.int32
                )
                prefix = torch.full(
                    (experts,), 0x3F3F3F3F, device="cuda", dtype=torch.int32
                )
                ep_scatter(
                    x,
                    scales if dtype == torch.float8_e4m3fn else None,
                    routes.cuda(),
                    padded.cuda(),
                    counts.cuda(),
                    prefix,
                    output,
                    output_scales,
                    indices,
                    output_index,
                    expert_alignment=block_e,
                )
                destinations = output_index.cpu()
                # The second kernel advances the prefix entries as row cursors.
                torch.testing.assert_close(
                    prefix.cpu(), (starts + counts).to(torch.int32)
                )
                seen = set()
                for token, row in enumerate(routes.tolist()):
                    for slot, expert in enumerate(row):
                        dest = destinations[token, slot].item()
                        if not 0 <= expert < experts:
                            self.assertEqual(dest, -1)
                            continue
                        start = starts[expert].item()
                        self.assertTrue(start <= dest < start + counts[expert].item())
                        self.assertNotIn(dest, seen)
                        seen.add(dest)
                        self.assertEqual(indices[dest].item(), expert)
                        torch.testing.assert_close(
                            output[dest].float(), x[token].float()
                        )
                        if dtype == torch.float8_e4m3fn:
                            torch.testing.assert_close(
                                output_scales[dest], scales[token]
                            )


if __name__ == "__main__":
    unittest.main()
