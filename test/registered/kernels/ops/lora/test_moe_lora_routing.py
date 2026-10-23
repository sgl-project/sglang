"""Numerical routing and CUDA-graph workspace regressions."""

import unittest
from collections import defaultdict

import torch

from sglang.kernels.ops.lora.common import routing as routing_module
from sglang.kernels.ops.lora.common.routing import build_route
from sglang.srt.lora.workspace import LoraWorkspace
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _tensors(route):
    return route.sorted_pair_ids, route.block_bucket_ids, route.num_pairs_post_padded


class TestMoeLoraRouting(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = torch.device("cuda", torch.cuda.current_device())

    @staticmethod
    def _inputs(rows, width, groups_per_slot, max_loras, dtype, shift=0):
        slots = (torch.arange(rows // width, dtype=dtype) + shift) % max_loras
        slots[::7] = -1
        slots[1::11] = max_loras
        if width == 1:
            return slots, None
        groups = (
            (torch.arange(rows, dtype=dtype) * 37 + shift) % groups_per_slot
        ).reshape(-1, width)
        groups[shift % 11 :: 11, -1] = -2
        groups[2 + shift :: 13, 0] = 37 if groups_per_slot == 1 else groups_per_slot
        slots[3] = max_loras - 1
        groups[3, 0] = groups_per_slot - 1
        return slots, groups

    @staticmethod
    def _expected(slots, groups, groups_per_slot, max_loras):
        # Independent host oracle: never call the production bucket-ID kernel.
        expected = defaultdict(list)
        width = 1 if groups is None else groups.shape[1]
        for token, slot in enumerate(slots.tolist()):
            for column in range(width):
                group = 0 if groups is None else int(groups[token, column])
                live = (
                    0 <= slot < max_loras
                    and group >= 0
                    and (groups_per_slot == 1 or group < groups_per_slot)
                )
                bucket = slot * groups_per_slot if live else -1
                if live and groups_per_slot != 1:
                    bucket += group
                expected[bucket].append(token * width + column)
        return expected

    def _assert_layout(self, result, expected, block_size, sentinel_last=False):
        sorted_ids, block_ids, total = result
        self.assertTrue(all(tensor.dtype == torch.int32 for tensor in result))
        rows = sum(map(len, expected.values()))
        # Large routing places the invalid bucket last; small/JIT place it first.
        order = sorted(
            expected.items(),
            key=lambda item: (sentinel_last and item[0] < 0, item[0]),
        )
        labels = [
            bucket
            for bucket, pairs in order
            for _ in range((len(pairs) + block_size - 1) // block_size)
        ]
        padded = int(total.item())
        self.assertEqual(padded, len(labels) * block_size)
        self.assertLessEqual(padded, sorted_ids.numel())
        self.assertEqual(block_ids[: len(labels)].cpu().tolist(), labels)
        values, offset = sorted_ids[:padded].cpu().tolist(), 0
        for _, wanted in order:
            size = ((len(wanted) + block_size - 1) // block_size) * block_size
            bucket_rows = values[offset : offset + size]
            # Atomic placement may permute rows, but must not lose or duplicate any.
            self.assertEqual(
                sorted(pair for pair in bucket_rows if pair != rows), wanted
            )
            self.assertEqual(bucket_rows.count(rows), size - len(wanted))
            offset += size

    def _parity_builders(self, slots, groups, groups_per_slot, max_loras, block_size):
        small_workspace, cuda_workspace = LoraWorkspace(), LoraWorkspace()
        for workspace in (small_workspace, cuda_workspace):
            workspace.begin_forward(graph_mode=True)
        buckets = cuda_workspace.tensor(
            "parity:bucket_ids",
            (slots.numel(), 1) if groups is None else groups.shape,
            dtype=torch.int32,
            device=self.device,
        )

        def small():
            return _tensors(
                routing_module._build_small_route(
                    slots,
                    groups,
                    groups_per_slot=groups_per_slot,
                    max_loras=max_loras,
                    block_size=block_size,
                    workspace=small_workspace,
                    tensor_prefix="parity:small",
                )
            )

        def cuda():
            routing_module._build_route_bucket_ids(
                slots,
                groups,
                groups_per_slot=groups_per_slot,
                max_loras=max_loras,
                out=buckets,
            )
            return routing_module._align_block_size_jit(
                buckets,
                block_size,
                groups_per_slot * max_loras,
                scratch=lambda size: cuda_workspace.tensor(
                    "parity:scratch", (size,), dtype=torch.int32, device=self.device
                ),
            )

        return small, cuda

    @staticmethod
    def _capture(build):
        # Workspace storage is stream-local: warm on the stream used for capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            build()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result = build()
        return graph, result

    def test_small_route_matches_cuda_and_host(self):
        # Sparse/dense capacities, 4/8 warps, shared folding, int64 and wide bins.
        for rows, width, groups_per_slot, max_loras, block_size, dtype in (
            (1, 1, 1, 4, 16, torch.int32),
            (512, 8, 64, 4, 32, torch.int32),
            (513, 1, 1, 4, 128, torch.int64),
            (768, 1, 1, 4, 128, torch.int32),
            (768, 8, 1, 4, 63, torch.int64),
            (512, 8, 1024, 4, 16, torch.int32),
        ):
            with self.subTest(rows=rows, width=width, groups=groups_per_slot):
                slots, groups = self._inputs(
                    rows, width, groups_per_slot, max_loras, dtype
                )
                expected = self._expected(slots, groups, groups_per_slot, max_loras)
                for build in self._parity_builders(
                    slots.to(self.device),
                    None if groups is None else groups.to(self.device),
                    groups_per_slot,
                    max_loras,
                    block_size,
                ):
                    self._assert_layout(build(), expected, block_size)

    def test_small_and_cuda_graphs_replay_changed_inputs(self):
        rows, width, max_loras, block_size = 768, 8, 4, 128
        host_slots, host_groups = self._inputs(rows, width, 1, max_loras, torch.int64)
        slots, groups = host_slots.to(self.device), host_groups.to(self.device)
        captures = [
            self._capture(build)
            for build in self._parity_builders(slots, groups, 1, max_loras, block_size)
        ]
        for shift in (1, 2, 0):
            new_slots, new_groups = self._inputs(
                rows, width, 1, max_loras, torch.int64, shift
            )
            if shift == 2:
                new_slots.fill_(-1)
            slots.copy_(new_slots)
            groups.copy_(new_groups)
            expected = self._expected(new_slots, new_groups, 1, max_loras)
            for graph, result in captures:
                for tensor in result:
                    tensor.fill_(-7)
                graph.replay()
                self._assert_layout(result, expected, block_size)

    def test_large_route_without_workspace(self):
        rows = routing_module._LARGE_ROUTE_MIN_PAIRS
        slots, _ = self._inputs(rows, 1, 1, 4, torch.int32)
        expected = self._expected(slots, None, 1, 4)
        slots = slots.to(self.device)
        routes = [build_route(slots, max_loras=4, block_size=16) for _ in range(2)]
        for route in routes:
            self._assert_layout(_tensors(route), expected, 16, sentinel_last=True)
        self.assertNotEqual(
            routes[0].sorted_pair_ids.data_ptr(), routes[1].sorted_pair_ids.data_ptr()
        )

    def test_large_graph_resets_shared_and_per_expert_scratch(self):
        rows, width, experts, max_loras, block_size = 768, 8, 384, 32, 16
        host_slots, host_groups = self._inputs(
            rows, width, experts, max_loras, torch.int64
        )
        slots, groups = host_slots.to(self.device), host_groups.to(self.device)
        workspace = LoraWorkspace()
        workspace.begin_forward(graph_mode=True)

        def build():
            return [
                _tensors(
                    build_route(
                        slots,
                        group_ids=groups,
                        groups_per_slot=buckets,
                        max_loras=max_loras,
                        block_size=block_size,
                        workspace=workspace,
                        tensor_prefix="route:aligned",
                        capacity=2 * rows if buckets == 1 else None,
                    )
                )
                for buckets in (1, experts)
            ]

        graph, results = self._capture(build)
        counts = [
            value
            for key, value in workspace._graph_storage.items()
            if key[0].endswith(":counts")
        ]
        self.assertEqual(len(counts), 2)
        for shift in (1, 2, 0):
            new_slots, new_groups = self._inputs(
                rows, width, experts, max_loras, torch.int64, shift
            )
            if shift == 2:
                new_slots.fill_(-1)
            slots.copy_(new_slots)
            groups.copy_(new_groups)
            graph.replay()
            for buckets, result in zip((1, experts), results):
                expected = self._expected(new_slots, new_groups, buckets, max_loras)
                self._assert_layout(result, expected, block_size, sentinel_last=True)
            for count in counts:
                self.assertEqual(torch.count_nonzero(count).item(), 0)

    def test_parallel_workspace_joins_reused_graph_regions(self):
        workspace = LoraWorkspace()
        source = torch.full((64,), 3.0, device=self.device)
        side_output, output = torch.empty_like(source), torch.empty_like(source)

        def side():
            side_output.copy_(source).add_(1)

        def compute():
            return output.copy_(source).mul_(2)

        def build():
            # Consecutive layers reuse the same ready/done events.
            for _ in range(2):
                workspace.run_parallel(
                    name="shared_region", device=self.device, compute=compute, side=side
                )
            output.add_(side_output)

        graph, _ = self._capture(build)
        for value in (5, 11):
            source.fill_(value)
            graph.replay()
            torch.cuda.synchronize(self.device)
            torch.testing.assert_close(
                output, torch.full_like(output, 3 * value + 1), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
