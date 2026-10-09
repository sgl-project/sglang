"""CUDA graph executable reuse across capture sizes and rejected updates."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_executor.runner_backend import cuda_graph_dedup_mixin as dedup
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestCudaGraphDedup(CustomTestCase):
    @staticmethod
    def capture(size, *, extra_node=False):
        inputs = torch.zeros(size, device="cuda")
        outputs = torch.empty_like(inputs)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            torch.add(inputs, 1, out=outputs)
            if extra_node:
                outputs.mul_(2)
        return graph, inputs, outputs

    def test_grid_sizes_share_one_executable(self):
        registry = dedup.DedupedCudaGraphRegistry()
        self.addCleanup(registry.close)
        captures = [self.capture(size) for size in (4096, 8192)]
        self.assertEqual(
            dedup.graph_signature(captures[0][0].raw_cuda_graph()),
            dedup.graph_signature(captures[1][0].raw_cuda_graph()),
        )
        with patch.object(
            registry, "instantiate", wraps=registry.instantiate
        ) as instantiate:
            graphs = [registry.register(capture[0]) for capture in captures]
        self.assertEqual(instantiate.call_count, 1)
        self.assertEqual(registry.stats(), (2, 1))
        self.assertEqual(graphs[0].group.current_raw_graph, graphs[1].raw_graph)
        registry.seal()
        registry.seal()
        for value in (3, 7):
            for graph, (_, inputs, outputs) in zip(graphs, captures):
                inputs.fill_(value)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(
                    torch.equal(outputs, torch.full_like(outputs, value + 1))
                )

    def test_rejected_registration_preserves_live_executable(self):
        registry = dedup.DedupedCudaGraphRegistry()
        self.addCleanup(registry.close)
        original, inputs, outputs = self.capture(4096)
        incompatible, _, _ = self.capture(4096, extra_node=True)
        self.addCleanup(incompatible.reset)
        graph = registry.register(original)
        signature = dedup.graph_signature(graph.raw_graph)
        with (
            patch.object(dedup, "graph_signature", return_value=signature),
            self.assertRaisesRegex(AssertionError, "register update failed"),
        ):
            registry.register(incompatible)
        self.assertEqual(registry.stats(), (1, 1))
        self.assertEqual(graph.group.current_raw_graph, graph.raw_graph)
        inputs.fill_(11)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(outputs, torch.full_like(outputs, 12)))


if __name__ == "__main__":
    unittest.main()
