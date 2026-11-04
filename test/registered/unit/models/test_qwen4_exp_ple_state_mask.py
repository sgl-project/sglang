"""Qwen4-Exp PLE state-index mask: a multiply gives torch.where's int64 values, and
records no memset node in a captured CUDA graph (GPU part, skipped without CUDA)."""

import unittest

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.models.qwen4_exp import _mask_state_indices
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _where_mask(valid, state_indices):
    return torch.where(valid, state_indices, torch.zeros_like(state_indices))


class TestPleStateMaskValues(CustomTestCase):
    def test_default_off_keeps_original_where(self):
        self.assertFalse(envs.SGLANG_ENABLE_PLE_STATE_INDEX_MASK.default)
        state = torch.tensor([0, 7, 2**62], dtype=torch.int64)
        valid = torch.tensor([True, False, True])
        with envs.SGLANG_ENABLE_PLE_STATE_INDEX_MASK.override(False):
            self.assertTrue(torch.equal(_mask_state_indices(valid, state), _where_mask(valid, state)))

    def test_equals_where_with_zeros(self):
        g = torch.Generator().manual_seed(0)
        for bs in (1, 2, 3, 8, 32, 96, 1024):
            for high in (2, 900, 2**31, 2**62):
                state = torch.randint(0, high, (bs,), dtype=torch.long, generator=g)
                valid = torch.rand(bs, generator=g) > 0.3
                with envs.SGLANG_ENABLE_PLE_STATE_INDEX_MASK.override(True):
                    got = _mask_state_indices(valid, state)
                want = _where_mask(valid, state)
                self.assertEqual(got.dtype, torch.long)
                self.assertTrue(torch.equal(got, want), (bs, high))


def _graph_node_types(graph):
    from cuda.bindings import driver as cu

    raw = cu.CUgraph(graph.raw_cuda_graph())
    _, _, n = cu.cuGraphGetNodes(raw, 0)
    _, nodes, n = cu.cuGraphGetNodes(raw, n)
    return [cu.cuGraphNodeGetType(node)[1] for node in nodes[:n]]


@pytest.mark.gpu
@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestPleStateMaskGraph(CustomTestCase):
    def _capture(self, fn, valid, state):
        out = {}
        fn(valid, state)  # warm up outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            out["o"] = fn(valid, state)
        graph.replay()
        torch.cuda.synchronize()
        return graph, out["o"]

    def test_no_memset_node_and_equal_values(self):
        from cuda.bindings import driver as cu

        memset = cu.CUgraphNodeType.CU_GRAPH_NODE_TYPE_MEMSET
        g = torch.Generator(device="cuda").manual_seed(1)
        for bs in (1, 3, 8, 32, 96):
            state = torch.randint(
                0, 900, (bs,), device="cuda", dtype=torch.long, generator=g
            )
            valid = torch.rand(bs, device="cuda", generator=g) > 0.3
            with envs.SGLANG_ENABLE_PLE_STATE_INDEX_MASK.override(True):
                graph, got = self._capture(_mask_state_indices, valid, state)
            # Positive control: the where() form records the memset this avoids.
            control, want = self._capture(_where_mask, valid, state)
            self.assertTrue(torch.equal(got, want), bs)
            self.assertEqual(_graph_node_types(graph).count(memset), 0, bs)
            self.assertGreaterEqual(_graph_node_types(control).count(memset), 1, bs)


if __name__ == "__main__":
    unittest.main()
