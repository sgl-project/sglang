"""Regression test for the ViT CUDA-graph `sin_cos_ws` use-after-free (#29216).

The base ``ViTCudaGraphRunner`` used to share a single rotary cos/sin
workspace and regrow it on demand. Captured CUDA graphs bake views into
the workspace at capture time, so a later regrow freed the buffer that
earlier graphs still read on replay — silent corruption.

The fix keys the workspace by ``graph_key``, one exact-size allocation
per shape. These tests pin the invariants that prevent the UAF:

- ``sin_cos_ws`` is a dict (no single shared buffer to regrow).
- Each ``graph_key`` gets its own ``(cos, sin)`` tensors at exact
  ``(graph_key, head_dim)`` size.
- Repeat allocation for the same key overwrites in place — capture-time
  pointers stay live.
- Independent keys have distinct storage; later allocations never alias
  earlier ones.

The test runs on CPU; it exercises the runner's allocation bookkeeping
without entering ``torch.cuda.graph(...)`` capture, which is covered by
the e2e nightly suite ``test_vlms_vit_cuda_graph.py``.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.multimodal.vit_cuda_graph_runner import ViTCudaGraphRunner
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=5, stage="base-a", runner_config="1-gpu-small")


HEAD_DIM = 32


class _StubBlock(nn.Module):
    def forward(self, x, cu_seqlens, position_embeddings=None, output_ws=None):
        return x


def _make_runner(dtype=torch.float32, device="cpu"):
    vit = SimpleNamespace(
        device=torch.device(device),
        dtype=dtype,
        blocks=nn.ModuleList([_StubBlock()]),
    )
    return ViTCudaGraphRunner(vit)


class TestViTCudaGraphRunnerSinCosWs(CustomTestCase):
    def test_sin_cos_ws_is_per_key_dict(self):
        runner = _make_runner()
        self.assertIsInstance(runner.sin_cos_ws, dict)
        self.assertEqual(len(runner.sin_cos_ws), 0)

    def test_allocate_uses_exact_size_and_keys_dict(self):
        runner = _make_runner()
        for graph_key in (64, 256, 4096):
            cos_ws, sin_ws = runner._allocate_sin_cos_ws(graph_key, HEAD_DIM)
            self.assertEqual(cos_ws.shape, (graph_key, HEAD_DIM))
            self.assertEqual(sin_ws.shape, (graph_key, HEAD_DIM))
            self.assertEqual(cos_ws.dtype, runner.dtype)
            self.assertEqual(sin_ws.dtype, runner.dtype)
            self.assertIs(runner.sin_cos_ws[graph_key][0], cos_ws)
            self.assertIs(runner.sin_cos_ws[graph_key][1], sin_ws)
        self.assertEqual(set(runner.sin_cos_ws), {64, 256, 4096})

    def test_independent_keys_have_distinct_storage(self):
        runner = _make_runner()
        cos_a, sin_a = runner._allocate_sin_cos_ws(64, HEAD_DIM)
        cos_b, sin_b = runner._allocate_sin_cos_ws(256, HEAD_DIM)
        self.assertNotEqual(cos_a.data_ptr(), cos_b.data_ptr())
        self.assertNotEqual(sin_a.data_ptr(), sin_b.data_ptr())
        self.assertFalse(
            cos_a.untyped_storage().data_ptr() == cos_b.untyped_storage().data_ptr()
        )

    def test_later_allocations_do_not_invalidate_earlier_buffers(self):
        """The UAF regression: a later, larger allocation must not move or
        free the storage of an earlier key. The captured graph reads the
        original ``data_ptr``; if it moves, the graph reads freed memory.
        """
        runner = _make_runner()
        cos_small, sin_small = runner._allocate_sin_cos_ws(64, HEAD_DIM)
        cos_small_addr = cos_small.data_ptr()
        sin_small_addr = sin_small.data_ptr()

        for graph_key in (256, 1024, 4096, 8192):
            runner._allocate_sin_cos_ws(graph_key, HEAD_DIM)

        self.assertIs(runner.sin_cos_ws[64][0], cos_small)
        self.assertIs(runner.sin_cos_ws[64][1], sin_small)
        self.assertEqual(runner.sin_cos_ws[64][0].data_ptr(), cos_small_addr)
        self.assertEqual(runner.sin_cos_ws[64][1].data_ptr(), sin_small_addr)


if __name__ == "__main__":
    unittest.main()
