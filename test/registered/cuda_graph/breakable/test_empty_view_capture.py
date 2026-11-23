"""Empty eager-break arguments must not pin their nonempty projection bases."""

import gc
import unittest
import weakref

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
    BreakableCUDAGraph,
    BreakableCUDAGraphCapture,
    _weak_ref_if_tensor,
    eager_on_graph,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")


class TestEmptyViewCapture(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")

    def test_nested_empty_views_release_base_and_preserve_metadata(self):
        base = torch.empty((19, 16, 256), dtype=torch.bfloat16, device="cuda")
        view = base[..., 256:]
        transposed = view.transpose(0, 1)
        metadata = [(item.shape, item.stride()) for item in (view, transposed)]
        base_ref = weakref.ref(base)
        result = _weak_ref_if_tensor(([view], (transposed,)))
        del view, transposed, base
        gc.collect()
        self.assertIsNone(base_ref())
        for item, (shape, stride) in zip((result[0][0], result[1][0]), metadata):
            self.assertEqual(item.shape, shape)
            self.assertEqual(item.stride(), stride)
            self.assertEqual(item.dtype, torch.bfloat16)
            self.assertEqual(item.untyped_storage().nbytes(), 0)
            self.assertEqual(item.device.type, "cuda")

    def test_empty_cpu_argument_keeps_identity(self):
        value = torch.empty((3, 0))
        self.assertIs(_weak_ref_if_tensor(value), value)

    def test_multiple_buckets_replay_after_projection_storage_is_reused(self):
        # These dimensions reproduce NoPE MLA's zero-width q_pe/k_pe views.
        # Both bases are graph temporaries and must be reusable after capture.
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()
        x = torch.zeros((512, 4096), device="cuda", dtype=torch.bfloat16)
        y = torch.empty((512,), device="cuda", dtype=torch.bfloat16)
        graphs = []
        empty_storage_bytes = []

        @eager_on_graph
        def attention(q, q_pe, k_pe, output):
            self.assertEqual(q_pe.shape[-1], 0)
            self.assertEqual(k_pe.shape[-1], 0)
            empty_storage_bytes.append(
                q_pe.untyped_storage().nbytes() + k_pe.untyped_storage().nbytes()
            )
            output.copy_(q[:, 0, 0])

        def forward(rows):
            q = (x[:rows] + 1).view(rows, 16, 256)
            latent = x[:rows, :2048] + 2
            q_pe = q[..., 256:]
            k_pe = latent[:, 2048:].unsqueeze(1)
            attention(q, q_pe, k_pe, y[:rows])
            # Reuse temporary storage in a following segment and later buckets.
            temporary = x[:rows] * 3
            y[:rows].add_(temporary[:, 0])

        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward(512)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        for rows in (512, 384, 256, 128):
            graph = BreakableCUDAGraph()
            with BreakableCUDAGraphCapture(graph, pool=pool, stream=stream):
                forward(rows)
            graphs.append((rows, graph))
        gc.collect()
        torch.cuda.synchronize()
        # Shared-pool inactive blocks are reserved, but should not stay active
        # solely through the empty bridge arguments (old path retains 15 MiB).
        self.assertLess(torch.cuda.memory_allocated() - allocated, 1024 * 1024)
        empty_storage_bytes.clear()
        for value, index in ((1, 0), (-2, 3), (4, 1), (0, 2), (3, 0)):
            rows, graph = graphs[index]
            x.fill_(value)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                y[:rows], torch.full_like(y[:rows], value * 4 + 1), rtol=0, atol=0
            )
        self.assertTrue(empty_storage_bytes)
        self.assertTrue(all(size == 0 for size in empty_storage_bytes))


if __name__ == "__main__":
    unittest.main()
