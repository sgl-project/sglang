"""Manual GPU checks for the experimental patch, without model serving."""

import os
import unittest

import decode_l2pf as patch
import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestPrefetchKernels(unittest.TestCase):
    def setUp(self):
        self.old_state = patch._st.copy()
        patch._st.update(
            side=torch.cuda.Stream(),
            sink=torch.zeros(4096, dtype=torch.int32, device="cuda"),
            join=None,
            next=None,
        )
        self.old_kind = patch.KIND

    def tearDown(self):
        torch.cuda.synchronize()
        patch.KIND = self.old_kind
        patch._st.clear()
        patch._st.update(self.old_state)

    def test_weight_views_are_bounded_and_deduplicated(self):
        module = torch.nn.Module()
        weight = torch.arange(65536, dtype=torch.int32, device="cuda")
        module.register_buffer("weight", weight)
        module.register_buffer("alias", weight)
        module.register_buffer("noncontiguous", weight[::2])
        module.register_buffer("small", weight[:16])
        module.register_buffer("cpu_weight", torch.zeros(65536, dtype=torch.int32))
        views = patch._chunks([module, module], 128 * 1024)
        self.assertEqual(len(views), 1)
        self.assertEqual(views[0].data_ptr(), weight.data_ptr())
        self.assertEqual(views[0].numel() * 4, 128 * 1024)

    def _exercise(self, kind):
        if kind == "bulk" and torch.cuda.get_device_capability()[0] < 9:
            self.skipTest("bulk prefetch requires SM90+")
        patch.KIND = kind
        # A partial final 64 KiB chunk exercises the bulk tail handling.
        weight = torch.arange(65536 + 32, dtype=torch.int32, device="cuda")
        before = weight.clone()
        torch.cuda.synchronize()
        side = patch._st["side"]
        with torch.cuda.stream(side):
            patch._prefetch([weight])
        side.synchronize()
        self.assertTrue(torch.equal(weight, before))
        graph = torch.cuda.CUDAGraph()
        # Capture fork and deferred join, including repeated graph replay.
        with torch.cuda.graph(graph):
            fork = torch.cuda.Event()
            fork.record()
            with torch.cuda.stream(side):
                side.wait_event(fork)
                patch._prefetch([weight])
                join = torch.cuda.Event()
                join.record()
            patch._st["join"] = join
            patch._drain()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        self.assertIsNone(patch._st["join"])
        self.assertTrue(torch.equal(weight, before))

    def test_bulk_eager_and_graph(self):
        self._exercise("bulk")

    def test_load_eager_and_graph(self):
        self._exercise("ld")


if __name__ == "__main__":
    if os.environ.get("SGLANG_DECODE_L2PF", "0") != "0":
        raise RuntimeError("Run kernel checks with SGLANG_DECODE_L2PF=0")
    unittest.main()
