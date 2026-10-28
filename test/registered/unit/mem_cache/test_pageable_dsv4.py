"""Pageable DSV4 whole-page and partial-token transfers on real CUDA buffers."""

import gc
import os
import unittest
import weakref
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host import common
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPageableDSV4(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.registration = patch.object(
            common, "_cuda_host_register", side_effect=AssertionError("registered")
        )
        self.registration.start()
        self.addCleanup(self.registration.stop)
        self.src = [
            torch.randint(0, 256, (32, 37440), dtype=torch.uint8, device="cuda")
            for _ in range(2)
        ]
        self.host = DeepSeekV4PagedHostPool(
            "test", self.src, 37440, 32, 256, layout="page_first_direct"
        )
        self.addCleanup(self.host.destroy)
        self.host.kv_buffer.fill_(73)

    @staticmethod
    def copy_tokens(dst, src, dest, source):
        for d, s in zip(dest.tolist(), source.tolist()):
            dp, ds = divmod(d, 64)
            sp, ss = divmod(s, 64)
            for base, width in ((0, 576), (36864, 8)):
                dst[dp, base + ds * width : base + (ds + 1) * width] = src[
                    sp, base + ss * width : base + (ss + 1) * width
                ]

    def test_partial_roundtrip_preserves_neighbors_scales_and_padding(self):
        # Cross the staging chunk boundary and select slots from different pages.
        source = torch.randperm(2048)[:1025]
        dest = torch.randperm(2048)[:1025]
        originals = [buf.cpu() for buf in self.src]
        expected = self.host.kv_buffer.clone()
        for layer in range(2):
            self.copy_tokens(expected[:, layer, 0], originals[layer], dest, source)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.host.backup_from_device_all_layer(None, dest, source, "direct")
        stream.synchronize()
        self.assertTrue(torch.equal(self.host.kv_buffer, expected))
        restore = torch.randperm(2048)[:1025]
        for layer, buf in enumerate(self.src):
            buf.fill_(19)
            result = torch.full_like(originals[layer], 19)
            self.copy_tokens(result, expected[:, layer, 0], restore, dest)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                self.host.load_to_device_per_layer(None, dest, restore, layer, "direct")
            stream.synchronize()
            self.assertTrue(torch.equal(buf.cpu(), result))

    def test_whole_pages_and_destroy(self):
        source_rows, dest_rows = torch.tensor([6, 2, 9]), torch.tensor([5, 1, 7])
        source = (source_rows[:, None] * 256 + torch.arange(256)).flatten()
        dest = (dest_rows[:, None] * 256 + torch.arange(256)).flatten()
        expected = self.host.kv_buffer.clone()
        for layer, buf in enumerate(self.src):
            expected[dest_rows, layer, 0] = buf.cpu()[source_rows]
        self.host.backup_from_device_all_layer(None, dest, source, "direct")
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(self.host.kv_buffer, expected))
        for layer, buf in enumerate(self.src):
            buf.zero_()
            self.host.load_to_device_per_layer(None, dest, source, layer, "direct")
            reference = torch.zeros_like(buf.cpu())
            reference[source_rows] = expected[dest_rows, layer, 0]
            self.assertTrue(torch.equal(buf.cpu(), reference))
        refs = [
            weakref.ref(self.host._staged_direct_backup),
            weakref.ref(self.host._staged_direct_load),
            weakref.ref(self.host.kv_buffer),
        ]
        self.host.destroy()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))
        self.host.destroy()

    def test_partial_transfer_rejects_invalid_device_indices(self):
        before = self.host.kv_buffer.clone()
        for index in (-1, 2048):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, "device index"):
                    self.host.backup_from_device_all_layer(
                        None, torch.tensor([0]), torch.tensor([index]), "direct"
                    )
        self.assertTrue(torch.equal(self.host.kv_buffer, before))


if __name__ == "__main__":
    unittest.main()
