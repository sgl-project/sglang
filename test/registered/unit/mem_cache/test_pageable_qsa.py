"""QSA pageable whole-page transfers and teardown on real CUDA buffers."""

import gc
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.pool_host import common
from sglang.srt.mem_cache.pool_host.qsa import QSAIndexerPoolHost
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPageableQSA(unittest.TestCase):
    @staticmethod
    def expand_pages(pages):
        return (pages[:, None] * 64 + torch.arange(64)).flatten()

    def test_roundtrip_and_destroy(self):
        for dtype in (torch.uint8, torch.bfloat16):
            with (
                self.subTest(dtype=dtype),
                patch.dict(os.environ, {"SGLANG_HICACHE_PAGEABLE": "1"}),
                patch.object(
                    common,
                    "_cuda_host_register",
                    side_effect=AssertionError("register"),
                ),
                patch.object(
                    common,
                    "_cuda_host_unregister",
                    side_effect=AssertionError("unregister"),
                ),
            ):
                buffers = [
                    torch.randint(0, 100, (256, 128), device="cuda").to(dtype)
                    for _ in range(2)
                ]
                originals = [
                    buf.cpu().view(torch.uint8).reshape(16, -1).clone()
                    for buf in buffers
                ]
                device = SimpleNamespace(
                    qsa_compress_ratio=4, qsa_compressed_k_buffer_pool=buffers
                )
                anchor = SimpleNamespace(
                    page_size=64,
                    page_num=16,
                    layout="page_first_direct",
                    device="cpu",
                    pin_memory=False,
                )
                pool = QSAIndexerPoolHost(device, anchor)
                try:
                    source = torch.tensor([1, 3, 0])
                    dest = torch.tensor([6, 2, 5])
                    restore = torch.tensor([4, 7, 9])
                    pool.kv_buffer.fill_(73)
                    expected = pool.kv_buffer.clone()
                    for layer in range(2):
                        expected[dest, layer, 0] = originals[layer][source]
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        pool.backup_from_device_all_layer(
                            None,
                            self.expand_pages(dest),
                            self.expand_pages(source).cuda(),
                            "direct",
                        )
                    stream.synchronize()
                    self.assertTrue(torch.equal(pool.kv_buffer, expected))
                    for layer, buf in enumerate(buffers):
                        buf.zero_()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            pool.load_to_device_per_layer(
                                None,
                                self.expand_pages(dest),
                                self.expand_pages(restore).cuda(),
                                layer,
                                "direct",
                            )
                        stream.synchronize()
                        reference = torch.zeros_like(originals[layer])
                        reference[restore] = originals[layer][source]
                        self.assertTrue(
                            torch.equal(
                                buf.cpu().view(torch.uint8).reshape(16, -1), reference
                            )
                        )
                    refs = [
                        weakref.ref(pool.kv_buffer),
                        weakref.ref(pool._staged_direct_backup),
                        weakref.ref(pool._staged_direct_load),
                    ]
                    pool.destroy()
                    gc.collect()
                    self.assertTrue(all(ref() is None for ref in refs))
                    pool.destroy()
                finally:
                    pool.destroy()


if __name__ == "__main__":
    unittest.main()
