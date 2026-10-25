from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")

import unittest

import torch

from sglang.srt.mem_cache.kv_page_zeroer import KVPageZeroer
from sglang.test.test_utils import CustomTestCase

PAGE_SIZE = 64
NUM_PAGES = 32


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestKVPageZeroerCuda(CustomTestCase):
    def test_zeroes_listed_pages_in_all_buffers(self):
        # MLA-like bf16 rows (576 dims) and a DSA index-K-like uint8 buffer.
        kv = [
            torch.randn((NUM_PAGES + 1) * PAGE_SIZE, 1, 576, device="cuda").to(
                torch.bfloat16
            )
            for _ in range(3)
        ]
        index_k = torch.randint(
            1, 255, (NUM_PAGES + 1, PAGE_SIZE * 132), dtype=torch.uint8, device="cuda"
        )
        buffers = [
            b.view(torch.uint8).view(-1, PAGE_SIZE * b[0].nbytes) for b in kv
        ] + [index_k]
        expected = [b.clone() for b in buffers]

        forward_stream = torch.cuda.Stream()
        zeroer = KVPageZeroer(buffers, wait_streams=[forward_stream])
        pages = torch.tensor([1, 5, 6, 31, 32], dtype=torch.int64, device="cuda")
        event = zeroer.zero(pages)
        event.synchronize()

        for buf, ref in zip(buffers, expected):
            ref[pages] = 0
            self.assertTrue(torch.equal(buf, ref))

    def test_waits_for_forward_stream(self):
        buf = torch.zeros((NUM_PAGES + 1, 4096), dtype=torch.uint8, device="cuda")
        forward_stream = torch.cuda.Stream()
        zeroer = KVPageZeroer([buf], wait_streams=[forward_stream])
        pages = torch.tensor([3], dtype=torch.int64, device="cuda")
        with torch.cuda.stream(forward_stream):
            torch.cuda._sleep(50_000_000)
            buf[3].fill_(9)
        zeroer.zero(pages).synchronize()
        self.assertEqual(int(buf[3].sum()), 0)


if __name__ == "__main__":
    unittest.main()
