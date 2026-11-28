"""Native KV move must index tokens on page-major per-layer views."""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool import move_kv_cache_native
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestNativeKVMPageMajorMove(CustomTestCase):
    def test_page_major_view_moves_by_token(self):
        pages, page_size, heads, dim = 4, 3, 2, 2
        k = torch.arange(pages * page_size * heads * dim, dtype=torch.float32).reshape(
            pages, page_size, heads, dim
        )
        v = k + 1000.0
        src = torch.tensor([1, 4, 7, 11])
        tgt = torch.tensor([0, 5, 8, 2])

        expected_k = k.clone().reshape(-1, heads, dim)
        expected_v = v.clone().reshape(-1, heads, dim)
        expected_k[tgt] = expected_k[src]
        expected_v[tgt] = expected_v[src]

        move_kv_cache_native([k], [v], tgt, src)

        torch.testing.assert_close(k.reshape(-1, heads, dim), expected_k)
        torch.testing.assert_close(v.reshape(-1, heads, dim), expected_v)


if __name__ == "__main__":
    unittest.main()
