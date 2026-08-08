"""Unit tests for srt/layers/attention/aiter_backend."""

import unittest

import torch

from sglang.srt.layers.attention.aiter_backend import _get_aiter_max_batch_size
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterMaxBatchSize(CustomTestCase):
    def test_includes_padding_row(self):
        # req_to_token has `req_to_token_pool.size + 1` rows: index 0 is the
        # padding row that `size` does not count. Overlap scheduling can hand
        # that row to the backend as an extra decode entry, so metadata must be
        # sized from the tensor's row capacity, not from the pool size.
        pool_size = 111
        req_to_token = torch.empty((pool_size + 1, 8), device="meta")

        max_bs = _get_aiter_max_batch_size(req_to_token, topk=1)
        self.assertEqual(max_bs, pool_size + 1)
        self.assertGreater(max_bs, pool_size)

    def test_scales_with_topk(self):
        # Speculative decoding draws `topk` draft rows per request, so the
        # metadata buffers must be topk times larger.
        req_to_token = torch.empty((112, 8), device="meta")

        for topk in (1, 2, 4, 8):
            with self.subTest(topk=topk):
                self.assertEqual(
                    _get_aiter_max_batch_size(req_to_token, topk=topk), 112 * topk
                )


if __name__ == "__main__":
    unittest.main()
