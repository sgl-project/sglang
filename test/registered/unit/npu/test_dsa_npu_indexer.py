import unittest

from sglang.srt.layers.attention.dsa.dsa_npu_indexer import (
    _resolve_eager_indexer_batch_size,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestEagerIndexerBatchSize(CustomTestCase):
    def test_decode_removes_dp_padding(self):
        self.assertEqual(
            _resolve_eager_indexer_batch_size(
                16,
                5,
                1,
                is_prefill=False,
                graph_mode=False,
            ),
            5,
        )

    def test_verify_keeps_real_tokens_per_request(self):
        self.assertEqual(
            _resolve_eager_indexer_batch_size(
                32,
                5,
                4,
                is_prefill=False,
                graph_mode=False,
            ),
            20,
        )

    def test_prefill_and_graph_keep_static_shape(self):
        for is_prefill, graph_mode in ((True, False), (False, True)):
            with self.subTest(is_prefill=is_prefill, graph_mode=graph_mode):
                self.assertEqual(
                    _resolve_eager_indexer_batch_size(
                        16,
                        5,
                        1,
                        is_prefill=is_prefill,
                        graph_mode=graph_mode,
                    ),
                    16,
                )


if __name__ == "__main__":
    unittest.main()
