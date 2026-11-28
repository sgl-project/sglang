import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa import dsa_fp4_indexer, dsa_indexer
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDSAIndexerQueryStream(CustomTestCase):
    def test_fp4_verify_projects_query_on_consumer_stream(self):
        main, aux = Mock(), Mock()
        current = main

        @contextmanager
        def use_stream(stream):
            nonlocal current
            previous, current = current, stream
            try:
                yield
            finally:
                current = previous

        indexer = dsa_fp4_indexer.Fp4Indexer.__new__(dsa_fp4_indexer.Fp4Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.alt_stream = aux
        indexer.n_heads = 1
        indexer.head_dim = 4
        indexer.softmax_scale = 1.0
        query = torch.ones((2, 4), dtype=torch.bfloat16)
        weights = torch.zeros((2, 1), dtype=torch.bfloat16)
        indexer._fused_k_weights = Mock(return_value=(query, weights))
        projection_streams, store_streams = [], []

        def project(_):
            projection_streams.append(current)
            return query, None

        indexer.wq_b = project
        indexer._fused_k_prepare_and_store = lambda *a, **kw: store_streams.append(
            current
        )
        for mode, capture, expected in (
            (ForwardMode.TARGET_VERIFY, True, main),
            (ForwardMode.TARGET_VERIFY, False, aux),
            (ForwardMode.DECODE, True, aux),
        ):
            with (
                self.subTest(mode=mode, capture=capture),
                patch.object(dsa_indexer, "get_is_capture_mode", return_value=capture),
                patch.object(
                    dsa_fp4_indexer, "indexer_head_gate", return_value=weights
                ),
                patch.object(torch.cuda, "current_stream", lambda: current),
                patch.object(torch.cuda, "stream", use_stream),
            ):
                batch = SimpleNamespace(forward_mode=mode, out_cache_loc=None)
                indexer._fused_q_prepare_and_store(query, query, None, batch, 0, None)
                self.assertIs(projection_streams.pop(), expected)
                self.assertIs(store_streams.pop(), aux)


if __name__ == "__main__":
    unittest.main()
