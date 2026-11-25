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
    def test_fp4_verify_records_query_consumer_stream(self):
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
        query = torch.ones((3, 4), dtype=torch.bfloat16)
        weights = torch.zeros((3, 1), dtype=torch.bfloat16)
        locations = torch.arange(3)
        indexer._fused_k_weights = Mock(return_value=(query, weights))
        projected = []

        def project(_):
            projected.append(current)
            return query, None

        indexer.wq_b = project
        indexer._fused_k_prepare_and_store = Mock()
        for mode, capture, count, dual in (
            (ForwardMode.TARGET_VERIFY, True, None, True),
            (ForwardMode.TARGET_VERIFY, True, 2, True),
            (ForwardMode.TARGET_VERIFY, False, None, True),
            (ForwardMode.DECODE, True, None, True),
            (ForwardMode.TARGET_VERIFY, True, None, False),
        ):
            record = Mock()
            with (
                self.subTest(mode=mode, capture=capture, count=count, dual=dual),
                patch.object(dsa_indexer, "get_is_capture_mode", return_value=capture),
                patch.object(
                    dsa_fp4_indexer, "indexer_head_gate", return_value=weights
                ),
                patch.object(torch.cuda, "current_stream", lambda: current),
                patch.object(torch.cuda, "stream", use_stream),
                patch.object(torch.Tensor, "record_stream", lambda q, s: record(q, s)),
            ):
                batch = SimpleNamespace(forward_mode=mode, out_cache_loc=locations)
                q, _ = indexer._fused_q_prepare_and_store(
                    query,
                    query,
                    locations,
                    batch,
                    0,
                    None,
                    num_tokens=count,
                    enable_dual_stream=dual,
                )
                self.assertIs(projected.pop(), aux if dual else main)
                if mode == ForwardMode.TARGET_VERIFY and capture and dual:
                    record.assert_called_once()
                    marked, consumer = record.call_args.args
                    self.assertIs(marked, q)
                    self.assertIs(consumer, main)
                else:
                    record.assert_not_called()


if __name__ == "__main__":
    unittest.main()
