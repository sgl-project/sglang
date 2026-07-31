import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _make_eager_idle_topk_result,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"


class TestDSAIndexerIdle(unittest.TestCase):
    def test_builds_invalid_topk_rows_for_padded_idle_tokens(self):
        result = _make_eager_idle_topk_result(
            torch.empty((2, 16)),
            index_topk=8,
            return_indices=True,
        )

        self.assertEqual(result.shape, (2, 8))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))

    def test_returns_none_when_indices_are_not_consumed(self):
        self.assertIsNone(
            _make_eager_idle_topk_result(
                torch.empty((2, 16)),
                index_topk=8,
                return_indices=False,
            )
        )

    def test_eager_idle_short_circuits_before_paged_mqa(self):
        indexer = SimpleNamespace(index_topk=8)
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)

        with (
            patch(f"{_INDEXER}._is_cuda", True),
            patch(f"{_INDEXER}.get_is_capture_mode", return_value=False),
            patch(
                f"{_INDEXER}._broadcast_indexer_topk_from_rank0",
                side_effect=lambda result: result,
            ),
            patch(
                f"{_INDEXER}.maybe_capture_indexer_topk",
                side_effect=lambda _layer_id, result: result,
            ),
        ):
            result = Indexer.forward_cuda(
                indexer,
                x=torch.empty((2, 16)),
                q_lora=torch.empty((2, 16)),
                positions=torch.empty((2,), dtype=torch.int64),
                forward_batch=batch,
                layer_id=3,
            )

        self.assertEqual(result.shape, (2, 8))
        self.assertTrue(torch.all(result == -1))


if __name__ == "__main__":
    unittest.main()
