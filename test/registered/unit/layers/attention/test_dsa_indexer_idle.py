import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.dsa.dsa_indexer import (
    Indexer,
    _make_eager_idle_topk_result,
    _skips_eager_idle,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_INDEXER = "sglang.srt.layers.attention.dsa.dsa_indexer"


class TestDSAIndexerIdle(unittest.TestCase):
    def test_builds_invalid_topk_rows_for_padded_idle_tokens(self):
        result = _make_eager_idle_topk_result(
            torch.empty((8, 16)),
            index_topk=4,
            return_indices=True,
        )

        self.assertEqual(result.shape, (8, 4))
        self.assertEqual(result.dtype, torch.int32)
        self.assertTrue(torch.all(result == -1))

    def test_returns_none_when_indices_are_not_consumed(self):
        self.assertIsNone(
            _make_eager_idle_topk_result(
                torch.empty((8, 16)),
                index_topk=4,
                return_indices=False,
            )
        )

    def test_capture_and_active_modes_keep_the_normal_path(self):
        for is_cuda, is_hip in ((True, False), (False, True), (False, False)):
            for mode in (ForwardMode.IDLE, ForwardMode.DECODE, ForwardMode.EXTEND):
                for capture in (False, True):
                    with (
                        self.subTest(
                            cuda=is_cuda, hip=is_hip, mode=mode, capture=capture
                        ),
                        patch(f"{_INDEXER}._is_cuda", is_cuda),
                        patch(f"{_INDEXER}._is_hip", is_hip),
                        patch(f"{_INDEXER}.get_is_capture_mode", return_value=capture),
                    ):
                        self.assertEqual(
                            _skips_eager_idle(SimpleNamespace(forward_mode=mode)),
                            (is_cuda or is_hip)
                            and mode == ForwardMode.IDLE
                            and not capture,
                        )

    def test_eager_idle_short_circuits_before_paged_mqa(self):
        for is_cuda, is_hip in ((True, False), (False, True)):
            with self.subTest(is_cuda=is_cuda, is_hip=is_hip):
                self._check_eager_idle_short_circuits(is_cuda, is_hip)

    def _check_eager_idle_short_circuits(self, is_cuda, is_hip):
        # On ROCm the idle rank of a DP2 x DCP2 server faulted in the paged
        # top-k v2 kernel when the indexer ran on its padding rows.
        indexer = SimpleNamespace(index_topk=4)
        batch = SimpleNamespace(forward_mode=ForwardMode.IDLE)

        with (
            patch(f"{_INDEXER}._is_cuda", is_cuda),
            patch(f"{_INDEXER}._is_hip", is_hip),
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
                x=torch.empty((8, 16)),
                q_lora=torch.empty((8, 16)),
                positions=torch.empty((8,), dtype=torch.int64),
                forward_batch=batch,
                layer_id=3,
            )

        self.assertEqual(result.shape, (8, 4))
        self.assertTrue(torch.all(result == -1))


if __name__ == "__main__":
    unittest.main()
