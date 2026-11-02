"""DSA indexer logits budget pinned before prefill CUDA graph capture.

Prefill CUDA graphs pin the budget before capture, so the graph pool does not
shrink it, and keep the smallest one of the TP group.

    python -m pytest test/registered/unit/layers/attention/test_dsa_indexer_logits_budget.py -v
"""

import unittest
from unittest import mock

from sglang.srt.layers.attention.dsa import dsa_indexer
from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

GIB = 1 << 30


class TestPinnedBudget(unittest.TestCase):
    def setUp(self):
        self._saved = dict(Indexer._mqa_logits_budget_bytes)
        Indexer._mqa_logits_budget_bytes.clear()

    def tearDown(self):
        Indexer._mqa_logits_budget_bytes.clear()
        Indexer._mqa_logits_budget_bytes.update(self._saved)

    def _measure(self, *values):
        return mock.patch.object(
            dsa_indexer, "mqa_logits_budget_bytes", side_effect=list(values)
        )

    def test_measures_once_and_later_calls_only_lower_it(self):
        with self._measure(2 * GIB) as measure:
            self.assertEqual(Indexer.pin_mqa_logits_budget(0, None), 2 * GIB)
            # After capture: free memory dropped, but the budget stays unless
            # it exceeds the cap.
            self.assertEqual(
                Indexer.pin_mqa_logits_budget(0, None, max_bytes=8 * GIB), 2 * GIB
            )
            self.assertEqual(Indexer.pin_mqa_logits_budget(0, None, max_bytes=GIB), GIB)
        self.assertEqual(measure.call_count, 1)
        self.assertEqual(Indexer._mqa_logits_budget_bytes[0], GIB)

    def test_takes_the_group_minimum(self):
        def all_reduce(tensor, op, group):
            tensor.fill_(min(int(tensor.item()), GIB // 2))

        with (
            self._measure(2 * GIB),
            mock.patch.object(dsa_indexer.dist, "get_world_size", return_value=8),
            mock.patch.object(dsa_indexer.dist, "all_reduce", side_effect=all_reduce),
        ):
            budget = Indexer.pin_mqa_logits_budget(0, object())
        self.assertEqual(budget, GIB // 2)
        self.assertEqual(Indexer._mqa_logits_budget_bytes[0], GIB // 2)


if __name__ == "__main__":
    unittest.main()
