import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _exec(deterministic):
    return SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )


class TestMinimaxGluonPrefillSplits(CustomTestCase):
    """The recommended split count of the Gluon sparse prefill follows the number of
    query tokens in the forward, so deterministic inference must keep one partition.
    """

    def _part_nums(self, deterministic):
        from sglang.kernels.ops.attention.minimax_sparse.prefill import gluon_prefill

        def recommended(num_seqs, _):
            return min(-(-1024 // num_seqs), 8)

        with (
            mock.patch.object(gluon_prefill, "_deterministic_inference", None),
            mock.patch.object(gluon_prefill, "get_recommended_splits", recommended),
            mock.patch(
                "sglang.srt.runtime_context.get_exec", return_value=_exec(deterministic)
            ),
        ):
            return [gluon_prefill._max_part_num(n) for n in (1, 200, 772, 4096)]

    def test_normal_mode_uses_the_recommended_splits(self):
        self.assertEqual(self._part_nums(deterministic=False), [8, 6, 2, 1])

    def test_deterministic_mode_uses_one_partition(self):
        self.assertEqual(self._part_nums(deterministic=True), [1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
