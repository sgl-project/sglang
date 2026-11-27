import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiniMaxSparseBackendScoping(CustomTestCase):
    """MiniMaxSparseAttnBackend.__init__ calls the module-level get_parallel() on
    the MSA path. A function-level import of get_parallel further down the same
    method made the name local to all of __init__, so a GPU start that took the
    MSA path raised UnboundLocalError. No CPU runner builds this backend, so the
    scoping is checked directly."""

    def test_get_parallel_is_not_a_local_of_init(self):
        from sglang.srt.layers.attention.minimax_sparse_backend import (
            MiniMaxSparseAttnBackend,
        )

        self.assertNotIn(
            "get_parallel", MiniMaxSparseAttnBackend.__init__.__code__.co_varnames
        )


if __name__ == "__main__":
    unittest.main()
