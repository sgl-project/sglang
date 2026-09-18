import unittest

from sglang.srt.utils import is_cuda, is_hip
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

HELPERS = ("logits_head_gate_graph", "scale_head_gate_graph")


class TestDsaHeadGateGuard(CustomTestCase):
    """The head-gate helpers are defined in one module and imported in another.

    Both sides carry their own platform condition, so the two can drift: widen
    the import without widening the definition and every DSA model fails to
    import on the platform in between. That is a startup failure far from its
    cause, and no CUDA runner can see it.
    """

    def test_definition_and_import_agree(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer, dsa_prefill_cuda_graph

        for name in HELPERS:
            with self.subTest(helper=name):
                self.assertEqual(
                    hasattr(dsa_prefill_cuda_graph, name),
                    hasattr(dsa_indexer, name),
                    f"{name} is defined on one side of the guard but not the other",
                )

    def test_helpers_exist_on_cuda_and_hip(self):
        from sglang.srt.layers.attention.dsa import dsa_prefill_cuda_graph

        expected = is_cuda() or is_hip()
        for name in HELPERS:
            with self.subTest(helper=name):
                self.assertEqual(hasattr(dsa_prefill_cuda_graph, name), expected)


if __name__ == "__main__":
    unittest.main()
