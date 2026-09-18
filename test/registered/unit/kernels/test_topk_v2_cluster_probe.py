"""Host-only regression tests for optional TopK v2 cluster dispatch."""

import unittest
from unittest.mock import call, patch

from sglang.kernels.jit.utils import occupancy
from sglang.kernels.ops.attention.dsv4 import topk
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTopKV2ClusterProbe(CustomTestCase):
    def test_zero_capacity_is_distinct_from_probe_failure(self):
        with patch.object(occupancy, "_get_max_active_clusters", return_value=0):
            with self.assertRaises(occupancy.NoSchedulableClustersError) as error:
                occupancy.get_max_active_clusters(16, occupancy=1)
        # Preserve the exception contract of existing callers.
        self.assertIsInstance(error.exception, ValueError)
        self.assertIn("no cluster of 16 fits at occupancy 1", str(error.exception))

    def test_positive_capacity_is_preserved(self):
        with patch.object(occupancy, "_get_max_active_clusters", return_value=7):
            self.assertEqual(occupancy.get_max_active_clusters(16, occupancy=1), 7)

    def _assert_compiled_capacity(self, capacity16):
        with (
            patch.object(topk, "is_arch_support_pdl", return_value=True),
            patch.object(
                occupancy, "_get_max_active_clusters", side_effect=[15, capacity16]
            ) as probe,
            patch.object(topk, "load_jit") as compile_module,
        ):
            # Bypass the process cache to exercise module configuration each time.
            topk._jit_topk_v2_module.__wrapped__()

        self.assertEqual(probe.call_args_list, [call(8, 2), call(16, 1)])
        flags = compile_module.call_args.kwargs["extra_cuda_cflags"]
        self.assertIn("-DSGL_TOPK_V2_MAX_C8_OCC2=15", flags)
        self.assertIn(f"-DSGL_TOPK_V2_MAX_C16_OCC1={capacity16}", flags)

    def test_zero_c16_capacity_is_compiled(self):
        self._assert_compiled_capacity(0)

    def test_single_c16_cluster_is_preserved(self):
        self._assert_compiled_capacity(1)

    def test_multiple_c16_clusters_are_preserved(self):
        self._assert_compiled_capacity(7)

    def _assert_probe_error_propagates(self, error_type):
        error = error_type("occupancy probe failed")
        with (
            patch.object(topk, "is_arch_support_pdl", return_value=True),
            patch.object(
                occupancy, "_get_max_active_clusters", side_effect=[15, error]
            ),
            patch.object(topk, "load_jit") as compile_module,
        ):
            with self.assertRaises(error_type) as raised:
                topk._jit_topk_v2_module.__wrapped__()
        self.assertIs(raised.exception, error)
        compile_module.assert_not_called()

    def test_unexpected_c16_runtime_error_is_not_swallowed(self):
        self._assert_probe_error_propagates(RuntimeError)

    def test_unexpected_c16_value_error_is_not_swallowed(self):
        self._assert_probe_error_propagates(ValueError)

    def test_c8_probe_failure_keeps_existing_fallback(self):
        with (
            patch.object(topk, "is_arch_support_pdl", return_value=True),
            patch.object(
                occupancy,
                "_get_max_active_clusters",
                side_effect=[RuntimeError("C8 probe failed"), 0],
            ),
            patch.object(topk, "load_jit") as compile_module,
        ):
            topk._jit_topk_v2_module.__wrapped__()
        self.assertEqual(
            compile_module.call_args.kwargs["extra_cuda_cflags"],
            ["-DSGL_TOPK_V2_MAX_C16_OCC1=0"],
        )

    def test_non_cluster_architecture_does_not_probe(self):
        with (
            patch.object(topk, "is_arch_support_pdl", return_value=False),
            patch.object(occupancy, "_get_max_active_clusters") as probe,
            patch.object(topk, "load_jit") as compile_module,
        ):
            topk._jit_topk_v2_module.__wrapped__()
        probe.assert_not_called()
        self.assertEqual(compile_module.call_args.kwargs["extra_cuda_cflags"], [])


if __name__ == "__main__":
    unittest.main()
