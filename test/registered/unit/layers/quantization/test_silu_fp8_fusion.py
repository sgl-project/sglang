import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.quantization import silu_fp8_fusion as fusion
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestSiluFp8Support(CustomTestCase):
    def setUp(self):
        super().setUp()
        fusion.silu_static_fp8_supported.cache_clear()

    def tearDown(self):
        fusion.silu_static_fp8_supported.cache_clear()
        super().tearDown()

    def test_consumer_capability(self):
        with (
            patch.object(fusion.torch.cuda, "is_available", return_value=True),
            patch.object(fusion.torch.version, "hip", None),
            patch.object(
                fusion.torch.cuda, "get_device_capability", return_value=(8, 9)
            ),
        ):
            for supported in (False, True):
                fusion.silu_static_fp8_supported.cache_clear()
                with patch(
                    "sglang.srt.layers.quantization.fp8_utils.cutlass_fp8_supported",
                    return_value=supported,
                ):
                    self.assertEqual(fusion.silu_static_fp8_supported(), supported)

    def test_no_cuda(self):
        with patch.object(fusion.torch.cuda, "is_available", return_value=False):
            self.assertFalse(fusion.silu_static_fp8_supported())

    def test_graph_backend_not_unused_compiler_field(self):
        from sglang.srt.layers.quantization.silu_fp8_fusion import (
            supports_silu_fp8_graph,
        )

        for backend in ("disabled", "full", "breakable"):
            for compiler in ("eager", "inductor"):
                phase = SimpleNamespace(backend=backend, tc_compiler=compiler)
                self.assertTrue(
                    supports_silu_fp8_graph(
                        SimpleNamespace(prefill=phase, decode=phase)
                    )
                )
        phase = SimpleNamespace(backend="tc_piecewise", tc_compiler="eager")
        self.assertFalse(
            supports_silu_fp8_graph(SimpleNamespace(prefill=phase, decode=phase))
        )

    def test_other_architecture(self):
        with (
            patch.object(fusion.torch.cuda, "is_available", return_value=True),
            patch.object(fusion.torch.version, "hip", None),
            patch.object(
                fusion.torch.cuda, "get_device_capability", return_value=(9, 0)
            ),
        ):
            self.assertFalse(fusion.silu_static_fp8_supported())


if __name__ == "__main__":
    unittest.main()
