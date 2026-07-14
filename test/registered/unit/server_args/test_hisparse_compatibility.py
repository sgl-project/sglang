import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.hisparse_hook import validate_hisparse
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparseCompatibility(unittest.TestCase):
    def test_rejects_decode_offload_before_model_inspection(self) -> None:
        """HiSparse rejects decode offload before model or pool construction."""
        server_args = SimpleNamespace(
            enable_hisparse=True,
            disaggregation_decode_enable_offload_kvcache=True,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Generic DSA.*incorrect data and leaks.*DeepSeek V4",
        ):
            validate_hisparse(server_args)

    def test_decode_offload_without_hisparse_remains_allowed(self) -> None:
        """Decode offload alone does not enter HiSparse compatibility checks."""
        server_args = SimpleNamespace(
            enable_hisparse=False,
            disaggregation_decode_enable_offload_kvcache=True,
        )

        validate_hisparse(server_args)


if __name__ == "__main__":
    unittest.main()
