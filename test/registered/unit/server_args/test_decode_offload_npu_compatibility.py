import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDecodeOffloadNpuCompatibility(unittest.TestCase):
    def test_rejects_decode_offload_on_npu(self) -> None:
        """Server argument validation rejects decode offload on NPU."""
        server_args = SimpleNamespace(
            disaggregation_decode_enable_offload_kvcache=True,
            enable_session_radix_cache=False,
            enable_hierarchical_cache=False,
            disable_radix_cache=False,
        )

        with (
            patch("sglang.srt.server_args.is_npu", return_value=True),
            self.assertRaisesRegex(ValueError, "not supported on NPU"),
        ):
            ServerArgs._handle_cache_compatibility(server_args)


if __name__ == "__main__":
    unittest.main()
