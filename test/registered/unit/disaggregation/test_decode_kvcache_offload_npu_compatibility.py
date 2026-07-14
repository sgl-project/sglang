import unittest
from typing import NoReturn
from unittest.mock import patch

from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
    DecodeKVCacheOffloadManager,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _DependencyProbe:
    def __getattribute__(self, name: str) -> NoReturn:
        raise AssertionError(f"Decode offload dependency was accessed: {name}")


class TestDecodeKVCacheOffloadNpuCompatibility(unittest.TestCase):
    def test_rejects_npu_before_touching_dependencies(self) -> None:
        """The defensive NPU gate runs before constructor side effects."""
        manager = object.__new__(DecodeKVCacheOffloadManager)
        dependency_probe = _DependencyProbe()

        with (
            patch(
                "sglang.srt.disaggregation.decode_kvcache_offload_manager.is_npu",
                return_value=True,
            ),
            self.assertRaisesRegex(ValueError, "not supported on NPU"),
        ):
            manager.__init__(
                req_to_token_pool=dependency_probe,
                token_to_kv_pool_allocator=dependency_probe,
                tp_group=dependency_probe,
                tree_cache=dependency_probe,
                server_args=dependency_probe,
            )

        self.assertEqual(vars(manager), {})


if __name__ == "__main__":
    unittest.main()
