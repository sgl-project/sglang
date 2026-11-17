import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.mooncake.utils import (
    _validate_efa_allocator_compatibility,
    check_mooncake_custom_mem_pool_enabled,
)
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMooncakeEfaAllocator(unittest.TestCase):
    def _check(self, protocol, custom_mem_pool=None, allocator_env=None):
        allocator_env = allocator_env or {}
        with (
            patch.object(envs.MOONCAKE_PROTOCOL, "get", return_value=protocol),
            patch.object(
                envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL,
                "get",
                return_value=custom_mem_pool,
            ),
            patch(
                "sglang.srt.disaggregation.mooncake.utils.os.environ.get",
                side_effect=lambda key, default="": allocator_env.get(key, default),
            ),
        ):
            result = check_mooncake_custom_mem_pool_enabled()
            _validate_efa_allocator_compatibility(*result)
            return result

    def test_efa_rejects_custom_memory_pools(self):
        for pool_type in ("true", "NVLINK", "BAREX"):
            with self.subTest(pool_type=pool_type):
                with self.assertRaisesRegex(
                    ValueError, "incompatible with MOONCAKE_PROTOCOL=efa"
                ):
                    self._check("efa", custom_mem_pool=pool_type)

    def test_efa_rejects_expandable_segments(self):
        for var in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
            with self.subTest(var=var):
                with self.assertRaisesRegex(ValueError, var):
                    self._check(
                        "efa",
                        allocator_env={
                            var: "garbage_collection_threshold:0.8,"
                            "expandable_segments:True"
                        },
                    )


if __name__ == "__main__":
    unittest.main()
