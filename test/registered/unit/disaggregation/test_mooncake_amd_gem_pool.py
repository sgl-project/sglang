import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.mooncake import utils as mooncake_utils
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMooncakeAmdGemPool(unittest.TestCase):
    def _with_pool_env(self, value):
        return patch.object(
            envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL, "get", return_value=value
        )

    def test_amd_gem_is_a_supported_pool_type(self):
        with self._with_pool_env("AMD_GEM"):
            self.assertEqual(
                mooncake_utils.check_mooncake_custom_mem_pool_enabled(),
                (True, "AMD_GEM"),
            )

    def test_amd_gem_pool_uses_hip_gem_allocator(self):
        hip_gem = MagicMock()
        fake_allocator_module = types.ModuleType("mooncake.allocator")
        fake_allocator_module.HipGemAllocator = hip_gem
        fake_mooncake = types.ModuleType("mooncake")
        fake_mooncake.allocator = fake_allocator_module
        mem_pool = MagicMock(name="MemPool")
        with (
            self._with_pool_env("AMD_GEM"),
            patch.dict(
                sys.modules,
                {
                    "mooncake": fake_mooncake,
                    "mooncake.allocator": fake_allocator_module,
                },
            ),
            patch.object(mooncake_utils.torch.cuda, "MemPool", mem_pool),
            patch.object(mooncake_utils.torch.cuda, "device", MagicMock()),
        ):
            enabled, pool, pool_type = mooncake_utils.init_mooncake_custom_mem_pool(
                "cuda:3"
            )
        hip_gem.get_allocator.assert_called_once_with("cuda:3")
        mem_pool.assert_called_once_with(
            hip_gem.get_allocator.return_value.allocator.return_value
        )
        self.assertEqual(
            (enabled, pool, pool_type), (True, mem_pool.return_value, "AMD_GEM")
        )


if __name__ == "__main__":
    unittest.main()
