import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.mori import utils as mori_utils
from sglang.srt.environ import envs
from sglang.srt.mem_cache import utils as mem_cache_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMoriAmdGemPool(unittest.TestCase):
    def _with_env(self, mori=None, mooncake=None):
        return (
            patch.object(envs.SGLANG_MORI_CUSTOM_MEM_POOL, "get", return_value=mori),
            patch.object(
                envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL, "get", return_value=mooncake
            ),
        )

    def _fake_mori(self):
        hip_gem = MagicMock(name="HipGemAllocator")
        fake_io = types.ModuleType("mori.io")
        fake_io.HipGemAllocator = hip_gem
        fake_mori = types.ModuleType("mori")
        fake_mori.io = fake_io
        modules = patch.dict(sys.modules, {"mori": fake_mori, "mori.io": fake_io})
        return hip_gem, modules

    def test_pool_types(self):
        mori_env, mooncake_env = self._with_env(mori="amd_gem")
        with mori_env, mooncake_env:
            self.assertEqual(
                mori_utils.check_mori_custom_mem_pool_enabled(), (True, "AMD_GEM")
            )
        mori_env, mooncake_env = self._with_env()
        with mori_env, mooncake_env:
            self.assertEqual(
                mori_utils.check_mori_custom_mem_pool_enabled(), (False, None)
            )
        mori_env, mooncake_env = self._with_env(mori="NVLINK")
        with mori_env, mooncake_env, self.assertRaisesRegex(ValueError, "Unsupported"):
            mori_utils.check_mori_custom_mem_pool_enabled()

    def test_amd_gem_pool_uses_mori_hip_gem_allocator(self):
        hip_gem, modules = self._fake_mori()
        mem_pool = MagicMock(name="MemPool")
        device_ctx = MagicMock(name="device")
        mori_env, mooncake_env = self._with_env(mori="AMD_GEM")
        with (
            mori_env,
            mooncake_env,
            modules,
            patch.object(mori_utils.torch.cuda, "MemPool", mem_pool),
            patch.object(mori_utils.torch.cuda, "device", device_ctx),
        ):
            result = mem_cache_utils.maybe_init_custom_mem_pool("cuda:3")
        hip_gem.get_allocator.assert_called_once_with("cuda:3")
        device_ctx.assert_called_once_with("cuda:3")
        mem_pool.assert_called_once_with(
            hip_gem.get_allocator.return_value.allocator.return_value
        )
        self.assertEqual(result, (True, mem_pool.return_value, "AMD_GEM"))

    def test_pins_through_the_pd_nics(self):
        _, modules = self._fake_mori()
        disagg = MagicMock(disaggregation_ib_device="ionic_0,ionic_1")
        mori_env, mooncake_env = self._with_env(mori="AMD_GEM")
        with (
            mori_env,
            mooncake_env,
            modules,
            patch.dict(os.environ, {"MORI_RDMA_DEVICES": "mlx5_0"}),
            patch.object(mori_utils, "get_disagg", return_value=disagg),
            patch.object(mori_utils.torch.cuda, "MemPool", MagicMock()),
            patch.object(mori_utils.torch.cuda, "device", MagicMock()),
        ):
            mori_utils.init_mori_custom_mem_pool("cuda:0")
            self.assertEqual(os.environ["MORI_RDMA_DEVICES"], "ionic_0,ionic_1")

    def test_missing_allocator_fails_instead_of_falling_back(self):
        fake_mori = types.ModuleType("mori")
        mori_env, mooncake_env = self._with_env(mori="AMD_GEM")
        with (
            mori_env,
            mooncake_env,
            patch.dict(sys.modules, {"mori": fake_mori, "mori.io": None}),
            self.assertRaisesRegex(RuntimeError, "HipGemAllocator"),
        ):
            mori_utils.init_mori_custom_mem_pool("cuda:0")

    def test_mori_and_mooncake_pools_are_exclusive(self):
        mori_env, mooncake_env = self._with_env(mori="AMD_GEM", mooncake="NVLINK")
        with mori_env, mooncake_env, self.assertRaisesRegex(ValueError, "only one"):
            mem_cache_utils.maybe_init_custom_mem_pool("cuda:0")

    def test_unset_keeps_default_allocator(self):
        mori_env, mooncake_env = self._with_env()
        with mori_env, mooncake_env:
            self.assertEqual(
                mem_cache_utils.maybe_init_custom_mem_pool("cuda:0"),
                (False, None, None),
            )


if __name__ == "__main__":
    unittest.main()
