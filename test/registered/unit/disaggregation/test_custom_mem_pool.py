import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation import custom_mem_pool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCustomMemPoolResolver(unittest.TestCase):
    def test_disabled_without_server_args_or_mooncake_config(self):
        with (
            patch.object(
                custom_mem_pool,
                "get_server_args",
                side_effect=ValueError("not initialized"),
            ),
            patch.object(
                custom_mem_pool.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL,
                "get",
                return_value=None,
            ),
        ):
            self.assertEqual(
                custom_mem_pool.maybe_init_custom_mem_pool("cuda:0"),
                (False, None, None),
            )

    def test_mori_provider_is_selected_from_server_args(self):
        init_mori_pool = MagicMock(return_value=(True, "mori_pool", "mori_fabric"))
        mori_pool_module = types.ModuleType(
            "sglang.srt.disaggregation.mori.custom_mem_pool"
        )
        mori_pool_module.init_mori_custom_mem_pool = init_mori_pool
        mori_package = types.ModuleType("sglang.srt.disaggregation.mori")
        mori_package.__path__ = []
        server_args = SimpleNamespace(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="mori",
        )

        with (
            patch.object(custom_mem_pool, "get_server_args", return_value=server_args),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.mori": mori_package,
                    "sglang.srt.disaggregation.mori.custom_mem_pool": mori_pool_module,
                },
            ),
        ):
            result = custom_mem_pool.maybe_init_custom_mem_pool("cuda:0")

        self.assertEqual(result, (True, "mori_pool", "mori_fabric"))
        init_mori_pool.assert_called_once_with("cuda:0")

    def test_mori_config_does_not_affect_other_transfer_backends(self):
        server_args = SimpleNamespace(
            disaggregation_mode="prefill",
            disaggregation_transfer_backend="nixl",
        )
        init_mori_pool = MagicMock()
        mori_pool_module = types.ModuleType(
            "sglang.srt.disaggregation.mori.custom_mem_pool"
        )
        mori_pool_module.init_mori_custom_mem_pool = init_mori_pool
        mori_package = types.ModuleType("sglang.srt.disaggregation.mori")
        mori_package.__path__ = []

        with (
            patch.object(custom_mem_pool, "get_server_args", return_value=server_args),
            patch.object(
                custom_mem_pool.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL,
                "get",
                return_value=None,
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.mori": mori_package,
                    "sglang.srt.disaggregation.mori.custom_mem_pool": mori_pool_module,
                },
            ),
            patch.dict(os.environ, {"SGLANG_MORI_BACKEND": "fabric"}),
        ):
            result = custom_mem_pool.maybe_init_custom_mem_pool("cuda:0")

        self.assertEqual(result, (False, None, None))
        init_mori_pool.assert_not_called()

    def test_mori_provider_is_not_used_outside_pd_mode(self):
        server_args = SimpleNamespace(
            disaggregation_mode="null",
            disaggregation_transfer_backend="mori",
        )
        init_mori_pool = MagicMock()
        mori_pool_module = types.ModuleType(
            "sglang.srt.disaggregation.mori.custom_mem_pool"
        )
        mori_pool_module.init_mori_custom_mem_pool = init_mori_pool
        mori_package = types.ModuleType("sglang.srt.disaggregation.mori")
        mori_package.__path__ = []

        with (
            patch.object(custom_mem_pool, "get_server_args", return_value=server_args),
            patch.object(
                custom_mem_pool.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL,
                "get",
                return_value=None,
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.mori": mori_package,
                    "sglang.srt.disaggregation.mori.custom_mem_pool": mori_pool_module,
                },
            ),
            patch.dict(os.environ, {"SGLANG_MORI_BACKEND": "fabric"}),
        ):
            result = custom_mem_pool.maybe_init_custom_mem_pool("cuda:0")

        self.assertEqual(result, (False, None, None))
        init_mori_pool.assert_not_called()

    def test_mooncake_env_behavior_is_preserved(self):
        init_mooncake_pool = MagicMock(return_value=(True, "mooncake_pool", "NVLINK"))
        mooncake_utils = types.ModuleType("sglang.srt.disaggregation.mooncake.utils")
        mooncake_utils.init_mooncake_custom_mem_pool = init_mooncake_pool
        mooncake_package = types.ModuleType("sglang.srt.disaggregation.mooncake")
        mooncake_package.__path__ = []

        with (
            patch.object(
                custom_mem_pool,
                "get_server_args",
                side_effect=ValueError("not initialized"),
            ),
            patch.object(
                custom_mem_pool.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL,
                "get",
                return_value="NVLINK",
            ),
            patch.dict(
                sys.modules,
                {
                    "sglang.srt.disaggregation.mooncake": mooncake_package,
                    "sglang.srt.disaggregation.mooncake.utils": mooncake_utils,
                },
            ),
        ):
            result = custom_mem_pool.maybe_init_custom_mem_pool("cuda:0")

        self.assertEqual(result, (True, "mooncake_pool", "NVLINK"))
        init_mooncake_pool.assert_called_once_with("cuda:0")


if __name__ == "__main__":
    unittest.main()
