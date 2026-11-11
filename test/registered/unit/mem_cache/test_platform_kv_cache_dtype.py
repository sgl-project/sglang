"""KV cache dtypes that out-of-tree platform plugins register."""

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.arg_groups.choices import KV_CACHE_DTYPE_CHOICES
from sglang.srt.mem_cache import kv_cache_configurator as kcc
from sglang.srt.mem_cache import platform_kv_cache_dtype
from sglang.srt.mem_cache.kv_cache_dtype import configure_kv_cache_dtype
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_NAME = "example_platform_kv"


class _PlatformMHAPool(MHATokenToKVPool):
    pass


class _PastTheGuard(Exception):
    pass


class TestPlatformKVCacheDtype(CustomTestCase):
    def setUp(self):
        # The registry and the CLI choices are process-global.
        choices = list(KV_CACHE_DTYPE_CHOICES)
        self.addCleanup(KV_CACHE_DTYPE_CHOICES.__setitem__, slice(None), choices)
        registry = patch.dict(
            platform_kv_cache_dtype._REGISTERED_KV_CACHE_DTYPES, clear=True
        )
        registry.start()
        self.addCleanup(registry.stop)

    def test_registration_adds_the_cli_choice_once(self):
        """General plugins load in every process, so registration repeats."""
        platform_kv_cache_dtype.register_kv_cache_dtype(_NAME)
        platform_kv_cache_dtype.register_kv_cache_dtype(_NAME)
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(["--model-path", "dummy", "--kv-cache-dtype", _NAME])
        self.assertEqual(args.kv_cache_dtype, _NAME)
        self.assertEqual(KV_CACHE_DTYPE_CHOICES.count(_NAME), 1)

    def test_registration_rejects_built_in_and_conflicting_names(self):
        with self.assertRaisesRegex(ValueError, "already built in"):
            platform_kv_cache_dtype.register_kv_cache_dtype("fp8_e4m3")
        platform_kv_cache_dtype.register_kv_cache_dtype(_NAME)
        with self.assertRaisesRegex(ValueError, "already registered"):
            platform_kv_cache_dtype.register_kv_cache_dtype(
                _NAME, storage_dtype=torch.float8_e4m3fn
            )

    def test_the_platform_pool_gets_the_registered_storage_dtype(self):
        platform_kv_cache_dtype.register_kv_cache_dtype(
            _NAME, storage_dtype=torch.float8_e4m3fn
        )
        self.assertIs(self._storage_dtype(_NAME), torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self._storage_dtype("unregistered_kv")

    def test_a_registered_dtype_needs_a_platform_pool_for_the_model_kind(self):
        """No in-tree pool stores a registered dtype, so a kind left to core fails."""
        platform_kv_cache_dtype.register_kv_cache_dtype(_NAME)
        platform = MagicMock()
        platform.get_kv_pool_cls.side_effect = lambda *, kind: (
            _PlatformMHAPool if kind == "mha" else None
        )
        for use_mla_backend, is_dsa_model in ((True, False), (True, True)):
            with self.subTest(use_mla_backend=use_mla_backend, is_dsa=is_dsa_model):
                with self.assertRaisesRegex(ValueError, "needs a platform KV pool"):
                    self._build_pool(platform, use_mla_backend, is_dsa_model)
        with self.assertRaises(_PastTheGuard):
            self._build_pool(platform, use_mla_backend=False, is_dsa_model=False)

    @staticmethod
    def _storage_dtype(name):
        _, dtype = configure_kv_cache_dtype(
            server_args_kv_cache_dtype=name,
            model=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=False,
            is_dflash=False,
            speculative_draft_attention_backend=None,
        )
        return dtype

    @staticmethod
    def _build_pool(platform, use_mla_backend, is_dsa_model):
        configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
        configurator.is_draft_worker = False
        configurator.kv_cache_dtype = torch.uint8
        configurator.kv_cache_dtype_str = _NAME
        configurator.use_mla_backend = use_mla_backend
        with (
            patch.object(kcc, "current_platform", platform),
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(
                    enable_page_major_kv_layout=False, enable_unified_memory=False
                ),
            ),
            # The first check after the guard; reaching it means the guard passed.
            patch.object(kcc, "is_float4_e2m1fn_x2", side_effect=_PastTheGuard),
        ):
            configurator._build_token_to_kv_pool(
                sizes=None,
                is_dsa_model=is_dsa_model,
                is_compressed_kv_model=False,
                req_to_token_pool=None,
            )


if __name__ == "__main__":
    unittest.main()
