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
from sglang.srt.mem_cache.memory_pool import (
    DSATokenToKVPool,
    MHATokenToKVPool,
    MLATokenToKVPool,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_NAME = "example_platform_kv"


class _PlatformMHAPool(MHATokenToKVPool):
    pass


class _PlatformMLAPool(MLATokenToKVPool):
    pass


class _Built(Exception):
    """Raised where a test stops pool construction once it got that far."""


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
        """The runner's label stays the registered name: the pool guard and the
        platform's pool both key off it."""
        platform_kv_cache_dtype.register_kv_cache_dtype(
            _NAME, storage_dtype=torch.float8_e4m3fn
        )
        self.assertEqual(self._configure(_NAME), (_NAME, torch.float8_e4m3fn))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            self._configure("unregistered_kv")

    def test_only_constructing_a_kind_the_platform_lacks_fails(self):
        """A registered dtype resolves a missing kind to a class that refuses to be
        built, so kinds a layout never builds stay harmless."""
        configurator = _configurator(kv_cache_dtype_str="fp8_e4m3")
        with patch.object(kcc, "current_platform", _platform(mha=_PlatformMHAPool)):
            self.assertIs(
                configurator._resolve_kv_pool_class(
                    kind="mla", default=MLATokenToKVPool
                ),
                MLATokenToKVPool,
            )
            platform_kv_cache_dtype.register_kv_cache_dtype(_NAME)
            configurator.kv_cache_dtype_str = _NAME
            self.assertIs(
                configurator._resolve_kv_pool_class(
                    kind="mha", default=MHATokenToKVPool
                ),
                _PlatformMHAPool,
            )
            mla_pool_class = configurator._resolve_kv_pool_class(
                kind="mla", default=MLATokenToKVPool
            )
        self.assertTrue(issubclass(mla_pool_class, MLATokenToKVPool))
        with self.assertRaisesRegex(ValueError, "pool of kind 'mla'"):
            mla_pool_class(size=8)

    def test_hybrid_dsa_swa_needs_the_platform_mla_pool_for_its_swa_layers(self):
        """The SWA side of a hybrid DSA model is an MLA pool: a platform that
        supplies only the DSA pool must be refused, not fall back to the in-tree
        MLA pool storing the registered dtype."""
        platform_kv_cache_dtype.register_kv_cache_dtype(_NAME)
        built = []
        dsa_pool = _recording_pool(DSATokenToKVPool, "dsa", built)
        mla_pool = _recording_pool(MLATokenToKVPool, "mla", built, stop=True)
        with self.assertRaisesRegex(ValueError, "pool of kind 'mla'"):
            self._build_pool(_platform(dsa=dsa_pool), use_mla_backend=True)
        self.assertEqual(built, ["dsa"])
        built.clear()
        with self.assertRaises(_Built):
            self._build_pool(
                _platform(dsa=dsa_pool, mla=mla_pool), use_mla_backend=True
            )
        self.assertEqual(built, ["dsa", "mla"])

    def test_in_tree_only_pool_paths_refuse_a_registered_dtype(self):
        """These paths build in-tree pools without consulting get_kv_pool_cls."""
        platform_kv_cache_dtype.register_kv_cache_dtype(_NAME)
        platform = _platform(mha=_PlatformMHAPool, mla=_PlatformMLAPool)
        with (
            patch.object(
                kcc.KVCacheConfigurator, "_build_ascend_swa_kv_pool", side_effect=_Built
            ),
            patch.object(
                kcc.KVCacheConfigurator,
                "_build_unlimited_ocr_v2_kv_pool",
                side_effect=_Built,
            ),
        ):
            for pool, overrides in (
                ("Ascend", {"attention_backend": "ascend"}),
                (
                    "Unlimited-OCR-v2",
                    {"is_hybrid_swa": False, "is_unlimited_ocr_v2": True},
                ),
            ):
                with (
                    self.subTest(pool=pool),
                    self.assertRaisesRegex(
                        ValueError, f"cannot use the {pool} KV pool"
                    ),
                ):
                    self._build_pool(platform, use_mla_backend=False, **overrides)
        configurator = _configurator(kv_cache_dtype_str=_NAME)
        configurator.use_mla_backend = False
        configurator.model_config = SimpleNamespace(hf_config=None)
        with (
            self.subTest(pool="NPU"),
            patch.object(kcc, "_is_npu", True),
            patch.object(kcc, "is_deepseek_dsa", return_value=False),
            self.assertRaisesRegex(ValueError, "cannot use the NPU KV pool"),
        ):
            configurator._hybrid_full_attention_pool_class(
                mha_pool_class=_PlatformMHAPool,
                mla_pool_class=_PlatformMLAPool,
                dsa_pool_class=DSATokenToKVPool,
            )

    @staticmethod
    def _build_pool(
        platform,
        *,
        use_mla_backend,
        attention_backend="triton",
        is_hybrid_swa=True,
        is_unlimited_ocr_v2=False,
    ):
        """Run the real pool dispatch for a hybrid DSA + SWA layout by default."""
        configurator = _configurator(kv_cache_dtype_str=_NAME)
        configurator.use_mla_backend = use_mla_backend
        configurator.is_hybrid_swa = is_hybrid_swa
        configurator.mambaish_config = None
        configurator.device = "cpu"
        configurator.model_config = SimpleNamespace(
            hf_config=None,
            kv_lora_rank=16,
            qk_rope_head_dim=8,
            swa_kv_lora_rank=16,
            swa_qk_rope_head_dim=8,
        )
        configurator.layer_info = SimpleNamespace(
            full_attention_layer_ids=[0], swa_attention_layer_ids=[1]
        )
        with (
            patch.object(kcc, "current_platform", platform),
            patch.object(
                kcc,
                "get_memory",
                return_value=SimpleNamespace(
                    enable_page_major_kv_layout=False, enable_unified_memory=False
                ),
            ),
            patch.object(
                kcc,
                "get_exec",
                return_value=SimpleNamespace(
                    kernel=SimpleNamespace(
                        attention_backend=attention_backend, enable_fp4_indexer=False
                    ),
                    features=SimpleNamespace(enable_memory_saver=False),
                ),
            ),
            patch.object(
                kcc, "get_schedule", return_value=SimpleNamespace(page_size=1)
            ),
            patch.object(kcc, "is_minimax_sparse", return_value=False),
            patch.object(kcc, "is_unlimited_ocr_v2", return_value=is_unlimited_ocr_v2),
            patch.object(kcc, "get_dsa_index_head_dim", return_value=8),
            patch.object(kcc, "calculate_mla_kv_cache_dim", return_value=24),
        ):
            configurator._build_token_to_kv_pool(
                sizes=SimpleNamespace(
                    max_total_num_tokens=8,
                    full_max_total_num_tokens=8,
                    swa_max_total_num_tokens=8,
                ),
                is_dsa_model=True,
                is_compressed_kv_model=False,
                req_to_token_pool=None,
            )

    @staticmethod
    def _configure(name):
        return configure_kv_cache_dtype(
            server_args_kv_cache_dtype=name,
            quant_config=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=False,
            is_dflash=False,
            speculative_draft_attention_backend=None,
        )


def _configurator(*, kv_cache_dtype_str):
    configurator = kcc.KVCacheConfigurator.__new__(kcc.KVCacheConfigurator)
    configurator.is_draft_worker = False
    configurator.kv_cache_dtype = torch.uint8
    configurator.kv_cache_dtype_str = kv_cache_dtype_str
    return configurator


def _platform(**pool_classes):
    platform = MagicMock()
    platform.get_kv_pool_cls.side_effect = lambda *, kind: pool_classes.get(kind)
    return platform


def _recording_pool(base, kind, built, *, stop=False):
    """A platform pool class that records its construction and allocates nothing."""

    class RecordingPool(base):
        def __init__(self, **kwargs):
            built.append(kind)
            if stop:
                raise _Built

    return RecordingPool


if __name__ == "__main__":
    unittest.main()
