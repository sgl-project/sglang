"""CPU-only unit tests for the platform KV seam.

Pins the resolution order of KV pool construction:
SRTPlatform.build_kv_pool(request) first, then the deprecated
get_*_kv_pool_cls class hooks (out-of-tree platforms only), then the
in-tree defaults — both for standalone pools
(KVCacheConfigurator._build_platform_kv_pool) and for the full-attention
leaf of hybrid-linear composites (_build_platform_hybrid_full_kv_pool).
Also pins the base-class contract that every hook defaults to None
("no platform opinion"), which is what keeps in-tree CUDA/ROCm behavior
untouched by the seam.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
from sglang.srt.mem_cache.kv_pool_request import KVPoolRequest
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

_CONFIGURATOR_MODULE = "sglang.srt.mem_cache.kv_cache_configurator"


def _make_request(**overrides) -> KVPoolRequest:
    fields = dict(
        kind="mha",
        size=128,
        page_size=1,
        dtype=torch.float16,
        device="cpu",
        layer_num=2,
        start_layer=0,
        end_layer=2,
        enable_memory_saver=False,
        head_num=4,
        head_dim=64,
    )
    fields.update(overrides)
    return KVPoolRequest(**fields)


def _fake_platform(*, pool=None, out_of_tree=False, legacy_cls=None):
    platform = SimpleNamespace()
    platform.build_kv_pool = MagicMock(return_value=pool)
    platform.is_out_of_tree = MagicMock(return_value=out_of_tree)
    platform.get_mha_kv_pool_cls = MagicMock(return_value=legacy_cls)
    platform.get_mla_kv_pool_cls = MagicMock(return_value=legacy_cls)
    platform.get_dsa_kv_pool_cls = MagicMock(return_value=legacy_cls)
    return platform


def _fake_configurator(*, mambaish_config=None, use_mla_backend=False):
    """A configurator double for the seam methods, bound to the real logic
    under test and MagicMocks for the heavyweight collaborators."""
    fake = SimpleNamespace()
    fake.mambaish_config = mambaish_config
    fake.use_mla_backend = use_mla_backend
    fake.layer_info = SimpleNamespace(
        num_effective_layers=2, start_layer=0, end_layer=2
    )
    fake.pool_page_size = 1
    fake.kv_cache_dtype = torch.float16
    fake.device = "cpu"
    fake.post_capture_kv_active = False
    fake.model_config = SimpleNamespace(
        get_num_kv_heads=lambda tp, dcp: 4,
        head_dim=64,
        kv_lora_rank=8,
        qk_rope_head_dim=16,
    )
    fake._kv_pool_kind = KVCacheConfigurator._kv_pool_kind.__get__(fake)
    fake._make_kv_pool_request = MagicMock(return_value=_make_request())
    fake._build_legacy_oot_kv_pool = MagicMock(return_value=None)
    return fake


_SIZES = SimpleNamespace(max_total_num_tokens=128, swa_max_total_num_tokens=None)


class TestPlatformInterfaceDefaults(CustomTestCase):
    def test_every_kv_hook_defaults_to_no_opinion(self):
        platform = SRTPlatform()
        self.assertIsNone(platform.build_kv_pool(request=_make_request()))
        self.assertIsNone(platform.get_mha_kv_pool_cls())
        self.assertIsNone(platform.get_mla_kv_pool_cls())
        self.assertIsNone(platform.get_dsa_kv_pool_cls())
        self.assertIsNone(platform.get_paged_allocator_cls())
        self.assertIsNone(platform.get_graph_runner_cls())
        self.assertIsNone(platform.get_piecewise_backend_cls())


class TestKVPoolKind(CustomTestCase):
    def test_kind_mapping(self):
        fake = _fake_configurator(use_mla_backend=True)
        self.assertEqual(fake._kv_pool_kind(is_dsa_model=True), "dsa")
        self.assertEqual(fake._kv_pool_kind(is_dsa_model=False), "mla")
        fake = _fake_configurator(use_mla_backend=False)
        self.assertEqual(fake._kv_pool_kind(is_dsa_model=False), "mha")


class TestBuildPlatformKVPool(CustomTestCase):
    def _run(self, fake, platform):
        with patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform):
            return KVCacheConfigurator._build_platform_kv_pool(
                fake, sizes=_SIZES, is_dsa_model=False
            )

    def test_platform_pool_wins(self):
        pool = object()
        fake = _fake_configurator()
        platform = _fake_platform(pool=pool, out_of_tree=True)
        self.assertIs(self._run(fake, platform), pool)
        fake._build_legacy_oot_kv_pool.assert_not_called()

    def test_in_tree_platform_falls_through(self):
        fake = _fake_configurator()
        platform = _fake_platform(pool=None, out_of_tree=False)
        self.assertIsNone(self._run(fake, platform))
        fake._build_legacy_oot_kv_pool.assert_not_called()

    def test_oot_platform_falls_back_to_legacy_hooks(self):
        legacy_pool = object()
        fake = _fake_configurator()
        fake._build_legacy_oot_kv_pool = MagicMock(return_value=legacy_pool)
        platform = _fake_platform(pool=None, out_of_tree=True)
        self.assertIs(self._run(fake, platform), legacy_pool)
        fake._build_legacy_oot_kv_pool.assert_called_once_with(
            kind="mha", max_total_num_tokens=128, is_dsa_model=False
        )

    def test_mambaish_models_skip_the_standalone_seam(self):
        fake = _fake_configurator(mambaish_config=object())
        platform = _fake_platform(pool=object(), out_of_tree=True)
        self.assertIsNone(self._run(fake, platform))
        platform.build_kv_pool.assert_not_called()


class TestLegacyOOTBuilders(CustomTestCase):
    def test_missing_legacy_hook_means_in_tree_default(self):
        platform = _fake_platform(legacy_cls=None)
        with patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform):
            self.assertIsNone(
                KVCacheConfigurator._build_oot_mha_kv_pool(
                    None, max_total_num_tokens=128
                )
            )


class TestHybridFullAttentionLeaf(CustomTestCase):
    def _run(self, fake, platform):
        with (
            patch(f"{_CONFIGURATOR_MODULE}.current_platform", platform),
            patch(
                f"{_CONFIGURATOR_MODULE}.get_exec",
                return_value=SimpleNamespace(
                    features=SimpleNamespace(enable_memory_saver=False)
                ),
            ),
            patch(
                f"{_CONFIGURATOR_MODULE}.get_spec",
                return_value=SimpleNamespace(speculative_algorithm=None),
            ),
            patch(
                f"{_CONFIGURATOR_MODULE}.get_parallel",
                return_value=SimpleNamespace(attn_tp_size=1, attn_dcp_size=1),
            ),
        ):
            return KVCacheConfigurator._build_platform_hybrid_full_kv_pool(
                fake,
                max_total_num_tokens=128,
                layer_num=2,
                quant_method=None,
            )

    def test_platform_leaf_pool_wins(self):
        leaf = object()
        fake = _fake_configurator()
        platform = _fake_platform(pool=leaf, out_of_tree=True)
        self.assertIs(self._run(fake, platform), leaf)

    def test_oot_legacy_class_is_constructed_with_leaf_dims(self):
        legacy_cls = MagicMock(name="OOTMHAPool")
        fake = _fake_configurator()
        platform = _fake_platform(pool=None, out_of_tree=True, legacy_cls=legacy_cls)
        result = self._run(fake, platform)
        self.assertIs(result, legacy_cls.return_value)
        legacy_cls.assert_called_once_with(
            size=128,
            page_size=1,
            dtype=torch.float16,
            head_num=4,
            head_dim=64,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
            enable_kv_cache_copy=False,
        )

    def test_in_tree_platform_defers_to_composite_default(self):
        fake = _fake_configurator()
        platform = _fake_platform(pool=None, out_of_tree=False)
        self.assertIsNone(self._run(fake, platform))


if __name__ == "__main__":
    unittest.main()
