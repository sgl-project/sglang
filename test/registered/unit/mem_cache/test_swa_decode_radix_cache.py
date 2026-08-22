"""Compatibility tests for device-only SWA decode-side radix reuse."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.mem_cache.kv_cache_builder import (
    _validate_decode_radix_cache_compatibility,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class _TailAllocator:
    page_size = 256

    def __init__(self, kv_cache):
        self._kv_cache = kv_cache

    def get_kvcache(self):
        return self._kv_cache

    def alloc_extend_swa_tail(self, *args, **kwargs):
        raise AssertionError("validation must not allocate")


def _model(*, dsv4: bool = False, swa_compress: bool = False):
    return SimpleNamespace(
        is_deepseek_v4_arch=dsv4,
        is_hybrid_swa_compress=swa_compress,
    )


def _unified_tree(*components: ComponentType):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_components = components
    return cache


def _dsv4_pool(*, unified_kv: bool):
    return SimpleNamespace(_unified_kv=unified_kv)


class TestSWADecodeRadixCompatibility(unittest.TestCase):
    def _validate(
        self,
        *,
        model=None,
        tree_cache=None,
        allocator=None,
        is_hybrid_swa=True,
        is_hybrid_ssm=False,
        enable_hierarchical_cache=False,
    ):
        _validate_decode_radix_cache_compatibility(
            model_config=model or _model(),
            tree_cache=tree_cache
            or _unified_tree(ComponentType.FULL, ComponentType.SWA),
            token_to_kv_pool_allocator=allocator or _TailAllocator(object()),
            is_hybrid_swa=is_hybrid_swa,
            is_hybrid_ssm=is_hybrid_ssm,
            enable_hierarchical_cache=enable_hierarchical_cache,
        )

    def test_selected_unified_cache_replaces_legacy_env_gate(self):
        with envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.override(False):
            self._validate()

    def test_non_unified_cache_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires UnifiedRadixCache"):
            self._validate(tree_cache=object())

    def test_hierarchical_cache_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "device-resident"):
            self._validate(enable_hierarchical_cache=True)

    def test_swa_compress_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "SWA-compress"):
            self._validate(model=_model(swa_compress=True))

    def test_hybrid_ssm_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mamba/SSM"):
            self._validate(is_hybrid_ssm=True)

    def test_dsv4_rocm_unified_kv_is_allowed(self):
        self._validate(
            model=_model(dsv4=True),
            allocator=_TailAllocator(_dsv4_pool(unified_kv=True)),
        )

    def test_dsv4_non_unified_kv_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unified-KV path"):
            self._validate(
                model=_model(dsv4=True),
                allocator=_TailAllocator(_dsv4_pool(unified_kv=False)),
            )

    def test_dsv4_extra_radix_component_is_rejected(self):
        tree_cache = _unified_tree(
            ComponentType.FULL,
            ComponentType.SWA,
            ComponentType.C128,
        )
        with self.assertRaisesRegex(ValueError, "FULL\\+SWA"):
            self._validate(
                model=_model(dsv4=True),
                tree_cache=tree_cache,
                allocator=_TailAllocator(_dsv4_pool(unified_kv=True)),
            )


if __name__ == "__main__":
    unittest.main()
