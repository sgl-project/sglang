"""Compatibility tests for device-only SWA decode-side radix reuse."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import kv_cache_builder
from sglang.srt.mem_cache.kv_cache_builder import (
    _validate_decode_radix_cache_compatibility,
    _validate_decode_radix_cache_prebuild_compatibility,
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
        model = model or _model()
        tree_cache = tree_cache or _unified_tree(ComponentType.FULL, ComponentType.SWA)
        allocator = allocator or _TailAllocator(object())
        _validate_decode_radix_cache_prebuild_compatibility(
            model_config=model,
            token_to_kv_pool_allocator=allocator,
            is_hybrid_swa=is_hybrid_swa,
            is_hybrid_ssm=is_hybrid_ssm,
            enable_hierarchical_cache=enable_hierarchical_cache,
        )
        _validate_decode_radix_cache_compatibility(
            model_config=model,
            tree_cache=tree_cache,
            is_hybrid_swa=is_hybrid_swa,
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

    def test_hierarchical_cache_rejection_precedes_tree_construction(self):
        worker = MagicMock()
        worker.is_hybrid_swa = True
        worker.sliding_window_size = 128
        worker.get_tokens_per_layer_info.return_value = (1024, 256)
        worker.get_memory_pool.return_value = (object(), SimpleNamespace(page_size=64))
        worker.model_runner.model_config = MagicMock()
        worker.model_runner.mtp_draft_device_pools = []

        model = _model()
        model.is_multimodal = False
        model.hf_config = MagicMock()

        with (
            patch.object(
                kv_cache_builder, "get_resolved_model_impl", return_value=object()
            ),
            patch.object(kv_cache_builder, "linear_attn_model_spec", return_value=None),
            patch.object(kv_cache_builder, "hybrid_gdn_config", return_value=None),
            patch.object(kv_cache_builder, "mamba2_config", return_value=None),
            patch.object(kv_cache_builder, "kimi_linear_config", return_value=None),
            patch.object(
                kv_cache_builder, "hybrid_lightning_config", return_value=None
            ),
            patch.object(kv_cache_builder, "is_deepseek_dsa", return_value=False),
            patch.object(
                kv_cache_builder,
                "resolve_decode_retraction_backup",
                return_value="cpu_tensor",
            ),
            patch.object(
                kv_cache_builder,
                "get_disagg",
                return_value=SimpleNamespace(
                    disaggregation_decode_enable_radix_cache=True,
                    disaggregation_mode="decode",
                ),
            ),
            patch.object(
                kv_cache_builder,
                "get_memory",
                return_value=SimpleNamespace(disable_radix_cache=False),
            ),
            patch.object(kv_cache_builder, "create_tree_cache") as create_tree_cache,
        ):
            with self.assertRaisesRegex(ValueError, "device-resident"):
                kv_cache_builder.build_kv_cache(
                    server_args=MagicMock(),
                    model_config=model,
                    tp_worker=worker,
                    page_size=64,
                    spec_algorithm=MagicMock(),
                    attn_tp_cpu_group=MagicMock(),
                    tp_cpu_group=MagicMock(),
                    attn_cp_cpu_group=MagicMock(),
                    enable_metrics=False,
                    enable_kv_cache_events=False,
                    ps=MagicMock(),
                    tp_group=MagicMock(),
                    pp_group=MagicMock(),
                    enable_hierarchical_cache=True,
                )

        create_tree_cache.assert_not_called()

    def test_swa_compress_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "SWA-compress"):
            self._validate(model=_model(swa_compress=True))

    def test_hybrid_ssm_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "Mamba/SSM"):
            self._validate(is_hybrid_ssm=True)

    def test_page_size_one_uses_non_tail_fallback(self):
        self._validate(allocator=SimpleNamespace(page_size=1))

    def test_paged_allocator_without_swa_tail_support_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "page_size > 1"):
            self._validate(allocator=SimpleNamespace(page_size=64))

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
