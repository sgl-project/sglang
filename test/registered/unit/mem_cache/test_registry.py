"""Unit tests for the radix-cache registry, routing, and selection chain."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.registry import (
    _RADIX_CACHE_REGISTRY,
    TreeCacheBuildContext,
    create_tree_cache,
    default_radix_cache_factory,
    get_radix_cache_factory,
    register_radix_cache_backend,
    registered_radix_cache_backends,
)
from sglang.test.test_utils import CustomTestCase


def _make_ctx(
    *,
    backend=None,
    enable_streaming=False,
    enable_lmcache=False,
    is_hybrid_swa=False,
    is_hybrid_ssm=False,
    enable_hierarchical_cache=False,
    disable_radix_cache=False,
    effective_chunked_prefill_size=None,
):
    server_args = MagicMock()
    server_args.radix_cache_backend = backend
    server_args.enable_streaming_session = enable_streaming
    server_args.enable_lmcache = enable_lmcache
    return TreeCacheBuildContext(
        server_args=server_args,
        params=MagicMock(),
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        enable_hierarchical_cache=enable_hierarchical_cache,
        disable_radix_cache=disable_radix_cache,
        effective_chunked_prefill_size=effective_chunked_prefill_size,
        tp_worker=MagicMock(),
        model_config=MagicMock(),
        tp_size=1,
        tp_rank=0,
        tp_group=MagicMock(),
    )


class _RegistryIsolationMixin:
    """Restore the global registry around each test so registrations
    from one test don't leak into the next.
    """

    def setUp(self):
        super().setUp()
        self._registry_snapshot = dict(_RADIX_CACHE_REGISTRY)

    def tearDown(self):
        _RADIX_CACHE_REGISTRY.clear()
        _RADIX_CACHE_REGISTRY.update(self._registry_snapshot)
        super().tearDown()


class TestRegisterRadixCacheBackend(_RegistryIsolationMixin, CustomTestCase):
    def test_register_then_lookup(self):
        factory = MagicMock()
        register_radix_cache_backend("oss_unit_test", factory)
        self.assertIs(get_radix_cache_factory("oss_unit_test"), factory)
        self.assertIn("oss_unit_test", registered_radix_cache_backends())

    def test_lookup_unknown_returns_none(self):
        self.assertIsNone(get_radix_cache_factory("definitely_not_registered"))

    def test_empty_name_raises(self):
        with self.assertRaises(ValueError):
            register_radix_cache_backend("", MagicMock())

    def test_whitespace_only_name_raises(self):
        with self.assertRaises(ValueError):
            register_radix_cache_backend("   ", MagicMock())

    def test_duplicate_registration_raises(self):
        register_radix_cache_backend("dupe", MagicMock())
        with self.assertRaises(ValueError):
            register_radix_cache_backend("dupe", MagicMock())


class TestCreateTreeCacheRouting(_RegistryIsolationMixin, CustomTestCase):
    def test_dispatches_to_registered_factory(self):
        cache = MagicMock()
        cache.supports_streaming_session.return_value = True
        factory = MagicMock(return_value=cache)
        register_radix_cache_backend("custom", factory)

        result = create_tree_cache(_make_ctx(backend="custom"))

        factory.assert_called_once()
        self.assertIs(result, cache)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_tree_cache(_make_ctx(backend="not_a_real_backend"))

    @patch("sglang.srt.mem_cache.registry.default_radix_cache_factory")
    def test_unset_backend_falls_back_to_default(self, default_factory):
        cache = MagicMock()
        cache.supports_streaming_session.return_value = True
        default_factory.return_value = cache

        result = create_tree_cache(_make_ctx(backend=None))

        default_factory.assert_called_once()
        self.assertIs(result, cache)

    def test_streaming_wrap_when_cache_does_not_support_it(self):
        inner = MagicMock()
        inner.supports_streaming_session.return_value = False
        register_radix_cache_backend("nonstreaming", MagicMock(return_value=inner))

        with patch(
            "sglang.srt.session.streaming_session.StreamingSession"
        ) as session_cls:
            session_cls.return_value = MagicMock(name="wrapped")
            result = create_tree_cache(
                _make_ctx(backend="nonstreaming", enable_streaming=True)
            )

        session_cls.assert_called_once_with(inner)
        self.assertIs(result, session_cls.return_value)

    def test_no_streaming_wrap_when_cache_supports_it(self):
        inner = MagicMock()
        inner.supports_streaming_session.return_value = True
        register_radix_cache_backend("streaming", MagicMock(return_value=inner))

        result = create_tree_cache(
            _make_ctx(backend="streaming", enable_streaming=True)
        )

        self.assertIs(result, inner)


class TestDefaultRadixCacheFactory(CustomTestCase):
    """Branch coverage for the built-in radix cache selection chain.

    Each cache class is imported lazily inside the factory, so we patch
    the class at its definition site to verify routing without depending
    on each cache's real constructor or runtime state.
    """

    def test_chunk_cache_when_chunked_prefill_and_disable_radix(self):
        ctx = _make_ctx(effective_chunked_prefill_size=512, disable_radix_cache=True)
        with patch("sglang.srt.mem_cache.chunk_cache.ChunkCache") as ChunkCache:
            ChunkCache.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            ChunkCache.assert_called_once_with(ctx.params)
            self.assertIs(result, ChunkCache.return_value)

    def test_swa_chunk_cache_when_chunked_prefill_disable_and_hybrid_swa(self):
        ctx = _make_ctx(
            effective_chunked_prefill_size=512,
            disable_radix_cache=True,
            is_hybrid_swa=True,
        )
        with patch("sglang.srt.mem_cache.chunk_cache.SWAChunkCache") as SWAChunkCache:
            SWAChunkCache.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            SWAChunkCache.assert_called_once_with(ctx.params)
            self.assertIs(result, SWAChunkCache.return_value)

    def test_cpp_radix_cache_when_env_flag_set(self):
        ctx = _make_ctx()
        # `radix_cache_cpp` requires ninja + C++ extension to import, so
        # we inject a stand-in module rather than letting patch() trigger
        # the real import.
        fake_module = MagicMock()
        with (
            patch(
                "sglang.srt.mem_cache.registry.envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {"sglang.srt.mem_cache.radix_cache_cpp": fake_module},
            ),
        ):
            result = default_radix_cache_factory(ctx)
            fake_module.RadixCacheCpp.assert_called_once_with(
                params=ctx.params, server_args=ctx.server_args
            )
            self.assertIs(result, fake_module.RadixCacheCpp.return_value)

    def test_unified_radix_cache_when_env_flag_set(self):
        ctx = _make_ctx()
        # Shim both factory imports — each transitively loads sgl_kernel.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with (
            patch(
                "sglang.srt.mem_cache.registry.envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get",
                return_value=True,
            ),
            patch.dict(
                "sys.modules",
                {
                    "sglang.srt.mem_cache.unified_cache_components": fake_components,
                    "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
                },
            ),
        ):
            result = default_radix_cache_factory(ctx)
            fake_radix.UnifiedRadixCache.assert_called_once_with(ctx.params)
            self.assertIs(result, fake_radix.UnifiedRadixCache.return_value)

    def test_hi_radix_cache_when_hierarchical(self):
        ctx = _make_ctx(enable_hierarchical_cache=True)
        # `hiradix_cache` imports `hicache_storage` and
        # `memory_pool_host`, both of which transitively load
        # `sgl_kernel`; inject a stand-in module.
        fake_module = MagicMock()
        with patch.dict(
            "sys.modules",
            {"sglang.srt.mem_cache.hiradix_cache": fake_module},
        ):
            result = default_radix_cache_factory(ctx)
            fake_module.HiRadixCache.assert_called_once_with(
                params=ctx.params, server_args=ctx.server_args
            )
            ctx.tp_worker.register_hicache_layer_transfer_counter.assert_called_once()
            self.assertIs(result, fake_module.HiRadixCache.return_value)

    def test_unified_radix_cache_when_hierarchical_and_hybrid_ssm(self):
        ctx = _make_ctx(enable_hierarchical_cache=True, is_hybrid_ssm=True)
        # Hybrid SSM with hierarchical cache now uses UnifiedRadixCache.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache_components": fake_components,
                "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
            },
        ):
            result = default_radix_cache_factory(ctx)
            fake_radix.UnifiedRadixCache.assert_called_once_with(ctx.params)
            fake_radix.UnifiedRadixCache.return_value.init_hicache.assert_called_once_with(
                ctx.server_args, ctx.params
            )
            ctx.tp_worker.register_hicache_layer_transfer_counter.assert_called_once()
            self.assertIs(result, fake_radix.UnifiedRadixCache.return_value)

    def test_unified_radix_cache_when_hierarchical_and_hybrid_swa(self):
        ctx = _make_ctx(enable_hierarchical_cache=True, is_hybrid_swa=True)
        # Hybrid SWA with hierarchical cache also uses UnifiedRadixCache.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache_components": fake_components,
                "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
            },
        ):
            result = default_radix_cache_factory(ctx)
            fake_radix.UnifiedRadixCache.assert_called_once_with(ctx.params)
            fake_radix.UnifiedRadixCache.return_value.init_hicache.assert_called_once_with(
                ctx.server_args, ctx.params
            )
            ctx.tp_worker.register_hicache_layer_transfer_counter.assert_called_once()
            self.assertIs(result, fake_radix.UnifiedRadixCache.return_value)

    def test_swa_radix_cache_when_hybrid_swa(self):
        ctx = _make_ctx(is_hybrid_swa=True)
        with patch("sglang.srt.mem_cache.swa_radix_cache.SWARadixCache") as SWA:
            SWA.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            SWA.assert_called_once_with(params=ctx.params)
            self.assertIs(result, SWA.return_value)

    def test_mamba_radix_cache_when_hybrid_ssm(self):
        ctx = _make_ctx(is_hybrid_ssm=True)
        with patch("sglang.srt.mem_cache.mamba_radix_cache.MambaRadixCache") as Mamba:
            Mamba.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            Mamba.assert_called_once_with(ctx.params)
            self.assertIs(result, Mamba.return_value)

    def test_lmc_radix_cache_when_enable_lmcache(self):
        ctx = _make_ctx(enable_lmcache=True)
        # The lmcache backend raises at import time when the `lmcache`
        # package isn't installed, so inject a stand-in module instead
        # of letting patch() trigger the real import.
        fake_module = MagicMock()
        with patch.dict(
            "sys.modules",
            {"sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache": fake_module},
        ):
            result = default_radix_cache_factory(ctx)
            fake_module.LMCRadixCache.assert_called_once_with(
                params=ctx.params,
                model_config=ctx.model_config,
                tp_size=ctx.tp_size,
                rank=ctx.tp_rank,
                tp_group=ctx.tp_group,
            )
            self.assertIs(result, fake_module.LMCRadixCache.return_value)

    def test_fallback_to_radix_cache(self):
        ctx = _make_ctx()
        with patch("sglang.srt.mem_cache.radix_cache.RadixCache") as RadixCache:
            RadixCache.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            RadixCache.assert_called_once_with(ctx.params)
            self.assertIs(result, RadixCache.return_value)


if __name__ == "__main__":
    unittest.main()
