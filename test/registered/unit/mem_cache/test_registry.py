"""Unit tests for the radix-cache registry, routing, and selection chain."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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


def _publish(testcase, **fields):
    """Install a published config for one case and restore on its cleanup."""
    from sglang.srt.runtime_context import get_context, get_server_args

    override = get_context().override_server_args(**fields)
    override.install()
    testcase.addCleanup(override.restore)
    return get_server_args()


def _make_ctx(
    testcase,
    *,
    backend=None,
    enable_streaming=False,
    enable_lmcache=False,
    is_hybrid_swa=False,
    is_hybrid_ssm=False,
    is_dsa=False,
    enable_hierarchical_cache=False,
    disable_radix_cache=False,
    effective_chunked_prefill_size=None,
    full_tokens_per_layer=None,
):
    # The factory reads the published bags for the cache-backend leaves, so the
    # fixture publishes them; the instance stays for the whole-object contract
    # `TreeCacheBuildContext` carries.
    server_args = _publish(
        testcase,
        radix_cache_backend=backend,
        enable_streaming_session=enable_streaming,
        enable_lmcache=enable_lmcache,
        enable_flexkv=False,
        enable_unified_cache_external_linker=False,
    )
    return TreeCacheBuildContext(
        server_args=server_args,
        params=MagicMock(),
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        is_dsa=is_dsa,
        enable_hierarchical_cache=enable_hierarchical_cache,
        disable_radix_cache=disable_radix_cache,
        effective_chunked_prefill_size=effective_chunked_prefill_size,
        tp_worker=MagicMock(),
        model_config=MagicMock(),
        tp_size=1,
        tp_rank=0,
        tp_group=MagicMock(),
        full_tokens_per_layer=full_tokens_per_layer,
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

        result = create_tree_cache(_make_ctx(self, backend="custom"))

        factory.assert_called_once()
        self.assertIs(result, cache)

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            create_tree_cache(_make_ctx(self, backend="not_a_real_backend"))

    @patch("sglang.srt.mem_cache.registry.default_radix_cache_factory")
    def test_unset_backend_falls_back_to_default(self, default_factory):
        cache = MagicMock()
        cache.supports_streaming_session.return_value = True
        default_factory.return_value = cache

        result = create_tree_cache(_make_ctx(self, backend=None))

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
                _make_ctx(self, backend="nonstreaming", enable_streaming=True)
            )

        session_cls.assert_called_once_with(inner)
        self.assertIs(result, session_cls.return_value)

    def test_no_streaming_wrap_when_cache_supports_it(self):
        inner = MagicMock()
        inner.supports_streaming_session.return_value = True
        register_radix_cache_backend("streaming", MagicMock(return_value=inner))

        result = create_tree_cache(
            _make_ctx(self, backend="streaming", enable_streaming=True)
        )

        self.assertIs(result, inner)


class TestDefaultRadixCacheFactory(CustomTestCase):
    """Branch coverage for the built-in radix cache selection chain.

    Each cache class is imported lazily inside the factory, so we patch
    the class at its definition site to verify routing without depending
    on each cache's real constructor or runtime state.
    """

    def test_chunk_cache_when_chunked_prefill_and_disable_radix(self):
        ctx = _make_ctx(
            self, effective_chunked_prefill_size=512, disable_radix_cache=True
        )
        with patch("sglang.srt.mem_cache.chunk_cache.ChunkCache") as ChunkCache:
            ChunkCache.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            ChunkCache.assert_called_once_with(ctx.params)
            self.assertIs(result, ChunkCache.return_value)

    def test_swa_chunk_cache_when_chunked_prefill_disable_and_hybrid_swa(self):
        ctx = _make_ctx(
            self,
            effective_chunked_prefill_size=512,
            disable_radix_cache=True,
            is_hybrid_swa=True,
        )
        with patch("sglang.srt.mem_cache.chunk_cache.SWAChunkCache") as SWAChunkCache:
            SWAChunkCache.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            SWAChunkCache.assert_called_once_with(ctx.params)
            self.assertIs(result, SWAChunkCache.return_value)

    def test_pure_swa_chunk_cache_when_chunked_prefill_disable_and_all_swa(self):
        ctx = _make_ctx(
            self,
            effective_chunked_prefill_size=512,
            disable_radix_cache=True,
            is_hybrid_swa=True,
            full_tokens_per_layer=0,
        )
        with patch(
            "sglang.srt.mem_cache.chunk_cache.PureSWAChunkCache"
        ) as PureSWAChunkCache:
            PureSWAChunkCache.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            PureSWAChunkCache.assert_called_once_with(ctx.params)
            self.assertIs(result, PureSWAChunkCache.return_value)

    def test_cpp_radix_cache_when_env_flag_set(self):
        ctx = _make_ctx(
            self,
        )
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

    def test_unified_radix_cache_is_the_default(self):
        ctx = _make_ctx(
            self,
        )
        # Shim both factory imports — each transitively loads sgl_kernel.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
                "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
            },
        ):
            result = default_radix_cache_factory(ctx)
            fake_radix.UnifiedRadixCache.assert_called_once_with(ctx.params)
            self.assertIs(result, fake_radix.UnifiedRadixCache.return_value)

    def test_unified_radix_cache_when_hierarchical(self):
        ctx = _make_ctx(self, enable_hierarchical_cache=True)
        # Full attention with hierarchical cache also uses UnifiedRadixCache.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
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

    def test_unified_radix_cache_when_hierarchical_and_hybrid_ssm(self):
        ctx = _make_ctx(self, enable_hierarchical_cache=True, is_hybrid_ssm=True)
        # Hybrid SSM with hierarchical cache now uses UnifiedRadixCache.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
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
        ctx = _make_ctx(self, enable_hierarchical_cache=True, is_hybrid_swa=True)
        # Hybrid SWA with hierarchical cache also uses UnifiedRadixCache.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
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

    def test_unified_radix_cache_when_hierarchical_and_dsa(self):
        ctx = _make_ctx(self, enable_hierarchical_cache=True, is_dsa=True)
        # DSA models (e.g. DeepSeek V3.2 / GLM-5.1) with hierarchical cache
        # use UnifiedRadixCache.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
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

    def test_unified_radix_cache_with_mori_external_linker(self):
        from sglang.srt.mem_cache.storage.umbp import umbp_direct_linker

        ctx = _make_ctx(self)
        object.__setattr__(
            ctx.server_args, "enable_unified_cache_external_linker", True
        )
        object.__setattr__(
            ctx.server_args, "unified_cache_external_linker_backend", "mori"
        )
        self.assertTrue(ctx.server_args.enable_unified_cache_external_linker)
        self.assertEqual(ctx.server_args.unified_cache_external_linker_backend, "mori")
        fake_components = MagicMock()
        fake_components.ComponentType.FULL = "full"
        fake_radix = MagicMock()
        cache = fake_radix.UnifiedRadixCache.return_value
        cache.components = ("full",)
        counter = MagicMock(name="layer_done_counter")
        cache.linker.layer_done_counter = counter
        linker = MagicMock(name="linker")

        with (
            patch.dict(
                "sys.modules",
                {
                    "sglang.srt.mem_cache.unified_cache.components": fake_components,
                    "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
                },
            ),
            patch.object(
                umbp_direct_linker,
                "UMBPDirectLinker",
                return_value=linker,
            ) as linker_cls,
        ):
            result = default_radix_cache_factory(ctx)

        linker_cls.assert_called_once_with(
            ctx.server_args,
            ctx.params,
            components={"full"},
        )
        cache.init_cache_linker.assert_called_once_with(linker)
        ctx.params.token_to_kv_pool_allocator.get_kvcache.return_value.register_layer_transfer_counter.assert_called_once_with(
            counter
        )
        ctx.tp_worker.register_hicache_layer_transfer_counter.assert_called_once_with(
            counter
        )
        self.assertIs(result, cache)

    def test_swa_radix_cache_when_hybrid_swa(self):
        ctx = _make_ctx(self, is_hybrid_swa=True)
        # SWA hybrid models now default to the unified radix tree.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
                "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
            },
        ):
            result = default_radix_cache_factory(ctx)
            fake_radix.UnifiedRadixCache.assert_called_once_with(ctx.params)
            self.assertIs(result, fake_radix.UnifiedRadixCache.return_value)

    def test_pure_swa_radix_cache_when_all_swa(self):
        ctx = _make_ctx(self, is_hybrid_swa=True, full_tokens_per_layer=0)
        with patch(
            "sglang.srt.mem_cache.pure_swa_radix_cache.PureSWARadixCache"
        ) as PureSWA:
            PureSWA.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            PureSWA.assert_called_once_with(params=ctx.params)
            self.assertIs(result, PureSWA.return_value)

    def test_mamba_radix_cache_when_hybrid_ssm(self):
        ctx = _make_ctx(self, is_hybrid_ssm=True)
        # Mamba hybrid models now default to the unified radix tree.
        fake_components = MagicMock()
        fake_radix = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
                "sglang.srt.mem_cache.unified_radix_cache": fake_radix,
            },
        ):
            result = default_radix_cache_factory(ctx)
            fake_radix.UnifiedRadixCache.assert_called_once_with(ctx.params)
            self.assertIs(result, fake_radix.UnifiedRadixCache.return_value)

    def test_lmc_radix_cache_when_enable_lmcache(self):
        ctx = _make_ctx(self, enable_lmcache=True)
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


if __name__ == "__main__":
    unittest.main()
