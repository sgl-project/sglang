"""Unit tests for the radix-cache registry, routing, and selection chain."""

from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.kv.router import STATE_PART, KVLayerRouter
from sglang.srt.mem_cache.registry import (
    _RADIX_CACHE_REGISTRY,
    TreeCacheBuildContext,
    create_tree_cache,
    create_unified_radix_cache,
    default_radix_cache_factory,
    get_radix_cache_factory,
    prepare_radix_cache_worker,
    register_radix_cache_backend,
    registered_radix_cache_backends,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.test_utils import CustomTestCase, enter_override


def _publish(testcase, **fields):
    """Install a published config for one case and restore on its cleanup."""
    from sglang.srt.runtime_context import get_context, get_server_args

    override = get_context().override_server_args(**fields)
    override.install()
    testcase.addCleanup(override.restore)
    return get_server_args()


def _slots(ctx):
    return [(slot.component, slot.family) for slot in ctx.params.tree_slots]


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
    full_tokens_per_layer=None,
    enable_kv_cache_sharding=False,
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
        enable_kv_cache_sharding=enable_kv_cache_sharding,
    )
    params = MagicMock()
    # Static pools: the tree's slots follow the pool shape.
    params.token_to_kv_pool_allocator.storage = None
    params.sliding_window_size = 128
    params.draft_swa_window = 0
    return TreeCacheBuildContext(
        server_args=server_args,
        params=params,
        is_hybrid_swa=is_hybrid_swa,
        is_hybrid_ssm=is_hybrid_ssm,
        is_dsa=is_dsa,
        enable_hierarchical_cache=enable_hierarchical_cache,
        disable_radix_cache=disable_radix_cache,
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

    def test_duplicate_registration_preserves_factory_and_preparer(self):
        factory, preparer = MagicMock(), MagicMock()
        register_radix_cache_backend("dupe", factory, prepare_worker=preparer)
        with self.assertRaises(ValueError):
            register_radix_cache_backend(
                "dupe", MagicMock(), prepare_worker=MagicMock()
            )
        self.assertIs(get_radix_cache_factory("dupe"), factory)
        _publish(self, radix_cache_backend="dupe")
        worker = object()
        prepare_radix_cache_worker(worker, True)
        preparer.assert_called_once_with(worker, True)

    def test_registration_does_not_read_configuration_or_prepare_worker(self):
        factory, preparer = MagicMock(), MagicMock()
        with patch(
            "sglang.srt.mem_cache.registry.get_memory",
            side_effect=AssertionError("configuration is not published yet"),
        ):
            register_radix_cache_backend(
                "early_plugin", factory, prepare_worker=preparer
            )
        self.assertIs(get_radix_cache_factory("early_plugin"), factory)
        factory.assert_not_called()
        preparer.assert_not_called()

    def test_invalid_preparer_does_not_reserve_backend_name(self):
        with self.assertRaisesRegex(TypeError, "prepare_worker must be callable"):
            register_radix_cache_backend(
                "invalid", MagicMock(), prepare_worker=object()
            )
        self.assertIsNone(get_radix_cache_factory("invalid"))
        self.assertNotIn("invalid", registered_radix_cache_backends())
        factory, preparer = MagicMock(), MagicMock()
        register_radix_cache_backend("invalid", factory, prepare_worker=preparer)
        self.assertIs(get_radix_cache_factory("invalid"), factory)
        _publish(self, radix_cache_backend="invalid")
        worker = object()
        prepare_radix_cache_worker(worker, False)
        preparer.assert_called_once_with(worker, False)


class TestPrepareRadixCacheWorker(_RegistryIsolationMixin, CustomTestCase):
    def test_uses_only_published_backend_before_cache_construction(self):
        selected, other = MagicMock(), MagicMock()
        factory = MagicMock()
        register_radix_cache_backend("selected", factory, prepare_worker=selected)
        register_radix_cache_backend("other", MagicMock(), prepare_worker=other)
        _publish(self, radix_cache_backend="selected")
        worker = SimpleNamespace(
            server_args=SimpleNamespace(radix_cache_backend="other")
        )

        prepare_radix_cache_worker(worker, False)

        selected.assert_called_once_with(worker, False)
        other.assert_not_called()
        factory.assert_not_called()

    def test_default_and_factory_only_backends_need_no_preparation(self):
        unused = MagicMock()
        register_radix_cache_backend("unused", MagicMock(), prepare_worker=unused)
        factory = MagicMock()
        register_radix_cache_backend("factory_only", factory)
        for name in (None, "factory_only"):
            with self.subTest(backend=name):
                _publish(self, radix_cache_backend=name)
                prepare_radix_cache_worker(object(), True)
        unused.assert_not_called()
        factory.assert_not_called()

    def test_prepares_once_before_all_runners_on_every_worker_call(self):
        events = []
        register_radix_cache_backend(
            "ordered",
            MagicMock(),
            prepare_worker=lambda worker, capture: events.append((worker, capture)),
        )
        _publish(self, radix_cache_backend="ordered")
        for worker_number in range(2):
            runners = [MagicMock(), MagicMock()]
            worker = SimpleNamespace(model_runner=runners[0], model_runner_list=runners)
            for index, runner in enumerate(runners):
                runner.init_cuda_graphs.side_effect = (
                    lambda *, capture_decode_cuda_graph, index=index: events.append(
                        (worker_number, index, capture_decode_cuda_graph)
                    )
                )
            for capture in (True, False, True):
                with self.subTest(worker=worker_number, capture=capture):
                    events.clear()
                    TpModelWorker.init_cuda_graphs(
                        worker, capture_decode_cuda_graph=capture
                    )
                    self.assertEqual(
                        events,
                        [
                            (worker, capture),
                            (worker_number, 0, capture),
                            (worker_number, 1, capture),
                        ],
                    )

    def test_unknown_backend_and_callback_failure_prevent_all_runner_calls(self):
        failure = RuntimeError("backend preparation failed")
        register_radix_cache_backend(
            "failing", MagicMock(), prepare_worker=MagicMock(side_effect=failure)
        )
        for name, error in (("unknown", ValueError), ("failing", RuntimeError)):
            for capture in (True, False):
                with self.subTest(backend=name, capture=capture):
                    _publish(self, radix_cache_backend=name)
                    runners = [MagicMock(), MagicMock()]
                    worker = SimpleNamespace(
                        model_runner=runners[0], model_runner_list=runners
                    )
                    with self.assertRaises(error) as caught:
                        TpModelWorker.init_cuda_graphs(
                            worker, capture_decode_cuda_graph=capture
                        )
                    if name == "failing":
                        self.assertIs(caught.exception, failure)
                    for runner in runners:
                        runner.init_cuda_graphs.assert_not_called()


class TestCreateTreeCacheRouting(_RegistryIsolationMixin, CustomTestCase):
    def test_dispatches_to_registered_factory(self):
        cache = MagicMock(spec=UnifiedRadixCache)
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
        cache = MagicMock(spec=UnifiedRadixCache)
        default_factory.return_value = cache

        result = create_tree_cache(_make_ctx(self, backend=None))

        default_factory.assert_called_once()
        self.assertIs(result, cache)

    def test_streaming_rejected_on_non_unified_cache(self):
        inner = MagicMock()
        register_radix_cache_backend("nonstreaming", MagicMock(return_value=inner))

        with self.assertRaisesRegex(NotImplementedError, "not verified"):
            create_tree_cache(
                _make_ctx(self, backend="nonstreaming", enable_streaming=True)
            )

    def test_swa_component_accepts_hisparse_allocator(self):
        # Disabled DeepSeek V4 HiSparse routes to UnifiedRadixCache, whose SWA
        # component must accept the HiSparse allocator.
        from sglang.srt.mem_cache.allocator.hisparse import (
            DeepSeekV4HiSparseTokenToKVPoolAllocator,
        )
        from sglang.srt.mem_cache.kv.plan import TREE_WINDOW, TreeSlot
        from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent

        params = CacheInitParams(
            disable=False,
            req_to_token_pool=None,
            token_to_kv_pool_allocator=MagicMock(
                spec=DeepSeekV4HiSparseTokenToKVPoolAllocator
            ),
            page_size=64,
            sliding_window_size=128,
        )
        component = SWAComponent(
            MagicMock(),
            params,
            TreeSlot(component="SWA", family=TREE_WINDOW, window=128),
        )
        self.assertEqual(component.full_window_pages, 2)

    def test_pure_swa_with_disable_radix_skips_storage_backends(self):
        ctx = _make_ctx(
            self, disable_radix_cache=True, is_hybrid_swa=True, full_tokens_per_layer=0
        )
        enter_override(
            self,
            get_context().override_server_args(
                enable_unified_cache_external_linker=True
            ),
        )
        with patch(
            "sglang.srt.mem_cache.pure_swa_radix_cache.PureSWARadixCache"
        ) as PureSWARadixCache:
            PureSWARadixCache.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            PureSWARadixCache.assert_called_once_with(params=ctx.params)
            self.assertIs(result, PureSWARadixCache.return_value)

    def test_mamba_rejected_on_cache_without_mamba(self):
        inner = MagicMock()
        inner.supports_mamba.return_value = False
        register_radix_cache_backend("nomamba", MagicMock(return_value=inner))

        with self.assertRaisesRegex(NotImplementedError, "not verified"):
            create_tree_cache(_make_ctx(self, backend="nomamba", is_hybrid_ssm=True))

    def test_unified_radix_cache_when_hierarchical(self):
        ctx = _make_ctx(self, enable_hierarchical_cache=True)
        enter_override(
            self,
            get_context().override_server_args(
                enable_unified_cache_external_linker=True
            ),
        )
        # HiCache takes precedence when both storage backends are requested.
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
            result.init_cache_linker.assert_not_called()
            self.assertIs(result, fake_radix.UnifiedRadixCache.return_value)

    def _build_with_mori_linker(self, linker_pools):
        from sglang.srt.mem_cache.storage.umbp import umbp_direct_linker

        has_state = PoolName.MAMBA in linker_pools
        ctx = _make_ctx(self, is_hybrid_ssm=has_state)
        ctx.params.req_to_token_pool = SimpleNamespace()
        # The factory reads the linker settings from the bags.
        enter_override(
            self,
            get_context().override_server_args(
                enable_unified_cache_external_linker=True,
                unified_cache_external_linker_backend="mori",
            ),
        )
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_external_cache_linker=False, write_through_threshold=256
        )
        cache.tree_components = (ComponentType.FULL,) + (
            (ComponentType.MAMBA,) if has_state else ()
        )
        cache.components = {component: object() for component in cache.tree_components}
        cache._components_tuple = tuple(cache.components.values())
        counter = MagicMock(name="layer_done_counter")
        linker = SimpleNamespace(pools=linker_pools, layer_done_counter=counter)
        kvcache = ctx.params.token_to_kv_pool_allocator.get_kvcache.return_value
        kvcache.start_layer = 4
        kvcache.layer_transfer_counter = None
        kvcache.register_layer_transfer_counter.side_effect = lambda value: setattr(
            kvcache, "layer_transfer_counter", value
        )
        kvcache.layer_router = KVLayerRouter.from_layer_maps(
            kvcache, [("full", object(), {4: 0})]
        )
        kvcache.layer_router.bind_part(
            sub_pool="state", part=STATE_PART, view=object(), indices={4: 0}
        )

        with (
            patch("sglang.srt.runtime_context.mamba_cache_chunk_size", return_value=64),
            patch(
                "sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache",
                return_value=cache,
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
            components=set(cache.tree_components),
            extra_config={},
            forward_stream=ctx.tp_worker.model_runner.forward_stream,
        )
        self.assertIsInstance(cache.linker, UnifiedCacheLinkerWrapper)
        self.assertIs(cache.linker.cache_linker, linker)
        self.assertTrue(cache.tree_core.enable_external_cache_linker)
        self.assertIs(kvcache.layer_transfer_counter, counter)
        ctx.tp_worker.register_hicache_layer_transfer_counter.assert_called_once_with(
            counter
        )
        self.assertIs(result, cache)
        return kvcache.layer_router, counter

    def test_unified_radix_cache_with_mori_external_linker(self):
        router, counter = self._build_with_mori_linker({PoolName.KV: object()})
        self.assertIs(router.read(4, STATE_PART), router.layer(4, STATE_PART))
        counter.wait_until.assert_not_called()
        self.assertIs(router.read(4), router.layer(4))
        counter.wait_until.assert_called_once_with(0)

    def test_external_linker_backing_state_makes_state_reads_wait(self):
        router, counter = self._build_with_mori_linker(
            {PoolName.KV: object(), PoolName.MAMBA: object()}
        )
        self.assertIs(router.read(4, STATE_PART), router.layer(4, STATE_PART))
        counter.wait_until.assert_called_once_with(0)

    def test_custom_unified_cache_keeps_host_pool_setup(self):
        ctx = _make_ctx(self)
        enter_override(
            self,
            get_context().override_server_args(
                disaggregation_decode_retraction_backup="host_pool"
            ),
        )
        cache_class = MagicMock()
        result = create_unified_radix_cache(ctx, cache_class=cache_class)
        cache_class.assert_called_once_with(ctx.params)
        result.init_hicache.assert_called_once_with(ctx.server_args, ctx.params)
        ctx.tp_worker.register_hicache_layer_transfer_counter.assert_called_once_with(
            result.cache_controller.layer_done_counter
        )
        self.assertIs(result, cache_class.return_value)

    def test_pure_swa_radix_cache_when_all_swa(self):
        ctx = _make_ctx(self, is_hybrid_swa=True, full_tokens_per_layer=0)
        with patch(
            "sglang.srt.mem_cache.pure_swa_radix_cache.PureSWARadixCache"
        ) as PureSWA:
            PureSWA.return_value = MagicMock()
            result = default_radix_cache_factory(ctx)
            PureSWA.assert_called_once_with(params=ctx.params)
            self.assertIs(result, PureSWA.return_value)

    def test_lmcache_unified_radix_cache_when_enable_lmcache(self):
        ctx = _make_ctx(self, enable_lmcache=True)
        fake_module = MagicMock()
        fake_components = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache": fake_module,
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
            },
        ):
            result = default_radix_cache_factory(ctx)
            fake_module.LMCacheUnifiedRadixCache.assert_called_once_with(
                ctx.params,
                model_config=ctx.model_config,
                tp_size=ctx.tp_size,
                tp_rank=ctx.tp_rank,
                lmcache_config_file=None,
                forward_stream=ctx.tp_worker.model_runner.forward_stream,
            )
            self.assertEqual(_slots(ctx), [("FULL", "token")])
            self.assertIs(result, fake_module.LMCacheUnifiedRadixCache.return_value)

    def test_lmcache_supports_hybrid_swa_components(self):
        ctx = _make_ctx(self, enable_lmcache=True, is_hybrid_swa=True)
        fake_module = MagicMock()
        fake_components = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache": fake_module,
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
            },
        ):
            default_radix_cache_factory(ctx)

        self.assertEqual(_slots(ctx), [("FULL", "token"), ("SWA", "window")])

    def test_lmcache_supports_hybrid_ssm_components(self):
        ctx = _make_ctx(self, enable_lmcache=True, is_hybrid_ssm=True)
        fake_module = MagicMock()
        fake_components = MagicMock()
        with (
            patch.dict(
                "sys.modules",
                {
                    "sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache": fake_module,
                    "sglang.srt.mem_cache.unified_cache.components": fake_components,
                },
            ),
            patch("sglang.srt.runtime_context.mamba_cache_chunk_size", return_value=64),
        ):
            default_radix_cache_factory(ctx)

        self.assertEqual(_slots(ctx), [("FULL", "token"), ("MAMBA", "state")])

    def test_lmcache_supports_dsa_as_full_sidecar(self):
        ctx = _make_ctx(self, enable_lmcache=True, is_dsa=True)
        fake_module = MagicMock()
        fake_components = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.mem_cache.storage.lmcache.lmcache_unified_radix_cache": fake_module,
                "sglang.srt.mem_cache.unified_cache.components": fake_components,
            },
        ):
            default_radix_cache_factory(ctx)

        self.assertEqual(_slots(ctx), [("FULL", "token")])


if __name__ == "__main__":
    unittest.main()
