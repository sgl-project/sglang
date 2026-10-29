"""
Unit tests for the HiCache L3 (mooncake) object-key layout reported to external
KV routers via GET /get_hicache_l3_cache_layout.

python3 -m unittest hicache.test_hicache_l3_object_layout.TestHiCacheL3ObjectLayout
"""

import asyncio
import importlib
import threading
import unittest
from contextlib import ExitStack, nullcontext
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import cache_controller
from sglang.srt.managers import data_parallel_controller as dp_controller
from sglang.srt.managers.io_struct import GetHiCacheL3LayoutReq
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeBaseStore,
    MooncakeStore,
)
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context, get_parallel, get_server_args

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _sidecar(pool_name: PoolName, hit_policy: PoolHitPolicy) -> SidecarPoolSpec:
    indices_from_pool = (
        PoolName.SWA if hit_policy is PoolHitPolicy.TRAILING_PAGES else PoolName.KV
    )
    return SidecarPoolSpec(
        pool_name=pool_name,
        indices_from_pool=indices_from_pool,
        hit_policy=hit_policy,
    )


class TestHiCacheL3ObjectLayout(CustomTestCase):
    def setUp(self):
        super().setUp()
        override = get_context().override_server_args(
            tp_size=1, pp_size=1, dp_size=2, enable_dp_attention=False
        )
        override.install()
        self.addCleanup(override.restore)

    def _store(self, *, mla_suffix="", pp_rank=0, pp_size=1, registered=()):
        store = MooncakeStore.__new__(MooncakeStore)
        # Only the storage/network boundary is replaced; key generation is real.
        store.registered_pools = {name: self._host_pool() for name in registered}
        store.mem_pool_host = self._host_pool()
        store.mem_pool_host.kv_buffer = None
        store.mla_suffix = mla_suffix
        store.mha_suffix = "0"
        store.is_mla_backend = True
        store.pp_rank = pp_rank
        store.pp_size = pp_size
        store.dcp_size = 1
        store.config_prefix = "tag_my-model"
        store.should_split_heads = False
        store.enable_storage_metrics = False
        store._use_group_semantics = False
        return store

    @staticmethod
    def _host_pool(objects_per_page=1):
        return SimpleNamespace(
            kv_buffer=object(),
            page_size=64,
            get_page_buffer_meta=lambda indices: (
                [128] * objects_per_page,
                [64] * objects_per_page,
            ),
        )

    @staticmethod
    def _scheduler(store=None, *, hierarchical=True):
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.enable_hierarchical_cache = hierarchical
        scheduler.page_size = 64
        scheduler.max_total_num_tokens = 8192
        scheduler.max_req_input_len = 4096
        scheduler.startup_time = None
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(
            storage_backend=store, storage_backend_type="mooncake" if store else None
        )
        cache.sidecar_pool_specs = []
        cache.components = {}
        scheduler.tree_cache = cache
        return scheduler

    def test_physical_primary_and_components_match_actual_write_keys(self):
        store = self._store(registered=(PoolName.SWA, PoolName.MAMBA, PoolName.INDEXER))
        store.mem_pool_host = self._host_pool()
        mamba = store.registered_pools[PoolName.MAMBA]
        mamba.conv_buffer = [object(), object()]
        mamba.temporal_state_elem_size = 1
        mamba.get_page_buffer_meta = self._host_pool(3).get_page_buffer_meta
        scheduler = self._scheduler(store)
        scheduler.tree_cache.components[ComponentType.SWA] = SimpleNamespace(
            full_window_pages=3
        )
        scheduler.tree_cache.sidecar_pool_specs = [
            _sidecar(PoolName.INDEXER, PoolHitPolicy.ALL_PAGES)
        ]

        layout = scheduler.get_hicache_l3_cache_layout(GetHiCacheL3LayoutReq()).layout
        pools = layout["hicache_object_layout"]["pools"]
        self.assertEqual([p["pool"] for p in pools], ["kv", "swa", "mamba", "indexer"])
        self.assertEqual(
            [p["hit_policy"] for p in pools],
            ["all_pages", "trailing_pages", "trailing_pages", "all_pages"],
        )
        self.assertEqual(pools[1]["trailing_pages"], 3)
        self.assertEqual(pools[2]["trailing_pages"], 1)
        self.assertNotIn("trailing_pages", pools[3])
        self.assertFalse(layout["enabled"])
        self.assertIn("Auxiliary", layout["reason"])

        written = []

        def put(keys, pointers, sizes, group_ids, session_ids=None):
            written.extend(keys)
            return [0] * len(keys)

        indices = torch.arange(64, dtype=torch.int64)
        with (
            patch.object(
                store, "_batch_exist", side_effect=lambda keys: [0] * len(keys)
            ),
            patch.object(store, "_put_batch_zero_copy_impl", side_effect=put),
        ):
            self.assertEqual(store.batch_set_v1(["page-hash"], indices), [True])
            store.batch_set_v2(
                [
                    PoolTransfer(name=name, keys=["page-hash"], host_indices=indices)
                    for name in store.registered_pools
                ]
            )
        self.assertEqual(
            written,
            [
                f"tag_my-model_page-hash{suffix}"
                for pool in pools
                for suffix in pool["suffixes"]
            ],
        )

    def test_handler_and_startup_share_enabled_and_disabled_envelopes(self):
        store = self._store(registered=(PoolName.SWA,))
        for hierarchical, backend, pp_size, mla in (
            (True, store, 1, True),
            (False, None, 1, True),
            (True, None, 1, True),
            (True, store, 2, True),
            (True, store, 1, False),
        ):
            with self.subTest(
                hierarchical=hierarchical,
                backend=bool(backend),
                pp_size=pp_size,
                mla=mla,
            ):
                store.pp_size = pp_size
                store.is_mla_backend = mla
                store.registered_pools[PoolName.SWA].v_buffer = object()
                scheduler = self._scheduler(backend, hierarchical=hierarchical)
                scheduler.tree_cache.components[ComponentType.SWA] = SimpleNamespace(
                    full_window_pages=2
                )
                reply = scheduler.get_hicache_l3_cache_layout(GetHiCacheL3LayoutReq())
                self.assertEqual(
                    reply.layout, scheduler.get_init_info()["hicache_object_layout"]
                )
                self.assertEqual(
                    reply.layout["enabled"],
                    hierarchical and backend is not None and mla,
                )
                if not reply.layout["enabled"]:
                    self.assertTrue(reply.layout["reason"])

    def test_handler_does_not_depend_on_internal_state_serializer_import(self):
        scheduler_module = importlib.import_module(Scheduler.__module__)
        store = self._store(registered=(PoolName.INDEXER,))
        # Internal-state extraction can remove the old module-level import.
        with patch.dict(scheduler_module.__dict__):
            scheduler_module.__dict__.pop("msgspec_to_builtins", None)
            for hierarchical in (False, True):
                with self.subTest(hierarchical=hierarchical):
                    scheduler = self._scheduler(store, hierarchical=hierarchical)
                    reply = scheduler.get_hicache_l3_cache_layout(
                        GetHiCacheL3LayoutReq()
                    )
                    self.assertEqual(reply.layout["enabled"], hierarchical)

    def test_cache_without_controller_does_not_break_startup(self):
        """Pure-SWA and unattached caches must return an unavailable layout."""
        scheduler = self._scheduler()
        scheduler.tree_cache.cache_controller = None
        reply = scheduler.get_hicache_l3_cache_layout(GetHiCacheL3LayoutReq())
        self.assertFalse(reply.layout["enabled"])
        self.assertIn("controller", reply.layout["reason"])
        self.assertEqual(
            scheduler.get_init_info()["hicache_object_layout"], reply.layout
        )

    def test_sidecar_window_uses_its_source_and_preserves_its_policy(self):
        store = self._store(
            registered=(PoolName.INDEXER, PoolName.DEEPSEEK_V4_C4_STATE)
        )
        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=[
                SidecarPoolSpec(
                    PoolName.INDEXER, PoolName.MAMBA, PoolHitPolicy.TRAILING_PAGES
                ),
                SidecarPoolSpec(
                    PoolName.DEEPSEEK_V4_C4_STATE, PoolName.SWA, PoolHitPolicy.ALL_PAGES
                ),
            ],
            swa_trailing_pages=7,
            page_size=64,
        )
        pools = layout["hicache_object_layout"]["pools"]
        self.assertEqual(pools[0]["trailing_pages"], 1)
        self.assertEqual(pools[1]["hit_policy"], "all_pages")
        self.assertNotIn("trailing_pages", pools[1])

    def test_invalid_window_is_explicitly_disabled(self):
        for window in (None, 0, -1):
            with self.subTest(window=window):
                layout = self._store(
                    registered=(PoolName.SWA,)
                ).describe_l3_object_layout(
                    sidecar_pool_specs=[], swa_trailing_pages=window, page_size=64
                )
                self.assertFalse(layout["enabled"])
                self.assertIn("window", layout["reason"])

    def test_draft_swa_rank_sharding_is_independent_of_mla_primary(self):
        for pool_type, enabled, suffixes in (
            (
                MHATokenToKVPoolHost,
                False,
                ["_{rank}_draft_swa_k", "_{rank}_draft_swa_v"],
            ),
            (MLATokenToKVPoolHost, True, ["__draft_swa"]),
            (DeepSeekV4PagedHostPool, True, ["__draft_swa"]),
        ):
            for rank in (0, 1):
                with self.subTest(pool_type=pool_type.__name__, rank=rank):
                    store = self._store()
                    store.mem_pool_host = MLATokenToKVPoolHost.__new__(
                        MLATokenToKVPoolHost
                    )
                    vars(store.mem_pool_host).update(vars(self._host_pool()))
                    store.mha_suffix = str(rank)
                    store.registered_pools[PoolName.DRAFT_SWA] = pool_type.__new__(
                        pool_type
                    )
                    layout = store.describe_l3_object_layout(
                        sidecar_pool_specs=[
                            _sidecar(PoolName.DRAFT_SWA, PoolHitPolicy.TRAILING_PAGES)
                        ],
                        swa_trailing_pages=2,
                        page_size=64,
                    )
                    self.assertEqual(layout["enabled"], enabled)
                    if not enabled:
                        self.assertIn("Auxiliary", layout["reason"])
                    self.assertEqual(
                        layout["hicache_object_layout"]["pools"],
                        [
                            {
                                "pool": "kv",
                                "suffixes": ["__k"],
                                "hit_policy": "all_pages",
                            },
                            {
                                "pool": "draft_swa",
                                "suffixes": [
                                    suffix.format(rank=rank) for suffix in suffixes
                                ],
                                "hit_policy": "trailing_pages",
                                "trailing_pages": 2,
                            },
                        ],
                    )

    def test_layout_reports_suffixes_prefix_and_hit_policy(self):
        store = self._store(
            registered=(PoolName.DEEPSEEK_V4_C4, PoolName.DEEPSEEK_V4_C4_STATE)
        )

        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=[
                _sidecar(PoolName.DEEPSEEK_V4_C4, PoolHitPolicy.ALL_PAGES),
                _sidecar(PoolName.DEEPSEEK_V4_C4_STATE, PoolHitPolicy.TRAILING_PAGES),
            ],
            swa_trailing_pages=2,
            page_size=64,
        )

        self.assertEqual(layout["backend"], "mooncake")
        self.assertEqual(layout["page_size"], 64)

        object_layout = layout["hicache_object_layout"]
        self.assertEqual(object_layout["key_prefix"], "tag_my-model")
        self.assertTrue(object_layout["is_mla_backend"])
        self.assertEqual(object_layout["pp_rank"], 0)
        self.assertEqual(object_layout["pp_size"], 1)
        self.assertEqual(
            object_layout["pools"],
            [
                {
                    "pool": "deepseek_v4_c4",
                    "suffixes": ["__deepseek_v4_c4"],
                    "hit_policy": "all_pages",
                },
                {
                    "pool": "deepseek_v4_c4_state",
                    "suffixes": ["__deepseek_v4_c4_state"],
                    "hit_policy": "trailing_pages",
                    "trailing_pages": 2,
                },
            ],
        )

    def test_layout_skips_pools_that_are_not_registered(self):
        store = self._store(registered=(PoolName.DEEPSEEK_V4_C4,))

        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=[
                _sidecar(PoolName.DEEPSEEK_V4_C4, PoolHitPolicy.ALL_PAGES),
                _sidecar(PoolName.DEEPSEEK_V4_C128, PoolHitPolicy.ALL_PAGES),
            ],
            swa_trailing_pages=None,
            page_size=64,
        )

        pools = layout["hicache_object_layout"]["pools"]
        self.assertEqual([entry["pool"] for entry in pools], ["deepseek_v4_c4"])

    def test_current_dsv4_pools_match_registered_write_objects(self):
        names = tuple(
            name for name in PoolName if name.value.startswith("deepseek_v4_")
        )
        store = self._store()
        with patch.object(MooncakeBaseStore, "register_buffer"):
            for name in names:
                pool = self._host_pool()
                store.register_mem_host_pool_v2(pool, name)
        specs = [
            _sidecar(
                name,
                PoolHitPolicy.TRAILING_PAGES
                if name.value.endswith("_state")
                else PoolHitPolicy.ALL_PAGES,
            )
            for name in names
        ]
        indices = torch.arange(64, dtype=torch.int64)
        for prefix in (None, "tag_dsv4-format-v2"):
            with self.subTest(prefix=prefix):
                store.config_prefix = prefix
                layout = store.describe_l3_object_layout(
                    sidecar_pool_specs=specs, swa_trailing_pages=3, page_size=64
                )
                self.assertTrue(layout["enabled"])
                objects = layout["hicache_object_layout"]
                self.assertEqual(objects["key_prefix"], prefix)
                self.assertEqual(
                    [entry["pool"] for entry in objects["pools"]], list(names)
                )
                written = []

                def put(keys, pointers, sizes, group_ids, session_ids=None):
                    written.extend(keys)
                    return [0] * len(keys)

                with (
                    patch.object(
                        store, "_batch_exist", side_effect=lambda keys: [0] * len(keys)
                    ),
                    patch.object(store, "_put_batch_zero_copy_impl", side_effect=put),
                ):
                    result = store.batch_set_v2(
                        [
                            PoolTransfer(name=name, keys=["hash"], host_indices=indices)
                            for name in names
                        ]
                    )
                self.assertEqual(set(result), set(names))
                self.assertEqual(
                    written,
                    [
                        f"{prefix + '_' if prefix is not None else ''}hash{suffix}"
                        for entry in objects["pools"]
                        for suffix in entry["suffixes"]
                    ],
                )
                for entry, spec in zip(objects["pools"], specs):
                    self.assertEqual(entry["hit_policy"], spec.hit_policy.value)
                    if spec.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                        self.assertEqual(entry["trailing_pages"], 3)

    def test_primary_multibuffer_still_describes_one_object(self):
        """A layer-first MLA page packs several buffers into one physical key."""
        store = self._store()
        store.mem_pool_host = self._host_pool(4)
        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=(), swa_trailing_pages=None, page_size=64
        )
        self.assertTrue(layout["enabled"])
        self.assertEqual(
            layout["hicache_object_layout"]["pools"],
            [{"pool": "kv", "suffixes": ["__k"], "hit_policy": "all_pages"}],
        )
        keys, pointers, sizes = store._batch_preprocess(["hash"], torch.arange(64))
        self.assertEqual(keys, ["hash__k"])
        self.assertEqual(len(pointers[0]), 4)
        self.assertEqual(len(sizes[0]), 4)

    def test_rank_sharded_draft_and_empty_registration_are_unavailable(self):
        store = self._store()
        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=(), swa_trailing_pages=None, page_size=64
        )
        self.assertFalse(layout["enabled"])
        self.assertIn("No physical", layout["reason"])
        for pool_type, enabled in (
            (MHATokenToKVPoolHost, False),
            (MLATokenToKVPoolHost, True),
        ):
            with self.subTest(pool_type=pool_type.__name__):
                store.registered_pools[PoolName.DRAFT] = pool_type.__new__(pool_type)
                layout = store.describe_l3_object_layout(
                    sidecar_pool_specs=(), swa_trailing_pages=None, page_size=64
                )
                self.assertEqual(layout["enabled"], enabled)
                if not enabled:
                    self.assertIn("Auxiliary", layout["reason"])

    def test_deepseek_v4_uses_rank_replicated_storage_config(self):
        """Compressed MLA must remain eligible without checking model names."""
        controller = cache_controller.HiCacheController.__new__(
            cache_controller.HiCacheController
        )
        controller.mem_pool_device = DeepSeekV4TokenToKVPool.__new__(
            DeepSeekV4TokenToKVPool
        )
        controller.mem_pool_host = SimpleNamespace(
            layout="page_first", storage_format_tag=None
        )
        controller.storage_host_pool = controller.mem_pool_host
        controller.enable_storage_metrics = False
        controller.get_attn_cp_rank_and_size = lambda: (0, 1)
        with (
            get_context().override_server_args(tp_size=2, pp_size=1),
            get_parallel().override(tp_rank=0, pp_rank=0, attn_dcp_rank=0),
        ):
            config = controller._generate_storage_config()
        self.assertTrue(config.is_mla_model)
        store = self._store(registered=(PoolName.DEEPSEEK_V4_C4,))
        store.is_mla_backend = config.is_mla_model
        scheduler = self._scheduler(store)
        reply = scheduler.get_hicache_l3_cache_layout(GetHiCacheL3LayoutReq())
        self.assertTrue(reply.layout["enabled"])

    def test_plain_mla_primary_is_reported_without_unified_components(self):
        """Ordinary HiRadixCache has neither components nor sidecar specs."""
        store = self._store()
        store.mem_pool_host = self._host_pool()
        scheduler = self._scheduler(store)
        scheduler.tree_cache = SimpleNamespace(
            cache_controller=scheduler.tree_cache.cache_controller
        )
        layout = scheduler.get_hicache_l3_cache_layout(GetHiCacheL3LayoutReq()).layout
        self.assertTrue(layout["enabled"])
        self.assertEqual(
            layout["hicache_object_layout"]["pools"],
            [{"pool": "kv", "suffixes": ["__k"], "hit_policy": "all_pages"}],
        )

    def test_pp_stage_layout_uses_its_own_stage_suffixes(self):
        store = self._store(
            mla_suffix="1",
            pp_rank=1,
            pp_size=2,
            registered=(PoolName.DEEPSEEK_V4_C4,),
        )

        layout = store.describe_l3_object_layout(
            sidecar_pool_specs=[
                _sidecar(PoolName.DEEPSEEK_V4_C4, PoolHitPolicy.ALL_PAGES)
            ],
            swa_trailing_pages=None,
            page_size=64,
        )

        object_layout = layout["hicache_object_layout"]
        self.assertTrue(layout["enabled"])
        self.assertEqual(object_layout["pp_rank"], 1)
        self.assertEqual(object_layout["pp_size"], 2)
        self.assertEqual(
            object_layout["pools"],
            [
                {
                    "pool": "deepseek_v4_c4",
                    "suffixes": ["_1_deepseek_v4_c4"],
                    "hit_policy": "all_pages",
                }
            ],
        )

    @staticmethod
    def _controller(controller_type):
        controller = controller_type.__new__(controller_type)
        controller.env_lock = threading.Lock()
        controller._scheduler_init_info_lock = threading.Lock()
        controller._scheduler_init_infos = []
        controller.scheduler_procs = []
        controller.local_kv_event_sources = []
        controller.attn_replica_size = 1
        controller.prelaunched_schedulers = None
        return controller

    def test_process_replies_reach_parent_init_info(self):
        store = self._store(registered=(PoolName.INDEXER,))
        reply = self._scheduler(store).get_init_info()
        controller = self._controller(dp_controller.DataParallelController)
        with ExitStack() as stack:
            adapter = stack.enter_context(
                patch.object(dp_controller.TorchMemorySaverAdapter, "create")
            )
            adapter.return_value.configure_subprocess.side_effect = nullcontext
            stack.enter_context(
                patch.object(
                    dp_controller.numa_utils,
                    "configure_subprocess",
                    side_effect=lambda *args: nullcontext(),
                )
            )
            stack.enter_context(
                patch.object(
                    dp_controller, "maybe_reindex_device_id", side_effect=nullcontext
                )
            )
            stack.enter_context(patch.object(dp_controller.mp, "Process"))
            reader = SimpleNamespace(recv=lambda: reply)
            stack.enter_context(
                patch.object(dp_controller.mp, "Pipe", return_value=(reader, object()))
            )
            controller.run_scheduler_process_func = None
            for rank in range(2):
                controller.launch_tensor_parallel_group(
                    SimpleNamespace(), SimpleNamespace(), 0, rank
                )

            messages = []
            controller.event_loop = lambda: None
            stack.enter_context(
                patch.object(
                    dp_controller, "DataParallelController", return_value=controller
                )
            )
            for name in ("kill_itself_when_parent_died", "configure_logger"):
                stack.enter_context(patch.object(dp_controller, name))
            stack.enter_context(
                patch.object(dp_controller.setproctitle, "setproctitle")
            )
            stack.enter_context(patch.object(dp_controller.faulthandler, "enable"))
            stack.enter_context(patch.object(dp_controller.psutil, "Process"))
            dp_controller.run_data_parallel_controller_process(
                get_server_args(),
                SimpleNamespace(),
                SimpleNamespace(send=messages.append),
            )
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            messages[0]["hicache_object_layout"], reply["hicache_object_layout"]
        )

    def test_startup_aggregation_keeps_sources_and_prelaunched_pids(self):
        controller = self._controller(dp_controller.DataParallelController)
        reply = self._scheduler(
            self._store(registered=(PoolName.INDEXER,))
        ).get_init_info()
        first = dict(
            reply,
            startup_time={
                "load_weight": 4,
                "scheduler_e2e": 5,
                "cuda_graph": {"decode": 2},
            },
        )
        second = dict(
            reply,
            startup_time={
                "load_weight": 1,
                "scheduler_e2e": 8,
                "cuda_graph": {"decode": 1, "prefill": 3},
            },
        )
        # Hold the first group's aggregation while the second group becomes ready.
        entered = threading.Event()
        proceed = threading.Event()

        def record_first():
            entered.set()
            proceed.wait()
            controller._record_scheduler_init_infos([first])

        thread = threading.Thread(target=record_first)
        thread.start()
        entered.wait()
        try:
            controller._record_scheduler_init_infos([second])
        finally:
            proceed.set()
            thread.join()
        controller.local_kv_event_sources = [{"dp_rank": 1}, {"dp_rank": 0}]
        controller.event_loop = lambda: None
        for prelaunched in (None, {(0, 0): (123, object())}):
            with (
                self.subTest(prelaunched=prelaunched is not None),
                ExitStack() as stack,
            ):
                stack.enter_context(
                    get_context().override_server_args(grpc_port=50051, dp_size=2)
                )
                for name in (
                    "kill_itself_when_parent_died",
                    "configure_logger",
                    "load_plugins",
                    "wait",
                ):
                    stack.enter_context(patch.object(dp_controller, name))
                stack.enter_context(
                    patch.object(
                        dp_controller, "DataParallelController", return_value=controller
                    )
                )
                stack.enter_context(
                    patch.object(dp_controller.setproctitle, "setproctitle")
                )
                stack.enter_context(patch.object(dp_controller.faulthandler, "enable"))
                stack.enter_context(patch.object(dp_controller.psutil, "Process"))
                messages = []
                dp_controller.run_data_parallel_controller_process(
                    get_server_args(),
                    SimpleNamespace(),
                    SimpleNamespace(send=messages.append),
                    prelaunched_schedulers=prelaunched,
                )
                self.assertEqual(len(messages), 1)
                info = messages[0]
                self.assertEqual(
                    info["hicache_object_layout"], reply["hicache_object_layout"]
                )
                self.assertEqual(info["startup_time"]["load_weight"], 4)
                self.assertEqual(info["startup_time"]["scheduler_e2e"], 8)
                self.assertEqual(info["startup_time"]["cuda_graph"]["decode"], 2)
                self.assertEqual(info["startup_time"]["cuda_graph"]["prefill"], 3)
                self.assertEqual(
                    info["kv_event_sources"], [{"dp_rank": 0}, {"dp_rank": 1}]
                )
                self.assertEqual(
                    info[dp_controller.SCHEDULER_PIDS_ARG], [123] if prelaunched else []
                )

    def test_different_or_missing_dp_layouts_disable_registration(self):
        """A mixed replica set must not silently advertise the first reply."""
        reply = self._scheduler(
            self._store(registered=(PoolName.INDEXER,))
        ).get_init_info()
        disabled_reply = self._scheduler(hierarchical=False).get_init_info()
        legacy_reply = {
            key: value for key, value in reply.items() if key != "hicache_object_layout"
        }
        different_reply = self._scheduler(
            self._store(registered=(PoolName.DEEPSEEK_V4_C4,))
        ).get_init_info()
        for other in (disabled_reply, legacy_reply, different_reply):
            with self.subTest(other=other):
                controller = self._controller(dp_controller.DataParallelController)
                controller._record_scheduler_init_infos([reply])
                controller._record_scheduler_init_infos([other])
                layout = controller.get_init_info()["hicache_object_layout"]
                self.assertFalse(layout["enabled"])
                self.assertIn("Inconsistent", layout["reason"])
        controller = self._controller(dp_controller.DataParallelController)
        controller._record_scheduler_init_infos([legacy_reply, legacy_reply])
        self.assertNotIn("hicache_object_layout", controller.get_init_info())

    def _stage_store(self, rank, size=2, *, mla=True):
        store = self._store(
            mla_suffix=str(rank),
            pp_rank=rank,
            pp_size=size,
            registered=(PoolName.INDEXER,),
        )
        store.is_mla_backend = mla
        return store

    def _merged(self, *stores):
        controller = self._controller(dp_controller.DataParallelController)
        for store in stores:
            reply = self._scheduler(store).get_init_info()
            controller._record_scheduler_init_infos([reply])
        return controller.get_init_info()["hicache_object_layout"]

    def test_pp_stage_replies_merge_into_one_complete_layout(self):
        stages = (self._stage_store(0), self._stage_store(1))
        for replicas in (stages, stages * 2):
            with self.subTest(replies=len(replicas)):
                layout = self._merged(*replicas)
                inner = layout["hicache_object_layout"]
                self.assertTrue(layout["enabled"])
                self.assertEqual(inner["pp_size"], 2)
                self.assertIsNone(inner["pp_rank"])
                self.assertEqual(
                    [s for pool in inner["pools"] for s in pool["suffixes"]],
                    ["_0_indexer", "_1_indexer"],
                )

    def test_missing_or_conflicting_pp_stages_are_unavailable(self):
        layout = self._merged(self._stage_store(0))
        self.assertFalse(layout["enabled"])
        self.assertIn("Incomplete", layout["reason"])
        # The rank-local details survive so consumers can fall back.
        self.assertEqual(layout["hicache_object_layout"]["pp_size"], 2)
        self.assertEqual(layout["hicache_object_layout"]["pools"], [])
        layout = self._merged(self._stage_store(1))
        self.assertIn("Incomplete", layout["reason"])
        layout = self._merged(self._stage_store(0), self._stage_store(1, size=3))
        self.assertIn("Inconsistent", layout["reason"])
        disagreeing = self._stage_store(0)
        disagreeing.registered_pools[PoolName.DEEPSEEK_V4_C4] = self._host_pool()
        layout = self._merged(self._stage_store(0), disagreeing)
        self.assertIn("Inconsistent", layout["reason"])

    def test_mha_pp_stages_keep_rank_local_metadata(self):
        layout = self._merged(
            self._stage_store(0, mla=False), self._stage_store(1, mla=False)
        )
        self.assertFalse(layout["enabled"])
        self.assertEqual(layout["hicache_object_layout"]["pp_size"], 2)
        self.assertFalse(layout["hicache_object_layout"]["is_mla_backend"])

    def test_scheduler_infos_share_the_merged_layout(self):
        """Any startup info must be representative, whichever one a consumer reads."""
        from sglang.srt.mem_cache.hicache_storage import merge_scheduler_info_layouts

        infos = [
            self._scheduler(self._stage_store(rank)).get_init_info() for rank in (0, 1)
        ]
        merge_scheduler_info_layouts(infos)
        self.assertTrue(infos[0]["hicache_object_layout"]["enabled"])
        # Merging a merged layout again (DP controller, then engine) must be a no-op.
        before = infos[0]["hicache_object_layout"]
        merge_scheduler_info_layouts([dict(infos[0])])
        self.assertEqual(infos[0]["hicache_object_layout"], before)
        self.assertIs(
            infos[0]["hicache_object_layout"], infos[1]["hicache_object_layout"]
        )
        self.assertEqual(
            [
                s
                for pool in infos[0]["hicache_object_layout"]["hicache_object_layout"][
                    "pools"
                ]
                for s in pool["suffixes"]
            ],
            ["_0_indexer", "_1_indexer"],
        )
        legacy = [{"max_total_num_tokens": 1}, {"max_total_num_tokens": 1}]
        merge_scheduler_info_layouts(legacy)
        self.assertNotIn("hicache_object_layout", legacy[0])

    def test_endpoint_uses_the_startup_merge_for_pp(self):
        """Only pp rank 0 answers control requests, so the endpoint needs the merge."""
        from sglang.srt.entrypoints import http_server

        startup = self._merged(self._stage_store(0), self._stage_store(1))
        stage_zero = self._scheduler(self._stage_store(0))

        async def communicate(req):
            return [stage_zero.get_hicache_l3_cache_layout(req)]

        manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            get_hicache_l3_cache_layout_communicator=communicate,
        )

        async def collect():
            return await TokenizerControlMixin.get_hicache_l3_cache_layout(manager)

        manager.get_hicache_l3_cache_layout = collect
        state = SimpleNamespace(
            tokenizer_manager=manager,
            scheduler_info={"hicache_object_layout": startup},
        )
        with patch.object(http_server, "_global_state", state):
            layout = asyncio.run(http_server.get_hicache_l3_cache_layout())
        self.assertEqual(layout, startup)

    def test_ray_actor_replies_reach_engine_init_info(self):
        ray = ModuleType("ray")
        ray.remote = lambda cls: cls
        ray.get = lambda refs: refs
        placement = ModuleType("ray.util.placement_group")
        placement.PlacementGroup = object
        strategies = ModuleType("ray.util.scheduling_strategies")
        strategies.PlacementGroupSchedulingStrategy = object
        stubs = {
            "ray": ray,
            "ray.util": ModuleType("ray.util"),
            "ray.util.placement_group": placement,
            "ray.util.scheduling_strategies": strategies,
        }
        srt_package = importlib.import_module("sglang.srt")
        with patch.dict("sys.modules", stubs), patch.dict(vars(srt_package)):
            engine = importlib.import_module("sglang.srt.ray.engine")
            ray_dp = importlib.import_module("sglang.srt.ray.data_parallel_controller")
            actor_module = importlib.import_module("sglang.srt.ray.scheduler_actor")
            controller_type = ray_dp.RayDataParallelController
            expected = self._scheduler(
                self._store(registered=(PoolName.INDEXER,))
            ).get_init_info()

            def create_actor(**kwargs):
                actor = actor_module.SchedulerActor.__new__(actor_module.SchedulerActor)
                actor.scheduler = self._scheduler(
                    self._store(registered=(PoolName.INDEXER,))
                )
                return SimpleNamespace(
                    get_info=SimpleNamespace(remote=actor.get_info),
                    run_event_loop=SimpleNamespace(remote=lambda: None),
                )

            def create_controller(*args):
                controller = self._controller(controller_type)
                controller.scheduler_actors = []
                controller.event_loop_refs = []
                controller.is_custom_pg = False
                controller.bundle_for_node = [0]
                controller.rank0_node_ip = "127.0.0.1"
                controller.pg = object()
                for rank in range(2):
                    controller._launch_ray_tp_group(
                        SimpleNamespace(), SimpleNamespace(nccl_port=1234), rank
                    )
                return controller

            with (
                patch.object(
                    ray_dp, "_create_scheduler_actor", side_effect=create_actor
                ),
                patch.object(
                    ray_dp, "RayDataParallelController", side_effect=create_controller
                ),
                patch.object(engine, "declare_resolution"),
                patch.object(engine.threading, "Thread"),
            ):
                result = engine.RayEngine._launch_dp_scheduler_processes(
                    SimpleNamespace(),
                    SimpleNamespace(nccl_port=1234),
                    object(),
                    [0],
                    "127.0.0.1",
                )
            self.assertEqual(
                result.scheduler_infos[0]["hicache_object_layout"],
                expected["hicache_object_layout"],
            )

    def test_tokenizer_control_and_endpoint_reject_inconsistent_replies(self):
        """The public endpoint must not discard a disagreeing DP reply."""
        from sglang.srt.entrypoints import http_server

        enabled = self._scheduler(self._store(registered=(PoolName.INDEXER,)))
        disabled = self._scheduler(hierarchical=False)
        schedulers = ()

        async def communicate(req):
            return [
                scheduler.get_hicache_l3_cache_layout(req) for scheduler in schedulers
            ]

        manager = SimpleNamespace(
            auto_create_handle_loop=lambda: None,
            get_hicache_l3_cache_layout_communicator=communicate,
        )

        async def collect():
            return await TokenizerControlMixin.get_hicache_l3_cache_layout(manager)

        manager.get_hicache_l3_cache_layout = collect
        with patch.object(
            http_server, "_global_state", SimpleNamespace(tokenizer_manager=manager)
        ):
            for schedulers, expected_enabled in (
                ((enabled, enabled), True),
                ((disabled, disabled), False),
                ((enabled, disabled), False),
                ((), False),
            ):
                with self.subTest(replies=len(schedulers), enabled=expected_enabled):
                    layout = asyncio.run(http_server.get_hicache_l3_cache_layout())
                    self.assertEqual(layout["enabled"], expected_enabled)
                    if not expected_enabled:
                        self.assertTrue(layout["reason"])


if __name__ == "__main__":
    unittest.main()
