"""Striped host tier wiring: flag, config gate, staging, anchor swap, metrics."""

import ast
import functools
import inspect
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from prometheus_client import CollectorRegistry, Counter, Histogram

from sglang.srt.arg_groups import pipeline
from sglang.srt.arg_groups.hicache_hook import validate_hicache_host_shard
from sglang.srt.layers.attention.dsv4 import unified_kv_gate
from sglang.srt.managers import cache_controller as cache_controller_module
from sglang.srt.mem_cache import kv_cache_builder, l2_transfer, memory_pool_host
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.pool_host.striped_host import (
    StripedGatherError,
    StripedGatherExecutor,
    StripedHostPlacement,
    StripedLogicalHostPool,
    build_striped_load_plan,
    create_striped_gather_executor,
    plan_owner_imbalance,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.observability.metrics_collector import RadixCacheMetricsCollector
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE = 256
NUM_RANKS = 4
C4_LAYERS = 2
C128_LAYERS = 2
TRANSFER_LAYERS = C4_LAYERS + C128_LAYERS
DEVICE_PAGES = 40
# --hicache-ratio: 40 device pages -> 8 physical host pages per rank.
HICACHE_RATIO = 0.2
PER_RANK_PAGES = 8
# FP4 indexer page rows: 64 compressed slots per 256-token page.
INDEXER_SLOTS = 64
# Bytes per page row: DSv4 shapes (test_striped_host_narrow.py), fewer layers.
C4_BYTES, C4_ROPE_BYTES, C128_BYTES, C128_ROPE_BYTES = 32768, 8192, 1024, 256
INDEXER_BYTES, INDEXER_SCALE_BYTES = 64 * INDEXER_SLOTS, 4 * INDEXER_SLOTS
ROW_BYTES = C4_LAYERS * (
    C4_BYTES + C4_ROPE_BYTES + INDEXER_BYTES + INDEXER_SCALE_BYTES
) + C128_LAYERS * (C128_BYTES + C128_ROPE_BYTES)
STRIPED_POOLS = (
    PoolName.DEEPSEEK_V4_C4,
    PoolName.DEEPSEEK_V4_C4_INDEXER,
    PoolName.DEEPSEEK_V4_C4_INDEXER_SCALE,
    PoolName.DEEPSEEK_V4_C4_ROPE,
    PoolName.DEEPSEEK_V4_C128,
    PoolName.DEEPSEEK_V4_C128_ROPE,
)

# DSv4-Pro AgentX server flags plus the stripe flag. --enable-prefill-cp is left
# out: it is platform-validated before the dummy-model return, and the gate keys
# on attn_cp_size.
BASELINE = dict(
    model_path="dummy",
    tp_size=8,
    attn_cp_size=8,
    enable_cp_decode_attn_tp=True,
    cp_strategy="interleave",
    speculative_algorithm="DSPARK",
    page_size=256,
    enable_hierarchical_cache=True,
    hicache_ratio=1.5,
    hicache_io_backend="direct",
    hicache_mem_layout="page_first_direct",
    hicache_write_policy="write_through",
    hicache_host_shard="stripe",
)


# ---- Flag and config gate ----


class TestHostShardArgs(unittest.TestCase):
    def setUp(self):
        # The baseline recipe runs the DSv4 unified KV layout on ROCm.
        patch = mock.patch.object(
            unified_kv_gate, "is_unified_kv_triton", return_value=True
        )
        self.unified_kv = patch.start()
        self.addCleanup(patch.stop)

    def test_baseline_recipe_accepted(self):
        validate_hicache_host_shard(ServerArgs(**BASELINE))
        # Plain head-TP (attention TP == TP) also keeps the device KV replicated.
        validate_hicache_host_shard(
            ServerArgs(**{**BASELINE, "attn_cp_size": 1, "cp_strategy": None})
        )
        validate_hicache_host_shard(
            ServerArgs(**{**BASELINE, "speculative_algorithm": None})
        )

    def test_off_is_never_refused(self):
        # The stock default must not newly reject any configuration.
        for overrides in (
            {},
            {"hicache_storage_backend": "file", "pp_size": 2},
            {"enable_hierarchical_cache": False, "enable_dp_attention": True},
            {"tp_size": 1, "attn_cp_size": 1, "cp_strategy": None},
            {"speculative_algorithm": "EAGLE3", "hicache_write_policy": "write_back"},
        ):
            with self.subTest(overrides=overrides):
                validate_hicache_host_shard(
                    ServerArgs(**{**BASELINE, "hicache_host_shard": "off", **overrides})
                )
        self.unified_kv.return_value = False
        validate_hicache_host_shard(
            ServerArgs(**{**BASELINE, "hicache_host_shard": "off"})
        )

    def test_each_unsupported_config_refused(self):
        cases = (
            ({"enable_hierarchical_cache": False}, "requires --enable-hierarchical"),
            ({"tp_size": 1, "attn_cp_size": 1, "cp_strategy": None}, "--tp-size > 1"),
            ({"speculative_algorithm": "EAGLE3"}, "only DSpark"),
            ({"speculative_algorithm": "NEXTN"}, "only DSpark"),
            ({"hicache_write_policy": "write_back"}, "write-policy write_through"),
            (
                {"hicache_write_policy": "write_through_selective"},
                "write-policy write_through",
            ),
            ({"attn_dp_size": 2}, "attention data parallelism"),
            ({"enable_dp_attention": True}, "attention data parallelism"),
            ({"enable_cp_decode_attn_tp": False}, "requires --enable-cp-decode"),
            ({"enable_dsa_cache_layer_split": True}, "layer-split"),
            ({"pp_size": 2}, "pipeline parallelism"),
            ({"dcp_size": 2}, "decode context parallelism"),
            ({"enable_kv_cache_sharding": True}, "kv-cache-sharding"),
            ({"hicache_storage_backend": "file"}, "storage-backend"),
            ({"hicache_host_memory_mode": "buffer_only"}, "memory-mode cache"),
            ({"enable_unified_cache_external_linker": True}, "external-linker"),
            ({"disaggregation_mode": "decode"}, "aggregated serving only"),
            ({"disaggregation_mode": "prefill"}, "aggregated serving only"),
            ({"enable_pdmux": True}, "pdmux"),
            ({"enable_hisparse": True}, "hisparse"),
            ({"enable_lmcache": True}, "lmcache"),
        )
        for overrides, reason in cases:
            with self.subTest(overrides=overrides):
                args = ServerArgs(**{**BASELINE, **overrides})
                with self.assertRaisesRegex(
                    ValueError, f"--hicache-host-shard stripe .*{reason}"
                ):
                    validate_hicache_host_shard(args)
        # Without unified KV the paged SWA pool is not page-aligned to the anchor.
        self.unified_kv.return_value = False
        with self.assertRaisesRegex(ValueError, "unified KV layout"):
            validate_hicache_host_shard(ServerArgs(**BASELINE))

    def test_builder_requires_executor_iff_stripe(self):
        executor = object()
        for shard, gather, ok in (
            ("off", None, True),
            ("stripe", executor, True),
            ("stripe", None, False),
            ("off", executor, False),
        ):
            with (
                self.subTest(shard=shard, gather=gather),
                mock.patch.object(
                    kv_cache_builder,
                    "get_memory",
                    return_value=SimpleNamespace(hicache_host_shard=shard),
                ),
            ):
                params = SimpleNamespace(hicache_stripe_gather=gather)
                if ok:
                    kv_cache_builder.check_hicache_stripe_gather(params)
                else:
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        kv_cache_builder.check_hicache_stripe_gather(params)

    def test_gate_runs_on_the_final_resolved_view(self):
        # After the last KV-sharding gate: parallel sizes and the PD mode are final.
        source = inspect.getsource(pipeline.run_resolution_pipeline)
        calls = [
            node.args[0].id
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "run_hook"
            and isinstance(node.args[0], ast.Name)
        ]
        self.assertEqual(calls.count("validate_hicache_host_shard"), 1)
        last_sharding = len(calls) - 1 - calls[::-1].index("handle_kv_cache_sharding")
        self.assertGreater(calls.index("validate_hicache_host_shard"), last_sharding)


# ---- Staging before KV profiling ----


class _FakeComm:
    def __init__(self, world_size, rank, available=True):
        self.available = available
        self.world_size = world_size
        self.rank = rank
        self.device = torch.device("cpu")


def _tp_group(world_size=4, rank=1, comm=True):
    return SimpleNamespace(
        pynccl_comm=_FakeComm(world_size, rank) if comm else None,
        world_size=world_size,
        rank_in_group=rank,
    )


class TestGatherExecutorFactory(unittest.TestCase):
    def test_builds_tp_placement_and_staging(self):
        executor = create_striped_gather_executor(
            _tp_group(), page_size=PAGE, device="cpu", staging_bytes=4 << 10
        )
        self.assertEqual(executor.placement.key(), (4, 1, PAGE))
        self.assertEqual(executor._staging.numel(), 4 << 10)
        self.assertIs(executor.transport.comm.device, executor.transport.device)


class TestModelRunnerStaging(unittest.TestCase):
    def _runner(self, *, draft=False):
        runner = ModelRunner.__new__(ModelRunner)
        runner.is_draft_worker = draft
        runner.hicache_stripe_gather = None
        runner.page_size = PAGE
        runner.device = "cpu"
        runner.memory_pool_config = None
        runner.pre_model_load_memory = 0
        return runner

    def _alloc(self, runner, shard):
        order = []

        class _Stop(Exception):
            pass

        def configure(**kwargs):
            order.append("configure")
            raise _Stop

        executor = SimpleNamespace(
            staging_bytes=512 << 20, placement=SimpleNamespace(num_ranks=8)
        )

        def create(*, tp_group, page_size, device):
            order.append(("staging", page_size, device))
            return executor

        runner.init_kv_cache_configurator = lambda _layout: setattr(
            runner, "kv_cache_configurator", SimpleNamespace(configure=configure)
        )
        with (
            mock.patch(
                "sglang.srt.model_executor.model_runner.get_memory",
                return_value=SimpleNamespace(hicache_host_shard=shard),
            ),
            mock.patch(
                "sglang.srt.model_executor.model_runner.get_dp_tp_group",
                return_value=object(),
            ),
            mock.patch(
                "sglang.srt.model_executor.model_runner.create_striped_gather_executor",
                side_effect=create,
            ),
            self.assertRaises(_Stop),
        ):
            runner.alloc_memory_pool()
        return order, executor

    def test_staging_is_allocated_before_kv_profiling(self):
        runner = self._runner()
        order, executor = self._alloc(runner, "stripe")
        self.assertEqual(order, [("staging", PAGE, "cpu"), "configure"])
        self.assertIs(runner.hicache_stripe_gather, executor)
        # Re-profiling (alloc_memory_pool again) keeps the one staging buffer.
        order, _ = self._alloc(runner, "stripe")
        self.assertEqual(order, ["configure"])

    def test_no_staging_off_or_for_draft(self):
        for shard, draft in (("off", False), ("stripe", True)):
            with self.subTest(shard=shard, draft=draft):
                runner = self._runner(draft=draft)
                order, _ = self._alloc(runner, shard)
                self.assertEqual(order, ["configure"])
                self.assertIsNone(runner.hicache_stripe_gather)


# ---- Anchor swap in the DSv4 stack ----


def _cpu_patches(layout="page_first_direct", hicache_ratio=HICACHE_RATIO):
    def _ref_transfer_kv_direct(
        src_layers, dst_layers, src_indices, dst_indices, page_size
    ):
        assert page_size == 1
        for src, dst in zip(src_layers, dst_layers, strict=True):
            dst[dst_indices] = src[src_indices]

    def _ref_all_layer_direct_lf_pf(
        src_ptrs, dst_ptrs, src_indices, dst_indices, page_size
    ):
        assert page_size == 1
        for layer_id, src in enumerate(src_ptrs):
            dst_ptrs[0][dst_indices, layer_id, 0] = src[src_indices]

    def _ref_per_layer_direct_pf_lf(
        src_ptrs, dst_ptrs, src_indices, dst_indices, layer_id, page_size
    ):
        assert page_size == 1
        dst_ptrs[0][dst_indices] = src_ptrs[0][src_indices, layer_id, 0]

    memory = SimpleNamespace(
        hicache_mem_layout=layout,
        hicache_write_policy="write_through",
        hicache_io_backend="direct",
        hicache_host_memory_mode="cache",
        hicache_ratio=hicache_ratio,
        hicache_size=0,
    )
    patches = [
        mock.patch.object(
            memory_pool_host, "transfer_kv_direct", _ref_transfer_kv_direct, create=True
        ),
        mock.patch.object(
            memory_pool_host,
            "transfer_kv_all_layer_direct_lf_pf",
            _ref_all_layer_direct_lf_pf,
            create=True,
        ),
        mock.patch.object(
            memory_pool_host,
            "transfer_kv_per_layer_direct_pf_lf",
            _ref_per_layer_direct_pf_lf,
            create=True,
        ),
        mock.patch.object(
            memory_pool_host, "host_memory_budget_bytes", return_value=1 << 40
        ),
        mock.patch.object(assembler, "get_memory", return_value=memory),
        mock.patch.object(assembler, "_get_allocator_type", return_value="default"),
        # CPU-only torch cannot pin; everything else is the real host pool.
        mock.patch.object(
            assembler,
            "DeepSeekV4PagedHostPool",
            functools.partial(DeepSeekV4PagedHostPool, pin_memory=False),
        ),
        mock.patch.object(
            cache_controller_module,
            "device_module",
            SimpleNamespace(Event=lambda **kwargs: _FakeEvent()),
        ),
        mock.patch.object(
            l2_transfer, "device_module", SimpleNamespace(Stream=lambda: None)
        ),
    ]
    for patch in patches:
        patch.start()
    return patches


def _fake_dsv4_kvcache(seed):
    """Unified-KV DSv4 device pools on CPU (FP4 indexer, fp8 rope siblings)."""
    gen = torch.Generator().manual_seed(seed)

    def rows(num_layers, row_bytes):
        return [
            torch.randint(
                0, 256, (DEVICE_PAGES, row_bytes), dtype=torch.uint8, generator=gen
            )
            for _ in range(num_layers)
        ]

    def fp4(num_layers, width):
        return [
            torch.randint(
                0,
                256,
                (DEVICE_PAGES, 1, width, INDEXER_SLOTS),
                dtype=torch.uint8,
                generator=gen,
            )
            for _ in range(num_layers)
        ]

    regions = {4: (rows(C4_LAYERS, C4_BYTES), C4_BYTES)}
    regions[128] = (rows(C128_LAYERS, C128_BYTES), C128_BYTES)
    ropes = {4: (rows(C4_LAYERS, C4_ROPE_BYTES), C4_ROPE_BYTES)}
    ropes[128] = (rows(C128_LAYERS, C128_ROPE_BYTES), C128_ROPE_BYTES)
    return SimpleNamespace(
        _unified_kv=True,
        size=DEVICE_PAGES * PAGE,
        swa_size=0,
        swa_page_size=PAGE,
        swa_kv_pool=None,
        device="cpu",
        layer_num=TRANSFER_LAYERS,
        register_layer_transfer_counter=lambda counter: None,
        c4_kv_pool=SimpleNamespace(),
        c128_kv_pool=SimpleNamespace(),
        c4_indexer_kv_pool=SimpleNamespace(
            index_k_with_scale_buffer=None,
            index_k_payload_buffer=fp4(C4_LAYERS, 64),
            index_k_scale_buffer=fp4(C4_LAYERS, 4),
        ),
        unified_region_buffers=lambda ratio: regions[ratio],
        unified_rope_region_buffers=lambda ratio: ropes[ratio],
    )


_LAYER_MAPPINGS = assembler._DeepSeekV4LayerMappings(
    transfer_layer_id_max=TRANSFER_LAYERS,
    full={i: i for i in range(TRANSFER_LAYERS)},
    swa={},
    c4={2 * i + 1: i for i in range(C4_LAYERS)},
    c128={2 * i: i for i in range(C128_LAYERS)},
    c4_state={},
    c4_state_global_layers=[],
)


def _params(kvcache, tp_group, executor):
    allocator = mock.MagicMock()
    allocator.size_full = kvcache.size
    allocator.get_kvcache.return_value = kvcache
    del allocator.c128_attn_allocator
    del allocator.full_attn_allocator
    return CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=PAGE,
        tp_cache_group=tp_group,
        hicache_stripe_gather=executor,
    )


def _build(params, kvcache):
    return assembler.build_deepseek_v4_hicache_stack(
        params=params,
        kvcache=kvcache,
        load_cache_event=None,
        storage_backend=None,
        layer_mappings=_LAYER_MAPPINGS,
    )


class _NoCollectiveTransport:
    def __init__(self, world_size, rank):
        self.world_size = world_size
        self.rank = rank

    def all_gather(self, recv, send):
        raise AssertionError("no gather expected")


def _local_executor(rank=1, page_size=PAGE):
    placement = StripedHostPlacement(NUM_RANKS, rank, page_size)
    return StripedGatherExecutor(
        placement,
        _NoCollectiveTransport(NUM_RANKS, rank),
        device=torch.device("cpu"),
        staging_bytes=NUM_RANKS * 2 * ROW_BYTES,
    )


class TestStripedStackAssembly(unittest.TestCase):
    def setUp(self):
        self._patches = _cpu_patches()

    def tearDown(self):
        for patch in self._patches:
            patch.stop()

    def _build_mocked(self, executor):
        controller = mock.MagicMock()
        with mock.patch.object(
            assembler, "HybridCacheController", return_value=controller
        ) as controller_cls:
            kvcache = _fake_dsv4_kvcache(seed=0)
            group, built = _build(_params(kvcache, None, executor), kvcache)
        self.assertIs(built, controller)
        return group, controller, controller_cls.call_args.kwargs

    def test_stripe_swaps_in_an_n_fold_logical_anchor(self):
        executor = _local_executor()
        group, controller, kwargs = self._build_mocked(executor)
        anchor = group.anchor_entry.host_pool
        self.assertIsInstance(anchor, StripedLogicalHostPool)
        self.assertIs(anchor.placement, executor.placement)
        self.assertEqual(anchor.per_rank_pages, PER_RANK_PAGES)
        # Logical capacity is N x the per-rank physical pages.
        self.assertEqual(anchor.size, NUM_RANKS * PER_RANK_PAGES * PAGE)
        self.assertEqual(group.logical_size, NUM_RANKS * PER_RANK_PAGES * PAGE)
        self.assertEqual(
            sorted(e.name for e in group.entries[1:]), sorted(STRIPED_POOLS)
        )
        for entry in group.entries[1:]:
            self.assertEqual(entry.host_pool.num_host_pages, PER_RANK_PAGES)
        self.assertIs(kwargs["host_stripe"], executor.placement)
        controller.install_striped_load_drain.assert_called_once_with(executor)

    def test_off_builds_the_stock_anchor(self):
        group, controller, kwargs = self._build_mocked(None)
        anchor = group.anchor_entry.host_pool
        self.assertIs(type(anchor), LogicalHostPool)
        self.assertEqual(anchor.size, PER_RANK_PAGES * PAGE)
        self.assertIsNone(kwargs["host_stripe"])
        controller.install_striped_load_drain.assert_not_called()

    def test_init_refusals(self):
        with self.subTest("placement page size mismatch"):
            with self.assertRaisesRegex(ValueError, "page size"):
                self._build_mocked(_local_executor(page_size=128))
        with self.subTest("staging too small for one row"):
            executor = _local_executor()
            executor.staging_bytes = NUM_RANKS * ROW_BYTES - 1
            with self.assertRaisesRegex(StripedGatherError, "cannot hold"):
                self._build_mocked(executor)

    def test_other_stacks_refuse_stripe(self):
        cache = SimpleNamespace(components={})
        params = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: None),
            hicache_stripe_gather=_local_executor(),
        )
        with (
            mock.patch.object(
                assembler, "_select_strategy", return_value=assembler._MambaStrategy()
            ),
            mock.patch.object(assembler, "get_serving"),
            self.assertRaisesRegex(ValueError, "only the DeepSeek V4"),
        ):
            assembler.attach_hybrid_pool_to_unified_cache(
                cache, params, None, load_cache_event=None
            )


# ---- Late pools ----


class TestStripedLatePool(unittest.TestCase):
    def setUp(self):
        self._patches = _cpu_patches()
        kvcache = _fake_dsv4_kvcache(seed=0)
        with mock.patch.object(
            assembler, "HybridCacheController", return_value=mock.MagicMock()
        ):
            executor = _local_executor(rank=1)
            self.group, _ = _build(_params(kvcache, None, executor), kvcache)
        self.placement = executor.placement

    def tearDown(self):
        for patch in self._patches:
            patch.stop()

    def _controller(self, host_stripe):
        cc = HybridCacheController.__new__(HybridCacheController)
        cc.mem_pool_host = self.group
        cc.host_stripe = host_stripe
        return cc

    def test_late_host_pool_registration_refused(self):
        entry = self.group.entries[1]
        with self.assertRaisesRegex(ValueError, "after the stack is built"):
            self._controller(self.placement).register_host_pool_entry(entry)


# ---- Metrics ----


def _metrics_collector():
    registry = CollectorRegistry()

    class _Collector(RadixCacheMetricsCollector):
        _counter_cls = functools.partial(Counter, registry=registry)
        _histogram_cls = functools.partial(Histogram, registry=registry)

    return _Collector(labels={"tp_rank": "0"}), registry


class TestStripedMetrics(unittest.TestCase):
    def test_observe_striped_load(self):
        collector, registry = _metrics_collector()
        labels = {"tp_rank": "0"}
        # Stock: the stripe families emit no samples (HELP/TYPE lines only).
        samples = [
            line for line in _exposition(registry).splitlines() if line[:1] != "#"
        ]
        self.assertFalse([line for line in samples if "hicache_stripe" in line])
        collector.observe_striped_load(
            gathered_bytes=3000, gather_seconds=0.004, owner_imbalance=1.5
        )
        sample = registry.get_sample_value
        self.assertEqual(sample("sglang:hicache_stripe_loads_total", labels), 1)
        self.assertEqual(
            sample("sglang:hicache_stripe_gather_bytes_total", labels), 3000
        )
        self.assertEqual(
            sample("sglang:hicache_stripe_gather_duration_seconds_count", labels), 1
        )

    def test_plan_owner_imbalance(self):
        placement = StripedHostPlacement(4, 0, 4)
        for pages, expected in (([0, 1, 2, 3], 1.0), ([0, 4, 8, 1], 3.0), ([], 1.0)):
            pages = torch.tensor(pages, dtype=torch.int64)
            host = (pages[:, None] * 4 + torch.arange(4)).flatten()
            plan = build_striped_load_plan(
                placement, host_indices=host, device_indices=host.clone(), seq=0
            )
            self.assertEqual(plan_owner_imbalance(plan), expected)


def _exposition(registry):
    from prometheus_client import generate_latest

    return generate_latest(registry).decode()


class _FakeEvent:
    def record(self, stream=None):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass

    def wait(self, stream=None):
        pass

    def elapsed_time(self, other):
        return 0.0


if __name__ == "__main__":
    unittest.main()
