"""The unified pool's transfer surface: external names, PD wire entries, regions,
and the HiCache host stack built from the plan.

The PD registration and the host stack are compared with what the pool
classes chose (the per-class PD branches and HiCache strategies), which PD
peers already running and every L3 key written so far follow.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation import utils as pd
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
from sglang.srt.mem_cache.hybrid_cache.host_pool_config import is_mla_pool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_P16 = dict(page_size=16, full_tokens=64 * 16, swa_tokens=64 * 16)
CHAINS = {
    "swa": UnifiedGeometry(
        chain="swa", full_layer_ids=(0, 1), swa_layer_ids=(2, 3), **_P16
    ),
    "mamba": UnifiedGeometry(
        chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3), **_P16
    ),
    "mamba_mla": UnifiedGeometry(
        chain="mamba", full_layer_ids=(0, 1), state_layer_ids=(2, 3), mla=True, **_P16
    ),
    "tri": UnifiedGeometry(
        chain="tri",
        full_layer_ids=(0, 1),
        swa_layer_ids=(2, 3),
        state_layer_ids=(0, 2),
        **_P16,
    ),
}
_COMPONENTS = {
    "swa": {ComponentType.FULL, ComponentType.SWA},
    "mamba": {ComponentType.FULL, ComponentType.MAMBA},
    "tri": {ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA},
}


def _bundle(name):
    setup_allocator_context()
    return build_unified_bundle(CHAINS[name])


class TestPlanNames(CustomTestCase):
    def test_sub_pools_and_wire_entries_per_chain(self):
        expected = {
            "swa": (("full", "kv", "FULL"), ("swa", "swa", "SWA")),
            "mamba": (("full", "kv", "FULL"), ("mamba", "mamba", "MAMBA")),
            "tri": (
                ("full", "kv", "FULL"),
                ("swa", "swa", "SWA"),
                ("mamba", "mamba", "MAMBA"),
            ),
        }
        wires = {
            "swa": [("swa", "kv", "swa")],
            "mamba": [("mamba", "state", "mamba")],
            "tri": [("swa", "kv", "swa"), ("mamba", "state", "mamba")],
        }
        envelopes = {"swa": ("full", "swa"), "mamba": None, "tri": None}
        for name in ("swa", "mamba", "tri"):
            with self.subTest(chain=name):
                storage = _bundle(name).token_to_kv_pool_allocator.storage
                self.assertEqual(storage.owner, "full")
                self.assertEqual(
                    tuple(
                        (s.name, s.external.host_pool, s.external.tree_component)
                        for s in storage.sub_pools()
                    ),
                    expected[name],
                )
                self.assertEqual(
                    [
                        (e.sub_pool, e.part, e.state_type)
                        for e in storage.wire_entries()
                    ],
                    wires[name],
                )
                self.assertEqual(storage.page_envelope_pair(), envelopes[name])

    def test_layer_indices_follow_the_pool_routes(self):
        for name in CHAINS:
            with self.subTest(chain=name):
                bundle = _bundle(name)
                storage = bundle.token_to_kv_pool_allocator.storage
                router = bundle.token_to_kv_pool.layer_router
                for s in storage.sub_pools():
                    if s.holds_state:
                        expected = bundle.token_to_kv_pool_allocator.state.layer_map
                    else:
                        expected = {
                            layer_id: router.route(layer_id).index
                            for layer_id in router.sub_pool(s.name).layer_ids
                        }
                    self.assertEqual(s.layer_indices, expected)


def _kv_args_fields(kv_args):
    return {
        name: getattr(kv_args, name)
        for name in (
            "state_types",
            "state_data_ptrs",
            "state_data_lens",
            "state_item_lens",
            "state_dim_per_tensor",
            "state_conv_shard_groups",
            "state_slice_outer_counts",
            "state_layer_ids",
            "is_hybrid_mla_backend",
            "swa_kv_cache_layout",
            "mla_compression_ratios",
        )
    }


class TestPdRegistration(CustomTestCase):
    def test_plan_wire_matches_the_pool_class_branches(self):
        for name in CHAINS:
            with self.subTest(chain=name):
                bundle = _bundle(name)
                pool = bundle.token_to_kv_pool
                allocator = bundle.token_to_kv_pool_allocator
                state_pool = None if allocator.state is None else allocator.state.pool
                by_plan, by_class = SimpleNamespace(), SimpleNamespace()
                pd.setup_state_kv_args(
                    by_plan, pool, None, 4, state_pool, storage=allocator.storage
                )
                pd.setup_state_kv_args(by_class, pool, None, 4, state_pool)
                self.assertEqual(_kv_args_fields(by_plan), _kv_args_fields(by_class))
                self.assertEqual(
                    pd.get_kv_transfer_buf_infos(pool, allocator.storage),
                    pd.get_kv_transfer_buf_infos(pool),
                )


class TestHostSizing(CustomTestCase):
    def test_plan_sizes_match_the_pool_views(self):
        from sglang.srt.mem_cache import hicache_auto_size as sizing

        for name in CHAINS:
            with self.subTest(chain=name):
                bundle = _bundle(name)
                allocator = bundle.token_to_kv_pool_allocator
                params = SimpleNamespace(
                    token_to_kv_pool_allocator=allocator,
                    req_to_token_pool=getattr(bundle, "req_to_token_pool", None),
                    mtp_draft_device_pools=(),
                )
                by_plan = sizing._estimate_hicache_bytes(params, None)
                with patch.object(allocator, "storage", None):
                    by_views = sizing._estimate_hicache_bytes(params, None)
                self.assertEqual(by_plan, by_views)


class _RecordingCache:
    def __init__(self, bundle, components):
        self.token_to_kv_pool_allocator = bundle.token_to_kv_pool_allocator
        self.page_size = CHAINS["swa"].page_size
        self.components = {c: None for c in components}
        self.calls = []

    def evict_host(self, n, component):
        self.calls.append(("host", n, component))

    def evict_for_alloc(self, params):
        self.calls.append(("device", params))


class _Group:
    """A host pool group that only holds its entries; the recorded host pools
    carry none of the buffers a real group checks."""

    def __init__(self, entries):
        self.entry_map = {}
        for entry in entries:
            self.add_entry(entry)

    def add_entry(self, entry):
        self.entry_map[entry.name] = entry

    def get_pool(self, name):
        return self.entry_map[name].host_pool


def _host_pool_recorders():
    """Host pool builders that record what they were asked for."""

    def kv_host(*, kv_pool, page_size, use_mla=None, host_size=None, **kwargs):
        return SimpleNamespace(
            kind="kv",
            device_pool=kv_pool,
            page_size=page_size,
            use_mla=is_mla_pool(kv_pool) if use_mla is None else use_mla,
            host_size=host_size,
            drafts=tuple(kwargs.get("mtp_draft_device_pools", ())),
        )

    def state_host(pool, info, *, host_size, layout, mtp_draft_device_pools=()):
        return SimpleNamespace(
            kind="state",
            device_pool=pool,
            host_size=host_size,
            layout=layout,
            drafts=tuple(mtp_draft_device_pools),
        )

    def envelope_pair(*, device_pools, pool_names=("full", "swa"), **kwargs):
        assert not kwargs.pop("mtp_draft_device_pools", ())
        return tuple(
            SimpleNamespace(kind="envelope", device_pool=p, name=n, **kwargs)
            for p, n in zip(device_pools, pool_names)
        )

    return kv_host, state_host, envelope_pair


class TestHostStack(CustomTestCase):
    def _build(self, name, *, backend, by_plan):
        bundle = _bundle(name)
        allocator = bundle.token_to_kv_pool_allocator
        chain = CHAINS[name].chain
        cache = _RecordingCache(bundle, _COMPONENTS[chain])
        params = SimpleNamespace(
            token_to_kv_pool_allocator=allocator,
            req_to_token_pool=getattr(bundle, "req_to_token_pool", None),
            page_size=CHAINS[name].page_size,
            tp_cache_group="tp",
            attn_cp_cache_group="attn_cp",
            attn_tp_cache_group="attn_tp",
            pp_cache_group="pp",
            mtp_draft_device_pools=(),
            mtp_draft_mamba_pools=(),
        )
        memory = SimpleNamespace(
            hicache_ratio=2.0,
            hicache_size=0,
            hicache_mem_layout="page_first",
            hicache_storage_backend=backend,
            hicache_write_policy="write_through",
            hicache_io_backend="kernel",
            hicache_host_memory_mode="default",
        )
        controller = {}

        def record(self, *args, **kwargs):
            controller.update(kwargs, args=args)

        kv_host, state_host, pair = _host_pool_recorders()
        with (
            patch.object(hpa, "get_memory", return_value=memory),
            patch.object(hpa, "build_kv_host_pool", side_effect=kv_host),
            patch.object(hpa, "build_mamba_host_pool", side_effect=state_host),
            patch.object(
                hpa.UnifiedPageEnvelopeHostPool,
                "build_hybrid_swa_pool_pair",
                side_effect=pair,
            ),
            patch.object(hpa, "_get_allocator_type", return_value="default"),
            patch.object(hpa, "HostPoolGroup", _Group),
            patch.object(hpa.HybridCacheController, "__init__", record),
        ):
            if by_plan:
                result = hpa.build_unified_host_stack(
                    storage=allocator.storage,
                    params=params,
                    cache=cache,
                    load_cache_event=None,
                    storage_backend=backend,
                )
            else:
                kvcache = allocator.get_kvcache()
                result = hpa._select_strategy(kvcache, _COMPONENTS[chain]).build(
                    cache=cache,
                    kvcache=kvcache,
                    params=params,
                    server_args=None,
                    load_cache_event=None,
                    storage_backend=backend,
                )
        return result, controller, cache

    def _summary(self, result, controller, cache):
        limit = controller["transfer_layer_id_max"] + 4
        entries = []
        for entry in result.host_pool_group.entry_map.values():
            host = dict(vars(entry.host_pool))
            host.pop("name", None)  # the envelope side's label
            calls = []
            for fn, arg in ((entry.host_evict_fn, 3), (entry.device_evict_fn, 1 << 30)):
                cache.calls.clear()
                if fn is not None:
                    fn(arg)
                calls.append(list(cache.calls))
            entries.append(
                (
                    entry.name,
                    host,
                    entry.device_pool,
                    [entry.layer_mapper(t) for t in range(limit)],
                    entry.is_primary_index_anchor,
                    entry.packed_draft_device_pools,
                    calls,
                    entry.device_alloc_fn,
                    entry.device_free_fn,
                )
            )
        components = {
            c: next(
                n
                for n, e in result.host_pool_group.entry_map.items()
                if e.host_pool is pool
            )
            for c, pool in result.component_host_pools.items()
        }
        return entries, components, result.sidecars, result.loads_state_layer_wise

    def test_matches_the_pool_class_strategies(self):
        # The class strategies no longer build a page-envelope pair: a window
        # chain is compared on a backend that keeps no envelopes.
        for name in CHAINS:
            for backend in (None, "file"):
                if backend is None and name == "swa":
                    continue
                with self.subTest(chain=name, backend=backend):
                    plan = self._build(name, backend=backend, by_plan=True)
                    legacy = self._build(name, backend=backend, by_plan=False)
                    self.assertEqual(
                        {k: v for k, v in plan[1].items() if k != "args"},
                        {k: v for k, v in legacy[1].items() if k != "args"},
                    )
                    self.assertEqual(plan[1]["args"][2:], legacy[1]["args"][2:])
                    # Separate builds: compare by position, not identity.
                    plan_entries, *plan_rest = self._summary(*plan)
                    legacy_entries, *legacy_rest = self._summary(*legacy)
                    self.assertEqual(plan_rest, legacy_rest)
                    self.assertEqual(
                        [_shape(e) for e in plan_entries],
                        [_shape(e) for e in legacy_entries],
                    )

    def test_window_ends_share_a_page_envelope_backing(self):
        result, controller, cache = self._build("swa", backend=None, by_plan=True)
        storage = cache.token_to_kv_pool_allocator.storage
        kv, swa = result.host_pool_group.entry_map.values()
        self.assertEqual((kv.name, swa.name), (PoolName.KV, PoolName.SWA))
        for entry, sub_pool in ((kv, "full"), (swa, "swa")):
            self.assertEqual(entry.host_pool.kind, "envelope")
            self.assertEqual(entry.host_pool.name, sub_pool)
            self.assertIs(entry.host_pool.device_pool, storage.view(sub_pool))
        # Each side's host pressure evicts its own tree component.
        for entry, component in ((kv, ComponentType.FULL), (swa, ComponentType.SWA)):
            cache.calls.clear()
            entry.host_evict_fn(3)
            self.assertEqual(cache.calls, [("host", 3, component)])
        self.assertIsNone(kv.device_evict_fn)
        cache.calls.clear()
        swa.device_evict_fn(1 << 30)
        ((kind, params),) = cache.calls
        self.assertEqual(kind, "device")
        self.assertGreater(params.swa_num_tokens, 0)
        self.assertEqual(controller["transfer_layer_id_max"], 4)

    def test_owner_host_pool_is_named_by_its_sub_pool(self):
        result, _, _ = self._build("swa", backend="file", by_plan=True)
        self.assertEqual(
            [e.name for e in result.host_pool_group.entry_map.values()],
            [PoolName.KV, PoolName.SWA],
        )


def _shape(entry):
    """An entry with its objects replaced by what they are, so two builds over
    two pools of one geometry compare equal."""
    name, host, device_pool, layer_map, anchor, packed, calls, alloc, free = entry
    host = {k: type(v).__name__ if k == "device_pool" else v for k, v in host.items()}
    return (
        name,
        host,
        type(device_pool).__name__,
        layer_map,
        anchor,
        len(packed),
        [
            [(c[0], c[1], c[2]) if c[0] == "host" else (c[0], vars(c[1])) for c in call]
            for call in calls
        ],
        None if alloc is None else alloc.__qualname__,
        None if free is None else free.__qualname__,
    )


if __name__ == "__main__":
    unittest.main()
