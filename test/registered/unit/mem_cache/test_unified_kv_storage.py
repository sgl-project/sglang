"""Pool names, PD wire entries and host-stack routing must agree with the
byte layouts and storage keys used by PD peers and HiCache."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as hpa
from sglang.srt.mem_cache.kv.layout.mha import MHAHostLayout
from sglang.srt.mem_cache.kv.layout.mla import MLAHostLayout
from sglang.srt.mem_cache.kv.layout.state import StateHostLayout
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
    def test_nonzero_pipeline_stage_routes_local_views_and_global_metadata(self):
        for mla in (False, True):
            with self.subTest(mla=mla):
                setup_allocator_context()
                bundle = build_unified_bundle(
                    UnifiedGeometry(
                        chain="mamba",
                        mla=mla,
                        full_layer_ids=(4, 6),
                        state_layer_ids=(5, 7),
                        start_layer=4,
                        end_layer=8,
                        **_P16,
                    )
                )
                storage = bundle.storage
                view = storage.view("full")
                self.assertEqual(view.model_layer_ids, (4, 6))
                self.assertEqual(view.start_layer, 0)
                self.assertEqual(view.get_device_pool_infos()[0].layer_ids, (4, 6))
                first = view.kv_buffer[0] if mla else view.k_buffer[0]
                second = view.kv_buffer[1] if mla else view.k_buffer[1]
                first.fill_(11)
                second.fill_(29)
                route = bundle.token_to_kv_pool.layer_router.layer(6)
                self.assertTrue(torch.all(route.key() == 29))

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

    def test_tree_slots_in_drive_order(self):
        # What the prefix tree builds for a unified plan: today's component ids,
        # each with its sub-pool's family, in the order it drives their eviction
        # (owner, window, state).
        expected = {
            "swa": (("FULL", "token"), ("SWA", "window")),
            "mamba": (("FULL", "token"), ("MAMBA", "state")),
            "tri": (("FULL", "token"), ("SWA", "window"), ("MAMBA", "state")),
        }
        for name, slots in expected.items():
            with self.subTest(chain=name):
                storage = _bundle(name).token_to_kv_pool_allocator.storage
                self.assertEqual(
                    tuple((s.component, s.family) for s in storage.tree_slots()),
                    slots,
                )

    def test_static_tree_slots_follow_the_pool_shape(self):
        from sglang.srt.mem_cache.static_kv_storage import (
            StaticKVStorage,
            StaticPoolShape,
        )

        cases = {
            (False, False): (("FULL", "token"),),
            (True, False): (("FULL", "token"), ("SWA", "window")),
            (False, True): (("FULL", "token"), ("MAMBA", "state")),
            (True, True): (("FULL", "token"), ("SWA", "window"), ("MAMBA", "state")),
        }
        for (has_window, has_state), slots in cases.items():
            with self.subTest(has_window=has_window, has_state=has_state):
                storage = StaticKVStorage(
                    None,
                    None,
                    shape=StaticPoolShape(
                        has_window=has_window, has_state=has_state, token_capacity=1
                    ),
                )
                self.assertEqual(
                    tuple((s.component, s.family) for s in storage.tree_slots()),
                    slots,
                )

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

    def host(layout, options, *, draft_buffer_infos=()):
        return SimpleNamespace(
            host_layout=layout,
            page_size=options.page_size,
            host_size=options.size,
            layout=options.layout,
            drafts=draft_buffer_infos,
        )

    def envelope_pair(*, device_pools, pool_names=("full", "swa"), **kwargs):
        assert not kwargs.pop("mtp_draft_device_pools", ())
        return tuple(
            SimpleNamespace(kind="envelope", device_pool=p, name=n, **kwargs)
            for p, n in zip(device_pools, pool_names)
        )

    return host, envelope_pair


class TestHostStack(CustomTestCase):
    def _build(self, name, *, backend):
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

        host, pair = _host_pool_recorders()
        with (
            patch.object(hpa, "get_memory", return_value=memory),
            patch.object(MHAHostLayout, "build", host),
            patch.object(MLAHostLayout, "build", host),
            patch.object(StateHostLayout, "build", host),
            patch.object(
                hpa.UnifiedPageEnvelopeHostPool,
                "build_hybrid_swa_pool_pair",
                side_effect=pair,
            ),
            patch.object(hpa, "_get_allocator_type", return_value="default"),
            patch.object(hpa, "HostPoolGroup", _Group),
            patch.object(hpa.HybridCacheController, "__init__", record),
        ):
            result = hpa.build_unified_host_stack(
                storage=allocator.storage,
                params=params,
                cache=cache,
                load_cache_event=None,
                storage_backend=backend,
            )
        return result, controller, cache

    def test_window_ends_share_a_page_envelope_backing(self):
        result, controller, cache = self._build("swa", backend=None)
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

    def test_host_transfers_translate_through_the_storage(self):
        """A host copy maps the controller's ids to rows through the storage's
        id spaces, carried on its entry: the owner's and the state's ids are
        virtual, a window's host loads bind physical pages. The views hold no
        allocator callable."""
        for name in ("swa", "mamba", "tri"):
            with self.subTest(chain=name):
                result, _, cache = self._build(name, backend=None)
                allocator = cache.token_to_kv_pool_allocator
                storage = allocator.storage
                expected = {
                    storage.owner: allocator.full_attn_allocator.ids.translate,
                    "swa": None,
                    "mamba": None
                    if allocator.state is None
                    else allocator.state.slots.translate,
                }
                by_view = {
                    id(storage.view(s.name)): s.name for s in storage.sub_pools()
                }
                seen = set()
                for entry in result.host_pool_group.entry_map.values():
                    sub_pool = by_view[id(entry.device_pool)]
                    seen.add(sub_pool)
                    self.assertEqual(entry.device_translate, expected[sub_pool])
                    self.assertEqual(entry.device_translate, storage.host_ids(sub_pool))
                self.assertEqual(seen, set(storage.sub_pool_names()))

    def test_owner_host_pool_is_named_by_its_sub_pool(self):
        result, _, _ = self._build("swa", backend="file")
        self.assertEqual(
            [e.name for e in result.host_pool_group.entry_map.values()],
            [PoolName.KV, PoolName.SWA],
        )


if __name__ == "__main__":
    unittest.main()
