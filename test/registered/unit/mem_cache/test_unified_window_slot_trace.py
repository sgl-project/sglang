"""Prefix-cache results over unified storage do not depend on tree slot ids.

The prefix tree numbers its slots internally; everything outside it names a
window by its sub-pool. This test drives a fixed script of inserts, matches,
locks, evictions, decode retirements and host-stack wiring over storages with
one and with two windows, on the Python and the native tree cores, and records
every result by sub-pool or slot name. The record must equal the stored one
(`unified_window_slot_trace.json`), so renumbering slots, or moving where a
boundary maps its native window, changes nothing observable. Set
``SGLANG_TEST_REGEN_WINDOW_SLOT_TRACE=1`` to rewrite the stored record.
"""

import json
import os
import random
import unittest
from array import array
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.build import UnifiedBuildContext, build_kv_storage
from sglang.srt.mem_cache.kv.decl import MHAGeometry, WindowLayers, kv_layer_decls
from sglang.srt.mem_cache.kv.plan import plan_kv_storage
from sglang.srt.mem_cache.kv.sizing import KVSizingInputs, size_plan
from sglang.srt.mem_cache.unified_cache.component_type import tree_slot_ids
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import setup_allocator_context

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

_TRACE = Path(__file__).with_name("unified_window_slot_trace.json")
_PAGE = 4
_MHA = MHAGeometry(head_num=1, head_dim=8, v_head_dim=8, kv_cache_dtype=torch.bfloat16)


def _storage(windows):
    setup_allocator_context()
    decls = kv_layer_decls(
        full_layer_ids=(0,),
        full_geometry=_MHA,
        windows=[
            WindowLayers(layer_ids=(i + 1,), geometry=_MHA, window=w)
            for i, w in enumerate(windows)
        ],
    )
    plan = plan_kv_storage(decls, page_size=_PAGE)
    sized = size_plan(
        plan,
        KVSizingInputs(
            full_tokens=256,
            window_tokens=128,
            state_slots=0,
            max_running_requests=3,
            context_len=96,
            alignment=1,
        ),
    )
    ctx = UnifiedBuildContext(
        device="cpu",
        start_layer=0,
        end_layer=len(windows) + 1,
        enable_memory_saver=False,
        need_sort=False,
        max_num_reqs=3,
        model_context_len=96,
        lazy_compaction=False,
    )
    return build_kv_storage(plan, sized, ctx)


def _cache(storage, windows, backend):
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    storage.bind_window_retention(windows[0], {})
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=storage.allocator,
            page_size=_PAGE,
            sliding_window_size=windows[0],
            tree_slots=storage.tree_slots(),
            tree_core_backend=backend,
        )
    )


def _backend_of(cache) -> str:
    return "rust" if "rust" in type(cache.tree_core).__module__ else "python"


def _slot_names(storage):
    """Tree slot id -> slot name, as this build numbers the slots."""
    slots = storage.tree_slots()
    return {int(tree_slot_ids(slots)[s.component]): s.component for s in slots}


def _windows_of(cache):
    return [name for name, _ in cache.window_sub_pools()]


def _sizes(cache, storage):
    a = storage.allocator
    return {
        "full_evictable": cache.full_evictable_size(),
        "window_evictable": {
            n: cache.window_evictable_size(n) for n in _windows_of(cache)
        },
        "window_protected": {
            n: cache.window_protected_size(n) for n in _windows_of(cache)
        },
        "allocated": {b.sub_pool_name: b.allocated_count() for b in a.bands},
    }


def _reprefill(cache):
    """Tokens a match must leave to re-prefill when a matched window cannot be
    reused as is (forced on: the decoder's bounded window replay)."""
    saved = cache.decoder_swa_bounded_replay
    cache.decoder_swa_bounded_replay = True
    try:
        return cache.window_reprefill_tail_tokens(None)
    finally:
        cache.decoder_swa_bounded_replay = saved


def _tree_script(windows, backend):
    """Inserts, matches, locks and evictions; every result by name."""
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey

    storage = _storage(windows)
    cache = _cache(storage, windows, backend)
    a = storage.allocator
    rng = random.Random(7)
    trunk = list(range(10_000, 10_064))
    seqs = []
    for i in range(6):
        cut = rng.choice((16, 32, 48))
        tail = [20_000 + 100 * i + j for j in range(rng.choice((8, 16, 24)))]
        seqs.append(trunk[:cut] + tail)
    out = {
        "backend": _backend_of(cache),
        "windows": _windows_of(cache),
        "reprefill": _reprefill(cache),
        "steps": [],
    }
    steps = out["steps"]
    for seq in seqs:
        key = RadixKey(array("q", seq))
        match = cache.match_prefix(MatchPrefixParams(key=key))
        have = match.device_prefix_len
        # A fresh allocation for the whole key; the insert frees what the
        # tree already holds.
        value = a.alloc(len(seq))
        result = cache.insert(InsertParams(key=key, value=value.clone()))
        steps.append(
            {
                "op": "insert",
                "len": len(seq),
                "matched": have,
                "prefix_len": int(getattr(result, "prefix_len", -1)),
                **_sizes(cache, storage),
            }
        )
    locks = []
    for seq in seqs[:3]:
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        steps.append(
            {
                "op": "match",
                "device_prefix_len": match.device_prefix_len,
                "host_hits": dict(match.window_host_hit_lengths),
                "branching": match.swa_branching_seqlen,
            }
        )
        locks.append(cache.lock(match.last_device_node))
        steps.append({"op": "lock", **_sizes(cache, storage)})
    names = _windows_of(cache)
    for demand in (
        EvictParams(num_tokens=8),
        EvictParams(window_num_tokens={n: 12 for n in names}),
        EvictParams(num_tokens=16, window_num_tokens={names[-1]: 40}),
    ):
        result = cache.evict(demand)
        steps.append(
            {
                "op": "evict",
                "num_tokens_evicted": result.num_tokens_evicted,
                "window_evicted": dict(result.window_num_tokens_evicted),
                **_sizes(cache, storage),
            }
        )
    for lock in locks:
        cache.unlock(lock)
    steps.append({"op": "unlock", **_sizes(cache, storage)})
    for seq in seqs:
        match = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", seq))))
        steps.append(
            {
                "op": "rematch",
                "device_prefix_len": match.device_prefix_len,
                "branching": match.swa_branching_seqlen,
            }
        )
    result = cache.evict(
        EvictParams(num_tokens=1 << 20, window_num_tokens={n: 1 << 20 for n in names})
    )
    steps.append(
        {
            "op": "evict_all",
            "num_tokens_evicted": result.num_tokens_evicted,
            "window_evicted": dict(result.window_num_tokens_evicted),
            **_sizes(cache, storage),
            "violations": storage.accounting_violations(),
        }
    )
    return out


def _decode_script(windows, backend):
    """A decoding request retires each window by its own cursor."""
    from sglang.srt.environ import envs
    from sglang.srt.managers.schedule_batch import ReqKvInfo, ScheduleBatch
    from sglang.srt.mem_cache.base_prefix_cache import (
        EvictParams,
        InsertParams,
        MatchPrefixParams,
    )
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    with (
        envs.SGLANG_OPT_RELEASE_PREFILL_SWA.override(True),
        envs.SGLANG_SWA_EVICTION_INTERVAL.override(1),
    ):
        storage = _storage(windows)
        cache = _cache(storage, windows, backend)
        a = storage.allocator
        prefix = a.alloc(16)
        key = RadixKey(array("q", range(16)))
        cache.insert(InsertParams(key=key, value=prefix.clone()))
        node_id = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
        lock = cache.lock(node_id)
        req = SimpleNamespace(
            session=None,
            lock=lock,
            seqlen=17,
            decode_batch_idx=0,
            kv=ReqKvInfo(req_pool_idx=1, cache_protected_len=16),
        )
        storage.requests.req_to_token[1, :16] = prefix
        batch = ScheduleBatch(
            reqs=[req],
            tree_cache=cache,
            req_to_token_pool=storage.requests,
            token_to_kv_pool_allocator=a,
            forward_mode=ForwardMode.DECODE,
        )
        names = _windows_of(cache)
        out = {"backend": _backend_of(cache), "windows": names, "steps": []}
        for frontier in range(20, 49, 4):
            ids = a.alloc(4)
            storage.requests.req_to_token[1, frontier - 4 : frontier] = ids
            req.seqlen = frontier + 1
            req.decode_batch_idx = frontier - 16
            batch.maybe_evict_swa()
            cache.evict(EvictParams(window_num_tokens={n: 80 for n in names}))
            out["steps"].append(
                {
                    "frontier": frontier,
                    "cursors": {n: req.kv.get_evicted_seqlen(n) for n in names},
                    **_sizes(cache, storage),
                }
            )
        cache.free_kv_row(req.kv, [(16, 48)])
        cache.unlock(lock)
        cache.evict(
            EvictParams(num_tokens=80, window_num_tokens={n: 80 for n in names})
        )
        out["end"] = {
            **_sizes(cache, storage),
            "violations": storage.accounting_violations(),
        }
        return out


def _linker_script(windows, backend, store_window: bool):
    """The external cache linker keeps one window: the tree's, by family."""
    from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
        UnifiedCacheLinkerWrapper,
    )

    storage = _storage(windows)
    cache = _cache(storage, windows, backend)
    names = _slot_names(storage)
    by_object = {id(c): names[int(ct)] for ct, c in cache.components.items()}
    pools = {}
    if store_window:
        for component in cache.components.values():
            name = getattr(component, "pool_name", None)
            if name is not None:
                pools[name] = None
    linker = SimpleNamespace(pools=pools, layer_done_counter=object())
    try:
        wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    except ValueError:
        return {"backend": _backend_of(cache), "refused": True}
    return {
        "backend": _backend_of(cache),
        "refused": False,
        "window": None if wrapper._window is None else by_object[id(wrapper._window)],
        "window_pool": str(wrapper._swa_pool),
        "skip_window": wrapper._skip_swa,
        "components": [by_object[id(c)] for c in wrapper._components],
    }


def _host_stack_script():
    """The unified host stack wires each pool's host pressure to its own slot."""
    import test_unified_kv_storage as kvs

    builder = kvs.TestHostStack("test_owner_host_pool_is_named_by_its_sub_pool")
    out = {}
    for chain in ("swa", "tri"):
        result, controller, cache = builder._build(chain, backend=None)
        storage = cache.token_to_kv_pool_allocator.storage
        names = _slot_names(storage)
        entries = {}
        for entry in result.host_pool_group.entry_map.values():
            cache.calls.clear()
            if entry.host_evict_fn is not None:
                entry.host_evict_fn(3)
            host_calls = [
                (kind, n, names[int(component)]) for kind, n, component in cache.calls
            ]
            entries[str(entry.name)] = {"host_evict": host_calls}
        out[chain] = {
            "entries": entries,
            "component_host_pools": sorted(
                names[int(ct)] for ct in result.component_host_pools
            ),
            "transfer_layer_id_max": controller["transfer_layer_id_max"],
        }
    return out


def _rust_available() -> bool:
    try:
        cache = _cache(_storage((16,)), (16,), "rust")
        return _backend_of(cache) == "rust"
    except Exception:  # noqa: BLE001 - any build or import failure means absent
        return False


def _record():
    record = {}
    backends = ["python"] + (["rust"] if _rust_available() else [])
    for backend in backends:
        record[f"tree/{backend}/one"] = _tree_script((16,), backend)
        record[f"decode/{backend}/one"] = _decode_script((16,), backend)
        for stored in (True, False):
            record[f"linker/{backend}/one/{stored}"] = _linker_script(
                (16,), backend, stored
            )
    record["linker/python/two/True"] = _linker_script((16, 8), "python", True)
    record["tree/python/two"] = _tree_script((16, 8), "python")
    record["decode/python/two"] = _decode_script((16, 8), "python")
    record["tree/python/three"] = _tree_script((16, 8, 4), "python")
    record["decode/python/three"] = _decode_script((16, 8, 4), "python")
    record["host_stack"] = _host_stack_script()
    return json.loads(json.dumps(record, sort_keys=True))


class TestWindowSlotTrace(CustomTestCase):
    def test_results_do_not_depend_on_tree_slot_ids(self):
        record = _record()
        if os.environ.get("SGLANG_TEST_REGEN_WINDOW_SLOT_TRACE") == "1":
            _TRACE.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
            self.skipTest(f"rewrote {_TRACE.name}")
        stored = json.loads(_TRACE.read_text())
        missing = sorted(set(stored) - set(record))
        if missing and all("/rust/" in k for k in missing):
            # The native core could not be built here; compare the rest.
            stored = {k: v for k, v in stored.items() if k not in missing}
        for key in sorted(stored):
            with self.subTest(key=key):
                self.assertEqual(record.get(key), stored[key])
        self.assertEqual(sorted(record), sorted(stored))


if __name__ == "__main__":
    unittest.main()
