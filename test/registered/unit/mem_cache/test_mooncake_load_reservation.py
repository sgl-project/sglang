from types import SimpleNamespace

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeStore:
    def __init__(self, missing=()):
        self.missing = set(missing)
        self.open = {}
        self.start_calls = []

    def batch_get_session_start(self, keys):
        self.start_calls.append(list(keys))
        results = []
        for key in keys:
            if key in self.missing:
                results.append(-1)
            else:
                self.open[key] = self.open.get(key, 0) + 1
                results.append(0)
        return results

    def batch_get_session_end(self, keys):
        for key in keys:
            self.open[key] -= 1
            if not self.open[key]:
                del self.open[key]
        return 0


def _linker(store):
    linker = object.__new__(MooncakeDirectLinker)
    linker.storage = SimpleNamespace(
        store=store,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    linker.pool_group = SimpleNamespace(
        resolve_transfers=lambda transfers, **_: transfers
    )
    linker.load_reservations = {}
    linker.reserved_key_refs = {}
    linker.stats = {"reserve_miss": 0}
    return linker


def _transfer(*keys):
    return SimpleNamespace(keys=list(keys))


def test_shared_keys_open_one_session_until_last_release():
    store = _FakeStore()
    linker = _linker(store)
    assert linker.reserve_load("a", [_transfer("k1", "k2")])
    assert linker.reserve_load("b", [_transfer("k2", "k3")])
    # k2 is already pinned by "a", so "b" only opens k3.
    assert store.start_calls == [["k1", "k2"], ["k3"]]

    linker.release_load_reservation("a")
    assert store.open == {"k2": 1, "k3": 1}
    linker.release_load_reservation("b")
    assert store.open == {}
    assert linker.reserved_key_refs == {}


def test_evicted_key_turns_hit_into_miss_and_releases_partial_sessions():
    store = _FakeStore(missing={"k2"})
    linker = _linker(store)
    assert not linker.reserve_load("a", [_transfer("k1", "k2")])
    assert store.open == {}
    assert linker.load_reservations == {}
    assert linker.stats["reserve_miss"] == 1
    # Releasing a request that never reserved is a no-op.
    linker.release_load_reservation("a")


class _LayerPool:
    def prepare_locations(self, host_indices):
        return list(host_indices)

    def get_prepared_layer_range_meta(self, locations, layer):
        return [[0]] * len(locations), [[1]] * len(locations), [[0]] * len(locations)


class _Counter:
    def __init__(self):
        self.completed = []
        self.failed = None

    def complete(self, index, layer):
        self.completed.append(layer)

    def fail(self, index, error):
        self.failed = error


def _loading_linker(store, num_layers):
    linker = _linker(store)
    store.batch_get_into_multi_buffer_ranges = lambda keys, ptrs, sizes, offsets: (
        [1] * len(keys)
    )
    linker.pools = {"kv": _LayerPool()}
    linker.num_layers = num_layers
    linker.layer_done_counter = _Counter()
    linker.stats["lease_renewal"] = 0
    return linker


def test_load_renews_the_lease_before_it_lapses(monkeypatch):
    """A load that outlives one Mooncake lease must not read with an expired session."""
    from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker

    # Renewal at start, then per-layer checks; layer 1 starts 6 s in.
    clock = iter([0.0, 1.0, 6.0, 6.0, 7.0])
    monkeypatch.setattr(mooncake_direct_linker.time, "monotonic", lambda: next(clock))
    store = _FakeStore()
    linker = _loading_linker(store, num_layers=3)
    transfer = SimpleNamespace(name="kv", keys=["k1", "k2"], host_indices=[0, 1])
    linker.load_layer_wise(0, [[transfer]])
    # Renewed at load start and again before layer 1.
    assert store.start_calls == [["k1", "k2"], ["k1", "k2"]]
    assert linker.layer_done_counter.completed == [0, 1, 2]
    assert linker.layer_done_counter.failed is None


def test_failed_lease_renewal_fails_the_load_batch():
    store = _FakeStore(missing={"k2"})
    linker = _loading_linker(store, num_layers=2)
    transfer = SimpleNamespace(name="kv", keys=["k1", "k2"], host_indices=[0, 1])
    linker.load_layer_wise(0, [[transfer]])
    assert linker.layer_done_counter.completed == []
    assert isinstance(linker.layer_done_counter.failed, RuntimeError)
