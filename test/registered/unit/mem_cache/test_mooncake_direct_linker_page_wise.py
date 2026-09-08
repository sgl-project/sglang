from types import SimpleNamespace

import pytest

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)


class _Pool:
    def prepare_locations(self, host_indices):
        return list(host_indices)

    def get_prepared_layer_range_meta(self, locations, layer):
        return (
            [[location] for location in locations],
            [[layer + 1] for _ in locations],
            [[layer] for _ in locations],
        )


class _Counter:
    def __init__(self):
        self.completed = []
        self.failed = []

    def complete(self, counter_index, layer):
        self.completed.append((counter_index, layer))

    def fail(self, counter_index, error):
        self.failed.append((counter_index, error))


def _make_linker(store, *, threshold=10, num_layers=2):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.num_layers = num_layers
    linker.page_wise_load_threshold = threshold
    linker.pools = {PoolName.KV: _Pool()}
    linker.storage = SimpleNamespace(
        store=store,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    linker.layer_done_counter = _Counter()
    return linker


@pytest.mark.parametrize(
    ("key_count", "threshold", "expected_calls"),
    [(9, 10, [9, 9]), (10, 10, [10]), (10, 11, [10, 10])],
)
def test_page_wise_load_threshold(key_count, threshold, expected_calls):
    class _Store:
        def __init__(self):
            self.calls = []

        def batch_get_session_start(self, keys):
            return [0] * len(keys)

        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            self.calls.append(len(keys))
            return [sum(item) for item in sizes]

        def batch_get_session_end(self, keys):
            return None

    store = _Store()
    linker = _make_linker(store, threshold=threshold)
    transfer = SimpleNamespace(
        name=PoolName.KV,
        keys=[f"key-{index}" for index in range(key_count)],
        host_indices=list(range(key_count)),
    )

    linker.load_layer_wise(3, [("rid", [transfer])])

    assert store.calls == expected_calls
    assert linker.layer_done_counter.completed == [(3, 0), (3, 1)]
    assert linker.layer_done_counter.failed == []


