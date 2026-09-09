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
    linker.enable_page_wise_load = True
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


def test_page_wise_load_fetches_all_pages_in_one_call():
    class _Store:
        def __init__(self):
            self.calls = []

        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            self.calls.append(list(keys))
            return [sum(item) for item in sizes]

    store = _Store()
    linker = _make_linker(store, threshold=1, num_layers=3)
    keys = [f"key-{index}" for index in range(129)]

    result = linker._load_page_wise(
        4,
        [("rid", [])],
        {PoolName.KV: (keys, list(range(len(keys))))},
        {PoolName.KV: ["rid"] * len(keys)},
    )

    assert result == {"rid": True}
    assert [len(call) for call in store.calls] == [129]
    assert linker.layer_done_counter.completed == [(4, 0), (4, 1), (4, 2)]


def test_page_wise_scalar_error_is_attributed_without_escaping(caplog):
    class _Store:
        def batch_get_session_start(self, keys):
            return [0] * len(keys)

        def batch_get_into_multi_buffer_ranges(self, keys, ptrs, sizes, offsets):
            return 707

        def batch_get_session_end(self, keys):
            return None

    linker = _make_linker(_Store(), threshold=1)
    transfer = SimpleNamespace(
        name=PoolName.KV,
        keys=["page-a"],
        host_indices=[0],
    )

    # Worker-facing load failures are captured by load_layer_wise and published
    # through the layer counter instead of escaping the background thread.
    linker.load_layer_wise(5, [("rid", [transfer])])

    assert linker.layer_done_counter.failed
    assert "'rid': 'rid'" in caplog.text
    assert "707" in caplog.text


