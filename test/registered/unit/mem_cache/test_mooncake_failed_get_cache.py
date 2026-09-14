import sys
from unittest.mock import MagicMock

import pytest

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
    MooncakeStore,
    _FailedGetCache,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_store(ttl_seconds: float = 5.0) -> MooncakeStore:
    store = MooncakeStore.__new__(MooncakeStore)
    store.store = MagicMock()
    store.failed_get_cache = _FailedGetCache(ttl_seconds, max_entries=128)
    return store


def test_failed_get_is_negative_cached_for_exists():
    store = _make_store()
    store.store.batch_get_into.return_value = [16, -5]

    assert store._get_batch_zero_copy_impl(
        ["good", "stale"], [0x1000, 0x2000], [16, 16]
    ) == [16, -5]

    store.store.batch_is_exist.return_value = [1]
    assert store._batch_exist(["good", "stale"]) == [1, 0]
    store.store.batch_is_exist.assert_called_once_with(["good"])


def test_get_exception_negative_caches_all_attempted_keys():
    store = _make_store()
    store.store.batch_get_into.side_effect = RuntimeError("transfer timeout")

    with pytest.raises(RuntimeError, match="transfer timeout"):
        store._get_batch_zero_copy_impl(
            ["stale-a", "stale-b"], [0x1000, 0x2000], [16, 16]
        )

    assert store._batch_exist(["stale-a", "stale-b"]) == [0, 0]
    store.store.batch_is_exist.assert_not_called()


def test_successful_put_clears_negative_cache_entry():
    store = _make_store()
    store.failed_get_cache.add("restored")
    store._use_group_semantics = False
    store._replicate_config_cls = None
    store.store.batch_put_from.return_value = [0]

    assert store._put_batch_zero_copy_impl(["restored"], [0x1000], [16]) == [0]

    store.store.batch_is_exist.return_value = [1]
    assert store._batch_exist(["restored"]) == [1]
    store.store.batch_is_exist.assert_called_once_with(["restored"])


def test_failed_put_keeps_negative_cache_entry():
    store = _make_store()
    store.failed_get_cache.add("still-stale")
    store._use_group_semantics = False
    store._replicate_config_cls = None
    store.store.batch_put_from.return_value = [-5]

    assert store._put_batch_zero_copy_impl(["still-stale"], [0x1000], [16]) == [-5]
    assert store._batch_exist(["still-stale"]) == [0]
    store.store.batch_is_exist.assert_not_called()


def test_failed_get_cache_entry_expires(monkeypatch):
    now = 100.0
    monkeypatch.setattr(
        "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.time.monotonic",
        lambda: now,
    )
    store = _make_store(ttl_seconds=1.0)
    store.failed_get_cache.add("recovered")

    now = 101.01
    store.store.batch_is_exist.return_value = [1]
    assert store._batch_exist(["recovered"]) == [1]
    store.store.batch_is_exist.assert_called_once_with(["recovered"])


def test_clear_removes_failed_get_entries():
    store = _make_store()
    store.failed_get_cache.add("old")

    store.clear()

    store.store.remove_all.assert_called_once_with()
    store.store.batch_is_exist.return_value = [1]
    assert store._batch_exist(["old"]) == [1]


def test_failed_get_cache_is_bounded():
    cache = _FailedGetCache(ttl_seconds=5.0, max_entries=2)
    cache.add("oldest")
    cache.add("middle")
    cache.add("newest")

    assert not cache.contains("oldest")
    assert cache.contains("middle")
    assert cache.contains("newest")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
