"""The merge cache is a pure byte vault: tensors in, mapped tensors out.

What matters: put round-trips the exact merged bytes and returns a mapping,
a complete cache serves the same bytes back, a mismatched entry is refused,
disk shortage returns None instead of raising, and the key separates
combinations that must not share files.
"""

import os

import pytest
import torch
from safetensors.torch import load_file

from sglang.multimodal_gen.runtime.pipelines_core.lora.lora_merge_cache import (
    LoraMergeCache,
    lora_merge_cache_key,
)


@pytest.fixture(autouse=True)
def _cache_root(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", str(tmp_path / "cache"))


def test_put_round_trips_and_returns_a_mapping():
    merged = torch.randn(16, 16)
    cache = LoraMergeCache("k1", expected_bytes=merged.numel() * 4)

    mapped = cache.put("blocks.0.linear", merged)
    assert mapped is not None
    assert torch.equal(mapped, merged)
    cache.finalize()

    second = LoraMergeCache("k1", expected_bytes=0)
    assert second.is_complete()
    served = second.get("blocks.0.linear", merged.shape, merged.dtype)
    assert served is not None
    assert torch.equal(served, merged)


def test_an_incomplete_cache_is_not_complete():
    cache = LoraMergeCache("k2", expected_bytes=64)
    assert cache.put("a", torch.randn(4, 4)) is not None
    # no finalize -> no manifest
    assert not LoraMergeCache("k2", expected_bytes=0).is_complete()


def test_a_mismatched_entry_is_refused():
    cache = LoraMergeCache("k3", expected_bytes=64)
    assert cache.put("a", torch.randn(4, 4)) is not None
    cache.finalize()

    second = LoraMergeCache("k3", expected_bytes=0)
    assert second.is_complete()
    assert second.get("a", torch.Size([8, 8]), torch.float32) is None


def test_disk_shortage_returns_none(monkeypatch):
    import shutil as _shutil
    from types import SimpleNamespace

    monkeypatch.setattr(
        _shutil, "disk_usage", lambda _: SimpleNamespace(free=1, total=1, used=0)
    )
    cache = LoraMergeCache("k4", expected_bytes=1 << 40)
    assert cache.put("a", torch.randn(4, 4)) is None


def test_the_key_separates_combinations(tmp_path):
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"x" * 128)
    base = ["/models/h3", "transformer"]
    k = lora_merge_cache_key(base, [(str(lora), 1.0, None)])
    assert k != lora_merge_cache_key(base, [(str(lora), 0.5, None)])
    assert k != lora_merge_cache_key(base, [(str(lora), 1.0, 32)])
    assert k != lora_merge_cache_key(
        ["/models/h3", "transformer_2"], [(str(lora), 1.0, None)]
    )
    assert k == lora_merge_cache_key(base, [(str(lora), 1.0, None)])


def test_failed_put_keeps_the_previous_entry(monkeypatch):
    cache = LoraMergeCache("k5", expected_bytes=1)
    merged = torch.randn(2, 2)
    assert cache.put("a", merged) is not None

    def partial_write(tensors, path):
        with open(path, "wb") as handle:
            handle.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.lora_merge_cache."
        "safetensors_save_file",
        partial_write,
    )
    assert cache.put("a", torch.randn(2, 2)) is None
    (entry,) = os.listdir(cache.root)
    assert torch.equal(load_file(os.path.join(cache.root, entry))["weight"], merged)


def test_get_survives_store_evicted_after_is_complete():
    import shutil as _shutil

    cache = LoraMergeCache("raced", expected_bytes=1)
    assert cache.put("a", torch.randn(2, 2)) is not None
    cache.finalize()
    reader = LoraMergeCache("raced", expected_bytes=1)
    assert reader.is_complete()
    _shutil.rmtree(cache.root)
    assert reader.get("a", torch.Size([2, 2]), torch.float32) is None


def test_is_complete_treats_eviction_during_touch_as_a_miss(monkeypatch):
    """A store evicted between the file checks and the LRU touch is a miss."""
    cache = LoraMergeCache("touch_race", expected_bytes=1)
    assert cache.put("a", torch.randn(2, 2)) is not None
    cache.finalize()

    def evicted(_path):
        raise FileNotFoundError(_path)

    monkeypatch.setattr(os, "utime", evicted)
    reader = LoraMergeCache("touch_race", expected_bytes=1)
    assert not reader.is_complete()
    assert reader.get("a", torch.Size([2, 2]), torch.float32) is None


def test_low_space_evicts_least_recently_used_store(monkeypatch):
    import shutil as _shutil
    from types import SimpleNamespace

    roots = []
    for i, key in enumerate(("old", "recent")):
        cache = LoraMergeCache(key, expected_bytes=1)
        assert cache.put("a", torch.randn(2, 2)) is not None
        cache.finalize()
        os.utime(cache.root, (100 + i, 100 + i))
        roots.append(cache.root)

    real = _shutil.disk_usage
    # Space only frees up once the oldest store is gone.
    monkeypatch.setattr(
        _shutil,
        "disk_usage",
        lambda p: SimpleNamespace(free=1 << 40 if not os.path.isdir(roots[0]) else 1),
    )
    new = LoraMergeCache("new", expected_bytes=1 << 20)
    assert new.put("a", torch.randn(2, 2)) is not None
    assert [os.path.isdir(r) for r in roots] == [False, True]
    monkeypatch.setattr(_shutil, "disk_usage", real)
