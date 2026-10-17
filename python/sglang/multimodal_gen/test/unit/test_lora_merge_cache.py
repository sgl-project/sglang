"""The merge cache is a pure byte vault: tensors in, mapped tensors out.

What matters: put round-trips the exact merged bytes and returns a mapping,
a complete cache serves the same bytes back, a mismatched entry is refused,
disk shortage returns None instead of raising, and the key separates
combinations that must not share files.
"""

import os

import pytest
import torch

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


def test_key_tracks_file_content_identity_and_ignores_cwd(tmp_path, monkeypatch):
    lora = tmp_path / "adapter.safetensors"
    lora.write_bytes(b"x" * 128)
    base = ["org/model", "transformer"]
    k = lora_merge_cache_key(base, [(str(lora), 1.0, None)])
    # Same size, rewritten in place: mtime must change the key.
    os.utime(lora, ns=(1, 1))
    assert k != lora_merge_cache_key(base, [(str(lora), 1.0, None)])
    # A repo id and module name are not paths; the cwd must not leak in.
    key_a = lora_merge_cache_key(base, [(str(lora), 1.0, None)])
    cwd_with_dirs = tmp_path / "cwd"
    (cwd_with_dirs / "transformer").mkdir(parents=True)
    monkeypatch.chdir(cwd_with_dirs)
    assert lora_merge_cache_key(base, [(str(lora), 1.0, None)]) == key_a
    # Two files from one repo are told apart.
    other = tmp_path / "low_noise.safetensors"
    other.write_bytes(b"x" * 128)
    os.utime(other, ns=(1, 1))
    assert lora_merge_cache_key(base, [(str(lora), 1.0, None)]) != (
        lora_merge_cache_key(base, [(str(other), 1.0, None)])
    )


def test_failed_put_removes_only_the_temp_file(monkeypatch):
    cache = LoraMergeCache("k5", expected_bytes=1)
    good = cache.put("a", torch.randn(2, 2))
    assert good is not None
    final = os.path.join(cache.root, os.listdir(cache.root)[0])

    def boom(tensors, path):
        with open(path, "wb") as handle:
            handle.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(
        "sglang.multimodal_gen.runtime.pipelines_core.lora.lora_merge_cache."
        "safetensors_save_file",
        boom,
    )
    assert cache.put("a", torch.randn(2, 2)) is None
    assert os.path.exists(final)
    assert sorted(os.listdir(cache.root)) == [os.path.basename(final)]


def test_old_stores_are_evicted(monkeypatch):
    stores = []
    for i in range(6):
        cache = LoraMergeCache(f"s{i}", expected_bytes=1)
        assert cache.put("a", torch.randn(2, 2)) is not None
        cache.finalize()
        os.utime(cache.root, (100 + i, 100 + i))
        stores.append(cache.root)
    assert [os.path.isdir(r) for r in stores] == [False, False, True, True, True, True]


def test_adopting_a_store_protects_it_from_eviction():
    first = LoraMergeCache("keep", expected_bytes=1)
    assert first.put("a", torch.randn(2, 2)) is not None
    first.finalize()
    os.utime(first.root, (1, 1))
    assert LoraMergeCache("keep", expected_bytes=1).is_complete()
    assert os.path.getmtime(first.root) > 1000


def test_get_returns_none_when_store_vanished(tmp_path):
    cache = LoraMergeCache("gone", expected_bytes=1)
    assert cache.put("a", torch.randn(2, 2)) is not None
    cache.finalize()
    adopted = LoraMergeCache("gone", expected_bytes=1)
    assert adopted.is_complete()
    import shutil

    shutil.rmtree(adopted.root)
    assert adopted.get("a", torch.Size([2, 2]), torch.float32) is None
