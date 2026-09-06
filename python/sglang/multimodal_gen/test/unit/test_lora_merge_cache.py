"""The merge cache is a pure byte vault: tensors in, mapped tensors out.

What matters: put round-trips the exact merged bytes and returns a mapping,
a complete cache serves the same bytes back, a mismatched entry is refused,
disk shortage returns None instead of raising, and the key separates
combinations that must not share files.
"""

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
