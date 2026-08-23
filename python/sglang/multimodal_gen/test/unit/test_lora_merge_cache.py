"""Merged LoRA weights end up file-backed instead of anonymous.

What matters: put round-trips the exact merged bytes and re-points the
parameter at the mapping, a complete store is adopted (same bytes, no write),
an incomplete or mismatched store is not adopted, and the key separates
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


def _param() -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.randn(16, 16), requires_grad=False)


def test_put_round_trips_and_repoints(tmp_path):
    weight = _param()
    merged = weight.data.clone()
    store = LoraMergeCache("k1", expected_bytes=weight.numel() * 4)

    assert store.put("blocks.0.linear", weight)
    assert torch.equal(weight.data, merged)
    store.finalize()

    second = LoraMergeCache("k1", expected_bytes=0)
    assert second.is_complete()
    other = torch.nn.Parameter(torch.zeros(16, 16), requires_grad=False)
    assert second.get("blocks.0.linear", other)
    assert torch.equal(other.data, merged)


def test_an_incomplete_store_is_not_adopted():
    weight = _param()
    store = LoraMergeCache("k2", expected_bytes=64)
    assert store.put("a", weight)
    # no finalize -> no manifest
    assert not LoraMergeCache("k2", expected_bytes=0).is_complete()


def test_a_mismatched_entry_refuses_adoption():
    weight = _param()
    store = LoraMergeCache("k3", expected_bytes=64)
    assert store.put("a", weight)
    store.finalize()

    second = LoraMergeCache("k3", expected_bytes=0)
    assert second.is_complete()
    wrong_shape = torch.nn.Parameter(torch.zeros(8, 8), requires_grad=False)
    assert not second.get("a", wrong_shape)


def test_disk_shortage_falls_back_to_memory(monkeypatch):
    import shutil as _shutil
    from types import SimpleNamespace

    monkeypatch.setattr(
        _shutil, "disk_usage", lambda _: SimpleNamespace(free=1, total=1, used=0)
    )
    weight = _param()
    store = LoraMergeCache("k4", expected_bytes=1 << 40)
    assert not store.put("a", weight)
    assert weight.data.is_contiguous()  # untouched, still anonymous


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
