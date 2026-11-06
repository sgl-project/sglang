"""CPU regressions for persistent diffusion LoRA cache invalidation."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


@pytest.fixture
def merge_cache_module(monkeypatch, tmp_path):
    # Load the complete module without pipelines_core.__init__, which imports
    # GPU model loaders unrelated to this filesystem-only cache contract.
    path = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/multimodal_gen/runtime/pipelines_core/lora/lora_merge_cache.py"
    )
    spec = importlib.util.spec_from_file_location("lora_merge_cache_cpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("SGLANG_DIFFUSION_CACHE_ROOT", str(tmp_path / "cache"))
    return module


@pytest.mark.parametrize("atomic_replace", [False, True])
def test_same_size_adapter_update_does_not_reuse_merged_weights(
    merge_cache_module, tmp_path, atomic_replace
):
    """Overwriting a local adapter must not serve the previous adapter's delta."""
    module = merge_cache_module
    adapter = tmp_path / "adapter.safetensors"
    a = torch.ones(1, 2)
    b = torch.ones(2, 1)
    save_file({"lora_A.weight": a, "lora_B.weight": b}, adapter)
    original_stat = adapter.stat()
    base_paths = [str(tmp_path / "base"), "transformer"]

    def cache_for_adapter():
        key = module.lora_merge_cache_key(base_paths, [(str(adapter), 1.0, None)])
        return module.LoraMergeCache(key, expected_bytes=16)

    original = cache_for_adapter()
    assert original.put("proj", b @ a) is not None
    original.finalize()
    # An unchanged file still adopts the existing cache across startups.
    assert cache_for_adapter().is_complete()

    updated_b = b * 2
    destination = tmp_path / "replacement.safetensors" if atomic_replace else adapter
    save_file({"lora_A.weight": a, "lora_B.weight": updated_b}, destination)
    # Make the timestamp change deterministic even on coarse-resolution filesystems.
    os.utime(
        destination,
        ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000_000),
    )
    if atomic_replace:
        os.replace(destination, adapter)
    assert adapter.stat().st_size == original_stat.st_size

    restarted = cache_for_adapter()
    assert not restarted.is_complete(), "updated adapter adopted stale merged weights"
    expected = updated_b @ a
    assert restarted.put("proj", expected) is not None
    restarted.finalize()
    reloaded = cache_for_adapter()
    assert reloaded.is_complete()
    torch.testing.assert_close(
        reloaded.get("proj", expected.shape, expected.dtype), expected
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
