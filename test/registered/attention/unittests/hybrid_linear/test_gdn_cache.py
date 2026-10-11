"""SGLang-owned GDN persistence contract, without running GPU kernels."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.linear.gdn_cache import (
    GDNCache,
    decode_key,
    eligible_key,
    encode_key,
    gdn_prefill_cache,
    key_digest,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture
def cache_key(monkeypatch):
    # Types must already exist in loaded GDN/CuTe modules; manifests cannot
    # import arbitrary Python modules to reconstruct a key.
    modules = []
    for name, cls_name in (
        ("flashinfer.gdn_kernels.test_cache", "Kernel"),
        ("cutlass.test_cache", "EnableTVMFFI"),
    ):
        module = ModuleType(name)
        setattr(module, cls_name, type(cls_name, (), {"__module__": name}))
        monkeypatch.setitem(sys.modules, name, module)
        modules.append(module)
    return (
        ((modules[0].Kernel, object), ("dtype", torch.bfloat16), ("dim", 128)),
        ((modules[1].EnableTVMFFI, True),),
    )


class Compiled:
    def export_to_c(self, path, function_name):
        Path(path).write_text(function_name)


def load_module(path, enable_tvm_ffi):
    assert enable_tvm_ffi
    symbol = Path(path).read_text()
    if not symbol.startswith("sglang_gdn_"):
        raise ValueError("Corrupt test artifact")
    return SimpleNamespace(**{symbol: Compiled()})


def test_key_roundtrip_and_restricted_types(cache_key):
    assert eligible_key(cache_key)
    assert decode_key(encode_key(cache_key)) == cache_key
    assert not eligible_key((cache_key[0], ()))
    assert key_digest(encode_key(cache_key)) != key_digest(encode_key((128,)))
    with pytest.raises(TypeError):
        encode_key(object())
    with pytest.raises(ValueError):
        decode_key(["type", "os", "system"])


def test_export_restore_and_memory_hit(tmp_path, cache_key):
    writer = GDNCache(tmp_path, "identity", {cache_key: Compiled()}, load_module)
    assert writer.persist() == 1
    assert writer.persist() == 0
    memory = {}
    reader = GDNCache(tmp_path, "identity", memory, load_module)
    assert reader.restore() == 1
    assert cache_key in memory
    assert reader.restore() == 0
    assert not list(tmp_path.rglob("*.tmp.*"))
    assert GDNCache(tmp_path, "new-source", {}, load_module).restore() == 0


def test_corrupt_object_is_replaced_after_recompile(tmp_path, cache_key):
    writer = GDNCache(tmp_path, "identity", {cache_key: Compiled()}, load_module)
    writer.persist()
    target = next(tmp_path.rglob("*.o"))
    target.write_text("corrupt")
    reader = GDNCache(tmp_path, "identity", {}, load_module)
    assert reader.restore() == 0
    reader.memory_cache[cache_key] = Compiled()
    assert reader.persist() == 1
    assert GDNCache(tmp_path, "identity", {}, load_module).restore() == 1


def test_invalid_manifest_is_ignored(tmp_path, cache_key):
    cache = GDNCache(tmp_path, "identity", {cache_key: Compiled()}, load_module)
    cache.persist()
    manifest = json.loads(cache.manifest.read_text())
    encoded = next(iter(manifest["entries"].values()))
    manifest["entries"] = {"../../invalid": encoded}
    cache.manifest.write_text(json.dumps(manifest))
    assert GDNCache(tmp_path, "identity", {}, load_module).restore() == 0
    cache.manifest.write_text("broken JSON")
    assert GDNCache(tmp_path, "identity", {}, load_module).restore() == 0


def test_export_failure_does_not_commit_manifest(tmp_path, cache_key):
    class Broken:
        def export_to_c(self, path, function_name):
            Path(path).write_text("partial")
            raise RuntimeError("export failed")

    cache = GDNCache(tmp_path, "identity", {cache_key: Broken()}, load_module)
    assert cache.persist() == 0
    assert not cache.manifest.exists()
    assert not list(tmp_path.rglob("*.o"))
    assert not list(tmp_path.rglob("*.tmp.*"))


def test_disable_skips_dependency_cache(monkeypatch):
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    with gdn_prefill_cache():
        pass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
