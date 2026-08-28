import logging
import sys
import types

import pytest

from sglang.kernels.jit import cute_aot_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_default_cache_dir_tracks_sglang_cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cute_aot_cache,
        "_resolve_target_arch",
        lambda: pytest.fail("constructing a cache must not probe CUDA"),
    )
    monkeypatch.delenv("SGLANG_CUTE_AOT_CACHE_DIR", raising=False)
    monkeypatch.setenv("SGLANG_CACHE_DIR", str(tmp_path))
    cache = cute_aot_cache.get_jit_cache("consumer")
    assert isinstance(cache, cute_aot_cache.JITPersistentCache)

    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", str(tmp_path / "explicit"))
    cache = cute_aot_cache.get_jit_cache("consumer")
    assert isinstance(cache, cute_aot_cache.JITPersistentCache)

    monkeypatch.setenv("SGLANG_CUTE_AOT_CACHE_DIR", "")
    cache = cute_aot_cache.get_jit_cache("consumer")
    assert not isinstance(cache, cute_aot_cache.JITPersistentCache)


def test_target_arch_prefers_env_then_detects_gpu(monkeypatch):
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_90a")
    assert cute_aot_cache._resolve_target_arch() == "sm_90a"

    monkeypatch.delenv("CUTE_DSL_ARCH")
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(get_device_capability=lambda: (10, 0))
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert cute_aot_cache._resolve_target_arch() == "sm_100"


def test_disabled_factory_returns_process_local_cache(monkeypatch):
    monkeypatch.setattr(
        cute_aot_cache,
        "_resolve_target_arch",
        lambda: pytest.fail("disabled cache must not probe CUDA"),
    )
    cache = cute_aot_cache.get_jit_cache(cache_dir=None)
    cache[("key",)] = "compiled"
    assert cache[("key",)] == "compiled"


def test_routine_cache_activity_is_debug_only(caplog, tmp_path):
    cache = cute_aot_cache.JITPersistentCache(tmp_path, enable_tvm_ffi=False)
    with caplog.at_level(logging.INFO, logger=cute_aot_cache.__name__):
        assert ("missing",) not in cache
    assert not caplog.records


def test_factory_namespaces_persistent_cache(monkeypatch, tmp_path):
    fingerprint_calls = []

    def compute_fingerprint(*args):
        fingerprint_calls.append(args)
        return "source-fingerprint"

    monkeypatch.setattr(
        cute_aot_cache,
        "_compute_source_fingerprint",
        compute_fingerprint,
    )
    monkeypatch.setattr(cute_aot_cache, "_resolve_target_arch", lambda: "sm_100")
    cache = cute_aot_cache.get_jit_cache(
        "consumer",
        cache_dir=tmp_path,
        source_paths=(__file__,),
        enable_tvm_ffi=False,
    )
    assert isinstance(cache, cute_aot_cache.JITPersistentCache)
    assert cache.cache_path == tmp_path / "source-fingerprint" / "consumer"
    assert cache.enable_tvm_ffi is False
    assert fingerprint_calls[0][1:] == (False, "sm_100")


def test_disk_key_ignores_cuda_device_ordinal(tmp_path):
    TorchDevice = type("device", (), {"__module__": "torch"})
    first, second = TorchDevice(), TorchDevice()
    first.type, first.index = "cuda", 0
    second.type, second.index = "cuda", 7
    cache = cute_aot_cache.JITPersistentCache(tmp_path, enable_tvm_ffi=False)
    assert cache._key_to_hash((first,)) == cache._key_to_hash((second,))


@pytest.mark.parametrize("enable_tvm_ffi", [False, True], ids=["native", "tvm-ffi"])
def test_persistent_cache_exports_then_loads(monkeypatch, tmp_path, enable_tvm_ffi):
    key = ("shape", 128)
    first = cute_aot_cache.JITPersistentCache(tmp_path, enable_tvm_ffi=enable_tvm_ffi)
    key_hash = first._key_to_hash(key)
    temp_key = f".{key_hash}.tmp"
    temp_path = tmp_path / f"{temp_key}.o"
    export_calls = []

    class Compiled:
        def export_to_c(self, *args, **kwargs):
            export_calls.append((args, kwargs))
            if enable_tvm_ffi:
                assert not args
                assert kwargs == {
                    "object_file_path": str(temp_path),
                    "function_name": first.EXPORT_FUNCTION_PREFIX,
                }
            else:
                assert args == (str(tmp_path), temp_key)
                assert kwargs == {"function_prefix": first.EXPORT_FUNCTION_PREFIX}
            temp_path.write_bytes(b"object")

    first[key] = Compiled()
    object_path = tmp_path / f"{key_hash}.o"
    assert object_path.read_bytes() == b"object"
    assert not temp_path.exists()
    assert len(export_calls) == 1

    loaded = object()
    load_calls = []

    def load_object(path, function_prefix, enable_tvm_ffi):
        load_calls.append((path, function_prefix, enable_tvm_ffi))
        return loaded

    monkeypatch.setattr(cute_aot_cache, "_load_object", load_object)
    second = cute_aot_cache.JITPersistentCache(tmp_path, enable_tvm_ffi=enable_tvm_ffi)
    assert key in second
    assert second[key] is loaded
    assert load_calls == [(object_path, first.EXPORT_FUNCTION_PREFIX, enable_tvm_ffi)]


@pytest.mark.parametrize("enable_tvm_ffi", [False, True], ids=["native", "tvm-ffi"])
def test_failed_export_does_not_publish_partial_object(tmp_path, enable_tvm_ffi):
    key = ("shape", 256)
    cache = cute_aot_cache.JITPersistentCache(tmp_path, enable_tvm_ffi=enable_tvm_ffi)
    key_hash = cache._key_to_hash(key)
    temp_path = tmp_path / f".{key_hash}.tmp.o"

    class FailedCompile:
        def export_to_c(self, *args, **kwargs):
            temp_path.write_bytes(b"partial")
            raise RuntimeError("export failed")

    with pytest.raises(RuntimeError, match="export failed"):
        cache[key] = FailedCompile()

    assert not (tmp_path / f"{key_hash}.o").exists()
    assert not temp_path.exists()


def test_failed_load_discards_invalid_object(monkeypatch, tmp_path):
    key = ("shape", 512)
    cache = cute_aot_cache.JITPersistentCache(tmp_path, enable_tvm_ffi=True)
    object_path = tmp_path / f"{cache._key_to_hash(key)}.o"
    object_path.write_bytes(b"invalid")

    def fail_load(*args, **kwargs):
        raise OSError("invalid object")

    monkeypatch.setattr(cute_aot_cache, "_load_object", fail_load)

    assert key not in cache
    assert not object_path.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
