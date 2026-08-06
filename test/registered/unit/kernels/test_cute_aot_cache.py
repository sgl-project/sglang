import pytest

from sglang.kernels.jit import cute_aot_cache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_disabled_factory_returns_process_local_cache():
    cache = cute_aot_cache.get_jit_cache(cache_dir=None)
    cache[("key",)] = "compiled"
    assert cache.get(("key",)) == "compiled"


def test_factory_namespaces_persistent_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(
        cute_aot_cache,
        "_compute_source_fingerprint",
        lambda *args: "source-fingerprint",
    )
    cache = cute_aot_cache.get_jit_cache(
        "consumer",
        cache_dir=tmp_path,
        source_paths=(__file__,),
        enable_tvm_ffi=False,
    )
    assert isinstance(cache, cute_aot_cache.JITPersistentCache)
    assert cache.cache_path == tmp_path / "source-fingerprint" / "consumer"
    assert cache.enable_tvm_ffi is False


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
    prefix = first._function_prefix(key_hash)
    export_calls = []

    class Compiled:
        def export_to_c(self, *args, **kwargs):
            export_calls.append((args, kwargs))
            if enable_tvm_ffi:
                assert not args
                assert kwargs == {
                    "object_file_path": str(tmp_path / f"{key_hash}.o"),
                    "function_name": prefix,
                }
            else:
                assert args == (str(tmp_path), key_hash)
                assert kwargs == {"function_prefix": prefix}
            (tmp_path / f"{key_hash}.o").write_bytes(b"object")

    first[key] = Compiled()
    object_path = tmp_path / f"{key_hash}.o"
    assert object_path.read_bytes() == b"object"
    assert len(export_calls) == 1

    loaded = object()
    load_calls = []

    def load_object(path, function_prefix, *, enable_tvm_ffi):
        load_calls.append((path, function_prefix, enable_tvm_ffi))
        return loaded

    monkeypatch.setattr(cute_aot_cache, "_load_object", load_object)
    second = cute_aot_cache.JITPersistentCache(tmp_path, enable_tvm_ffi=enable_tvm_ffi)
    assert key in second
    assert second[key] is loaded
    assert load_calls == [(object_path, prefix, enable_tvm_ffi)]
