"""Hardware allocator admission and exact-geometry workspace query caching."""

import ctypes
import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

_path = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/weight_sync/gpu_delta_codec.py"
)
_spec = importlib.util.spec_from_file_location("gpu_delta_codec_under_test", _path)
codec = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = codec
_spec.loader.exec_module(codec)


@pytest.mark.parametrize(
    "backend,settings,segments,error",
    [
        ("cudaMallocAsync", False, [], "native CUDA"),
        ("native", True, [], "expandable_segments"),
        ("native", None, [], "expandable_segments"),
        ("native", False, [{"device": 0, "is_expandable": True}], "existing CUDA VMM"),
        ("native", False, [{"device": 1, "is_expandable": True}], None),
    ],
)
def test_hardware_allocator_uses_effective_settings(
    monkeypatch, backend, settings, segments, error
):
    monkeypatch.setattr(torch.cuda.memory, "get_allocator_backend", lambda: backend)
    monkeypatch.setattr(
        torch.cuda.memory,
        "_snapshot",
        lambda: {
            "allocator_settings": {"expandable_segments": settings},
            "segments": segments,
        },
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            codec._require_hardware_allocator(torch.device("cuda", 0))
    else:
        codec._require_hardware_allocator(torch.device("cuda", 0))


def test_workspace_query_cache_uses_exact_geometry_and_preserves_failure(monkeypatch):
    # Exercise the real query/cache code without a CUDA allocation or library.
    decoder = object.__new__(codec.NvcompDecoder)
    decoder.device = torch.device("cuda", 0)
    decoder.backend = "hardware"
    decoder._options = codec._SnappyOptions()
    decoder._temporary_sizes = {}
    calls, fail = [], True

    def query(count, maximum, options, size, total):
        calls.append((count, maximum, total))
        if total == 31 and fail:
            return 1
        # Zero is also a valid cache entry; it must not trigger another query.
        ctypes.cast(size, ctypes.POINTER(ctypes.c_size_t))[0] = 0
        return 0

    decoder._temporary = query
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())

    def frames(*lengths):
        return [codec.DecodeFrame(0, 1, 0, length) for length in lengths]

    assert decoder.temporary_bytes([]) == 0
    assert decoder.temporary_bytes(frames(10, 20)) == 0
    assert decoder.temporary_bytes(frames(20, 10)) == 0
    assert calls == [(2, 20, 30)]
    # Same count and maximum, different total: the native query must run again.
    with pytest.raises(RuntimeError, match="status=1"):
        decoder.temporary_bytes(frames(11, 20))
    fail = False
    assert decoder.temporary_bytes(frames(11, 20)) == 0
    assert decoder.temporary_bytes(frames(11, 20)) == 0
    assert calls == [(2, 20, 30), (2, 20, 31), (2, 20, 31)]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
