import io
import os

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

from sglang.srt.distributed.device_communicators import cuda_wrapper


def test_find_loaded_library_prefers_real_cudart_over_tilelang_stub(monkeypatch):
    maps = """\
7f000000-7f010000 r-xp 00000000 00:00 0 /site-packages/tilelang/lib/libcudart_stub.so
7f020000-7f030000 r-xp 00000000 00:00 0 /cuda/lib64/libcudart.so.13
"""

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(maps))

    assert (
        cuda_wrapper.find_loaded_library("libcudart") == "/cuda/lib64/libcudart.so.13"
    )
    assert cuda_wrapper.find_loaded_library(
        "libcudart", require_unique=True
    ) == os.path.realpath("/cuda/lib64/libcudart.so.13")


def test_find_loaded_library_unique_deduplicates_vmas(monkeypatch):
    maps = """\
7f000000-7f010000 r--p 00000000 00:00 0 /rocm/lib/libamdhip64.so.7
7f010000-7f020000 r-xp 00000000 00:00 0 /rocm/lib/libamdhip64.so.7
"""

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(maps))

    assert cuda_wrapper.find_loaded_library(
        "libamdhip64", require_unique=True
    ) == os.path.realpath("/rocm/lib/libamdhip64.so.7")


def test_find_loaded_library_unique_rejects_distinct_runtimes(monkeypatch):
    maps = """\
7f000000-7f010000 r-xp 00000000 00:00 0 /runtime/libamdhip64.so.7
7f020000-7f030000 r-xp 00000000 00:00 0 /toolchain/libamdhip64.so
"""

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(maps))

    assert cuda_wrapper.find_loaded_library("libamdhip64", require_unique=True) is None


def test_find_loaded_library_strips_deleted_suffix(monkeypatch):
    maps = """\
7f020000-7f030000 r-xp 00000000 00:00 0 /cuda/lib64/libcudart.so.13 (deleted)
"""

    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(maps))

    assert (
        cuda_wrapper.find_loaded_library("libcudart") == "/cuda/lib64/libcudart.so.13"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
