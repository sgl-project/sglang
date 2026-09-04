"""Unit tests for the memory-saver subprocess CUDA runtime path discovery.

`--enable-memory-saver` preloads torch_memory_saver's hook .so into scheduler
subprocesses via LD_PRELOAD. The hook links against libcudart but ships
without an RPATH, so in pip CUDA environments the child's dynamic loader
cannot resolve ``libcudart.so.<major>`` unless the wheel runtime directory is
on LD_LIBRARY_PATH (issue #36533). These tests cover the pure path-discovery
helpers on CPU only; no CUDA, server, or model loading involved.
"""

import sys

import pytest

from sglang.srt.utils import torch_memory_saver_adapter as adapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_lib(tmp_path, *parts):
    lib_dir = tmp_path.joinpath(*parts[:-1])
    lib_dir.mkdir(parents=True, exist_ok=True)
    (lib_dir / parts[-1]).touch()
    return lib_dir


class TestCudartLibDirs:
    def test_cu13_consolidated_layout(self, tmp_path):
        # CUDA 13 pip wheels install into a single nvidia/cu13 tree.
        lib_dir = _make_lib(tmp_path, "nvidia", "cu13", "lib", "libcudart.so.13")
        _make_lib(tmp_path, "nvidia", "cu13", "include", "cuda.h")

        assert adapter._cudart_lib_dirs([tmp_path / "nvidia"]) == [str(lib_dir)]

    def test_cu12_component_layout_filters_non_runtime(self, tmp_path):
        # CUDA 12 pip wheels install one component per directory; only the
        # cuda_runtime component holds libcudart and should be returned.
        runtime = _make_lib(
            tmp_path, "nvidia", "cuda_runtime", "lib", "libcudart.so.12"
        )
        _make_lib(tmp_path, "nvidia", "cublas", "lib", "libcublas.so.12")
        _make_lib(tmp_path, "nvidia", "cudnn", "lib", "libcudnn.so.9")

        assert adapter._cudart_lib_dirs([tmp_path / "nvidia"]) == [str(runtime)]

    def test_multiple_roots_and_majors_deduplicated(self, tmp_path):
        root_a = tmp_path / "site_a" / "nvidia"
        root_b = tmp_path / "site_b" / "nvidia"
        dir_cu12 = _make_lib(
            tmp_path, "site_a", "nvidia", "cuda_runtime", "lib", "libcudart.so.12"
        )
        dir_cu13 = _make_lib(
            tmp_path, "site_b", "nvidia", "cu13", "lib", "libcudart.so.13"
        )

        result = adapter._cudart_lib_dirs([root_a, root_b, root_a])

        assert result == [str(dir_cu12), str(dir_cu13)]

    def test_missing_or_empty_roots(self, tmp_path):
        empty = tmp_path / "nvidia"
        empty.mkdir()
        _make_lib(tmp_path, "nvidia", "nvjitlink", "lib", "libnvJitLink.so.13")

        assert adapter._cudart_lib_dirs([]) == []
        assert adapter._cudart_lib_dirs([tmp_path / "does_not_exist"]) == []
        assert adapter._cudart_lib_dirs([empty]) == []


class TestPrependLdLibraryPath:
    def test_without_existing_value(self):
        assert adapter._prepend_ld_library_path(["/a", "/b"], None) == "/a:/b"
        assert adapter._prepend_ld_library_path(["/a"], "") == "/a"

    def test_preserves_existing_entries(self):
        result = adapter._prepend_ld_library_path(["/nv/lib"], "/usr/lib:/opt/lib")
        assert result == "/nv/lib:/usr/lib:/opt/lib"

    def test_deduplicates_and_drops_empty_entries(self):
        result = adapter._prepend_ld_library_path(
            ["/nv/lib", "/nv/lib"], "/usr/lib::/nv/lib:"
        )
        assert result == "/nv/lib:/usr/lib"


class TestCudaRuntimeLdLibraryPath:
    @pytest.fixture
    def fake_discovery(self, monkeypatch, tmp_path):
        lib_dir = _make_lib(tmp_path, "nvidia", "cu13", "lib", "libcudart.so.13")
        monkeypatch.setattr(adapter, "_loaded_libcudart_dirs", lambda: [])
        monkeypatch.setattr(adapter, "_pip_nvidia_roots", lambda: [tmp_path / "nvidia"])
        return str(lib_dir)

    def test_sets_and_unsets_when_absent(self, monkeypatch, fake_discovery):
        import os

        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        with adapter._cuda_runtime_ld_library_path():
            assert os.environ["LD_LIBRARY_PATH"] == fake_discovery
        assert "LD_LIBRARY_PATH" not in os.environ

    def test_prepends_and_restores_existing_value(self, monkeypatch, fake_discovery):
        import os

        monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/lib")
        with adapter._cuda_runtime_ld_library_path():
            assert os.environ["LD_LIBRARY_PATH"] == f"{fake_discovery}:/usr/lib"
        assert os.environ["LD_LIBRARY_PATH"] == "/usr/lib"

    def test_noop_when_nothing_discovered(self, monkeypatch):
        import os

        monkeypatch.setattr(adapter, "_loaded_libcudart_dirs", lambda: [])
        monkeypatch.setattr(adapter, "_pip_nvidia_roots", lambda: [])
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
        with adapter._cuda_runtime_ld_library_path():
            assert "LD_LIBRARY_PATH" not in os.environ
        assert "LD_LIBRARY_PATH" not in os.environ

    def test_noop_when_already_present(self, monkeypatch, fake_discovery):
        import os

        monkeypatch.setenv("LD_LIBRARY_PATH", fake_discovery)
        with adapter._cuda_runtime_ld_library_path():
            assert os.environ["LD_LIBRARY_PATH"] == fake_discovery
        assert os.environ["LD_LIBRARY_PATH"] == fake_discovery


class TestDiscoverySmoke:
    def test_discovery_helpers_run_on_any_platform(self):
        # CPU machines and machines without /proc must degrade to empty
        # results, never raise.
        assert isinstance(adapter._loaded_libcudart_dirs(), list)
        roots = adapter._pip_nvidia_roots()
        assert isinstance(roots, list)
        assert isinstance(adapter._cudart_lib_dirs(roots), list)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
