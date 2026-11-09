"""Protect the simulator-only macOS compatibility boundaries."""

import subprocess
import sys
import textwrap

import pytest
from sglang_simulator.simulation.sglang import macos_compat


def run_python(code):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_macos_kernel_imports_fail_on_execution():
    run_python(
        """
        import os
        import sys
        from sglang_simulator.simulation.sglang.sgl_kernel_hook import install_load_utils_stub
        sys.platform = "darwin"
        os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
        install_load_utils_stub()
        import sgl_kernel
        from sgl_kernel import copy_all_layer_kv_cache_cpu
        from sglang_simulator.simulation.sglang.sgl_kernel_hook import _MACOS_IMPORT_ONLY_KERNELS
        assert sgl_kernel.__spec__.submodule_search_locations is not None
        assert not hasattr(sgl_kernel, "__file__")
        assert not hasattr(sgl_kernel, "unknown_kernel")
        assert not hasattr(sgl_kernel, "metal")
        assert not hasattr(sgl_kernel, "__getattr__")
        assert sgl_kernel.copy_all_layer_kv_cache_cpu is copy_all_layer_kv_cache_cpu
        install_load_utils_stub()
        assert sys.modules["sgl_kernel"] is sgl_kernel
        for name in _MACOS_IMPORT_ONLY_KERNELS:
            try:
                getattr(sgl_kernel, name)()
            except RuntimeError as exc:
                assert name in str(exc)
            else:
                raise AssertionError("Native kernel execution must not silently succeed")
        """
    )


@pytest.mark.parametrize("platform,cpu", [("linux", "1"), ("darwin", "0")])
def test_other_modes_do_not_replace_kernel_package(platform, cpu):
    run_python(
        f"""
        import os
        import sys
        from sglang_simulator.simulation.sglang.sgl_kernel_hook import install_load_utils_stub
        from sglang_simulator.simulation.sglang.macos_compat import install_macos_cpu_compat
        sys.platform = {platform!r}
        os.environ["SGLANG_USE_CPU_ENGINE"] = {cpu!r}
        install_load_utils_stub()
        install_macos_cpu_compat()
        assert "sgl_kernel" not in sys.modules
        assert "sglang.srt.distributed.bootstrap" not in sys.modules
        """
    )


def test_preserve_already_loaded_kernel_package():
    run_python(
        """
        import os
        import sys
        import types
        from sglang_simulator.simulation.sglang.sgl_kernel_hook import install_load_utils_stub
        sys.platform = "darwin"
        os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
        package = types.ModuleType("sgl_kernel")
        sys.modules["sgl_kernel"] = package
        install_load_utils_stub()
        assert sys.modules["sgl_kernel"] is package
        assert not hasattr(package, "__getattr__")
        """
    )


@pytest.mark.parametrize("field", ["tp_size", "pp_size", "dp_size"])
def test_native_cpu_setup_rejects_multiple_workers(field):
    run_python(
        f"""
        import sys
        import types
        from sglang_simulator.simulation.sglang.macos_compat import _skip_native_cpu_threads_env
        parallel = types.SimpleNamespace(tp_size=1, pp_size=1, dp_size=1)
        context = types.ModuleType("sglang.srt.runtime_context")
        context.get_parallel = lambda: parallel
        sys.modules[context.__name__] = context
        _skip_native_cpu_threads_env()
        setattr(parallel, {field!r}, 2)
        try:
            _skip_native_cpu_threads_env()
        except RuntimeError as exc:
            assert "single runtime worker" in str(exc)
        else:
            raise AssertionError("Multiple workers must not bypass native setup")
        """
    )


def test_host_topology_does_not_use_linux_tools(monkeypatch):
    import psutil

    monkeypatch.setattr(psutil, "cpu_count", lambda logical: 4)
    assert macos_compat._host_cpu_ids_by_node() == {0: {0, 1, 2, 3}}
    monkeypatch.setattr(psutil, "cpu_count", lambda logical: None)
    monkeypatch.setattr(macos_compat.os, "cpu_count", lambda: None)
    assert macos_compat._host_cpu_ids_by_node() == {0: {0}}
