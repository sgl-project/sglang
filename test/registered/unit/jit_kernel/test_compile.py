"""Unit tests for jit_kernel/utils/compile.py — staged JIT builds (#31347).

On multi-node cold starts with a shared cache directory (e.g. an NFS $HOME),
every rank computes the same tvm-ffi build directory and tvm-ffi's node-local
flock cannot serialize hosts, so concurrent in-place links fail with ESTALE.
These tests verify that `load_jit` builds in a host-private staging directory
and atomically publishes the finished .so, with the tvm-ffi boundary mocked so
no compiler or GPU is required.
"""

import contextlib
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.jit_kernel.utils import compile as jit_compile
from sglang.srt.environ import envs
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextlib.contextmanager
def _jit_test_env(hostname="host-a"):
    """Mock the tvm-ffi/arch boundary and point the cache at a temp dir.

    Yields (cache_dir, mock_cpp, mock_tvm). `build`/`build_inline` create a
    fake .so inside the requested build_directory (like the real ones);
    `load_module` records the path it was asked to load.
    """
    mock_cpp = MagicMock()
    mock_tvm = MagicMock()

    def _fake_build(name, **kwargs):
        build_dir = pathlib.Path(kwargs["build_directory"])
        build_dir.mkdir(parents=True, exist_ok=True)
        so_path = build_dir / f"{name}.so"
        so_path.write_bytes(b"fake-so")
        return str(so_path)

    mock_cpp.build_inline.side_effect = _fake_build
    mock_cpp.build.side_effect = _fake_build
    mock_tvm.load_module.side_effect = lambda path: ("loaded", path)

    with tempfile.TemporaryDirectory() as cache_dir:
        with (
            patch.dict("sys.modules", {"tvm_ffi": mock_tvm, "tvm_ffi.cpp": mock_cpp}),
            patch.object(
                jit_compile,
                "get_jit_cuda_arch",
                return_value=SimpleNamespace(target_name="9.0"),
            ),
            patch.object(jit_compile, "get_default_target_flags", return_value=[]),
            patch.object(jit_compile, "_tvm_ffi_version", return_value="0.0.0"),
            patch.object(jit_compile, "_host_tag", return_value=hostname),
            temp_set_env(TVM_FFI_CACHE_DIR=cache_dir),
        ):
            yield pathlib.Path(cache_dir), mock_cpp, mock_tvm


def _shared_dir(cache_dir: pathlib.Path, marker: str) -> pathlib.Path:
    return cache_dir / f"sgl_kernel_jit_{marker}__arch_9.0__tvmffi_0.0.0"


class TestStagedJitBuild(CustomTestCase):
    def test_cold_build_stages_per_host_and_publishes(self):
        """Cold start: build in a host-private staging dir, publish atomically."""
        with _jit_test_env(hostname="host-a") as (cache_dir, mock_cpp, mock_tvm):
            result = jit_compile.load_jit("stagedmod")

            shared = _shared_dir(cache_dir, "stagedmod")
            staging = shared / "stage__host-a"

            # Built inside the host-private staging dir, not the shared dir.
            mock_cpp.build_inline.assert_called_once()
            self.assertEqual(
                mock_cpp.build_inline.call_args.kwargs["build_directory"],
                str(staging),
            )
            # In-place build APIs must not touch the shared dir.
            mock_cpp.load_inline.assert_not_called()

            # Published: final .so exists at the shared path, staged copy moved.
            final_so = shared / "sgl_kernel_jit_stagedmod.so"
            self.assertTrue(final_so.is_file())
            self.assertFalse((staging / "sgl_kernel_jit_stagedmod.so").exists())

            # The module is loaded from the published path.
            mock_tvm.load_module.assert_called_once_with(str(final_so))
            self.assertEqual(result, ("loaded", str(final_so)))

    def test_different_hosts_get_different_staging_dirs(self):
        """Two hosts never share a mutable build dir, but share the final .so path."""
        staging_dirs = []
        final_sos = []
        for hostname in ("host-a", "host-b"):
            with _jit_test_env(hostname=hostname) as (cache_dir, mock_cpp, _):
                jit_compile.load_jit("crosshost")
                staging_dirs.append(
                    mock_cpp.build_inline.call_args.kwargs["build_directory"]
                )
                final_sos.append(
                    pathlib.Path(
                        mock_cpp.build_inline.call_args.kwargs["build_directory"]
                    ).parent
                    / "sgl_kernel_jit_crosshost.so"
                )
        self.assertNotEqual(staging_dirs[0], staging_dirs[1])
        self.assertTrue(staging_dirs[0].endswith("stage__host-a"))
        self.assertTrue(staging_dirs[1].endswith("stage__host-b"))
        # Both publish to the same shared, content-addressed location.
        self.assertEqual(final_sos[0].name, final_sos[1].name)
        self.assertEqual(final_sos[0].parent.name, final_sos[1].parent.name)

    def test_warm_cache_short_circuits_without_building(self):
        """A published .so is loaded directly; no build API is called."""
        with _jit_test_env() as (cache_dir, mock_cpp, mock_tvm):
            shared = _shared_dir(cache_dir, "warmmod")
            shared.mkdir(parents=True)
            final_so = shared / "sgl_kernel_jit_warmmod.so"
            final_so.write_bytes(b"cached-so")

            result = jit_compile.load_jit("warmmod")

            mock_cpp.build_inline.assert_not_called()
            mock_cpp.load_inline.assert_not_called()
            mock_tvm.load_module.assert_called_once_with(str(final_so))
            self.assertEqual(result, ("loaded", str(final_so)))

    def test_disable_env_restores_in_place_build(self):
        """Kill-switch: build in place in the shared dir, as before."""
        with _jit_test_env() as (cache_dir, mock_cpp, _):
            with envs.SGLANG_DISABLE_JIT_KERNEL_STAGED_BUILD.override(True):
                jit_compile.load_jit("killswitchmod")

            mock_cpp.build_inline.assert_not_called()
            mock_cpp.load_inline.assert_called_once()
            self.assertEqual(
                mock_cpp.load_inline.call_args.kwargs["build_directory"],
                str(_shared_dir(cache_dir, "killswitchmod")),
            )

    def test_explicit_build_directory_bypasses_staging(self):
        """A caller-provided build_directory is respected verbatim (no staging)."""
        with _jit_test_env() as (_, mock_cpp, _):
            with tempfile.TemporaryDirectory() as custom_dir:
                jit_compile.load_jit("explicitmod", build_directory=custom_dir)

                mock_cpp.build_inline.assert_not_called()
                mock_cpp.load_inline.assert_called_once()
                self.assertEqual(
                    mock_cpp.load_inline.call_args.kwargs["build_directory"],
                    custom_dir,
                )

    def test_publish_failure_falls_back_to_staged_so(self):
        """If the atomic publish fails, the staged .so is loaded directly."""
        with _jit_test_env(hostname="host-a") as (cache_dir, _, mock_tvm):
            with patch("os.replace", side_effect=OSError("read-only fs")):
                result = jit_compile.load_jit("fallbackmod")

            staged_so = (
                _shared_dir(cache_dir, "fallbackmod")
                / "stage__host-a"
                / "sgl_kernel_jit_fallbackmod.so"
            )
            mock_tvm.load_module.assert_called_once_with(str(staged_so))
            self.assertEqual(result, ("loaded", str(staged_so)))

    def test_non_header_only_path_also_stages(self):
        """header_only=False goes through build() with the same staging dir."""
        with _jit_test_env(hostname="host-a") as (cache_dir, mock_cpp, mock_tvm):
            jit_compile.load_jit("filemod", header_only=False)

            mock_cpp.build.assert_called_once()
            self.assertEqual(
                mock_cpp.build.call_args.kwargs["build_directory"],
                str(_shared_dir(cache_dir, "filemod") / "stage__host-a"),
            )
            mock_cpp.load.assert_not_called()
            final_so = _shared_dir(cache_dir, "filemod") / "sgl_kernel_jit_filemod.so"
            self.assertTrue(final_so.is_file())
            mock_tvm.load_module.assert_called_once_with(str(final_so))

    def test_host_tag_is_filesystem_safe(self):
        """Hostnames with path-hostile characters are sanitized."""
        with patch("socket.gethostname", return_value="node/1:gpu cluster"):
            self.assertEqual(jit_compile._host_tag(), "node_1_gpu_cluster")
        with patch("socket.gethostname", return_value="node-1.example.com"):
            self.assertEqual(jit_compile._host_tag(), "node-1.example.com")
        with patch("socket.gethostname", return_value=""):
            self.assertEqual(jit_compile._host_tag(), "unknown-host")


if __name__ == "__main__":
    unittest.main()
