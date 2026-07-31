"""Unit tests for kernels/jit/utils/compile.py — staged JIT builds (#31347).

On multi-node cold starts with a shared cache directory (e.g. an NFS $HOME),
every rank computes the same tvm-ffi build directory and tvm-ffi's node-local
flock cannot serialize hosts, so concurrent in-place links fail with ESTALE.
These tests verify that `load_jit` builds in a host-private staging directory
and atomically publishes the finished .so, with the tvm-ffi boundary mocked so
no compiler or GPU is required.
"""

import contextlib
import pathlib
import re
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.kernels.jit.utils import compile as jit_compile
from sglang.srt.environ import envs
from sglang.srt.utils.common import temp_set_env
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeFileLock:
    """Stand in for tvm_ffi.utils.FileLock: creates the lock file, no locking."""

    def __init__(self, lock_file_path):
        self.lock_file_path = lock_file_path

    def __enter__(self):
        pathlib.Path(self.lock_file_path).touch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


@contextlib.contextmanager
def _jit_test_env(hostname="host-a"):
    """Mock the tvm-ffi/arch boundary and point the cache at a temp dir.

    Yields (cache_dir, mock_cpp, mock_tvm). `build`/`build_inline` create a
    fake .so inside the requested build_directory (like the real ones);
    `load_module` records the path it was asked to load.
    """
    mock_cpp = MagicMock()
    mock_tvm = MagicMock()
    mock_utils = SimpleNamespace(FileLock=_FakeFileLock)

    def _fake_build(name, **kwargs):
        build_dir = pathlib.Path(kwargs["build_directory"])
        build_dir.mkdir(parents=True, exist_ok=True)
        # Leave object and ninja files behind like a real build does, so tests
        # can check what remains in the staging directory after publishing.
        (build_dir / "main.o").write_bytes(b"fake-object")
        (build_dir / "build.ninja").write_text("fake ninja file")
        so_path = build_dir / f"{name}.so"
        so_path.write_bytes(b"fake-so")
        return str(so_path)

    mock_cpp.build_inline.side_effect = _fake_build
    mock_cpp.build.side_effect = _fake_build
    mock_tvm.load_module.side_effect = lambda path: ("loaded", path)

    with tempfile.TemporaryDirectory() as cache_dir:
        with (
            patch.dict(
                "sys.modules",
                {
                    "tvm_ffi": mock_tvm,
                    "tvm_ffi.cpp": mock_cpp,
                    "tvm_ffi.utils": mock_utils,
                },
            ),
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
            with (
                patch("os.link", side_effect=OSError("filesystem is read only")),
                patch("os.replace", side_effect=OSError("filesystem is read only")),
            ):
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

    def test_second_publisher_keeps_first_published_library(self):
        """A later publisher must not replace a library that is already published.

        Two hosts race the same cold start. The first publish wins. The second
        publish must leave the shared file and its inode untouched, so a host
        that already loaded the file never sees it change underneath it, and
        the second publisher must still get a loaded module.
        """
        with _jit_test_env(hostname="host-a") as (cache_dir, _, mock_tvm):
            jit_compile.load_jit("racemod")
            shared = _shared_dir(cache_dir, "racemod")
            final_so = shared / "sgl_kernel_jit_racemod.so"
            first_inode = final_so.stat().st_ino
            first_bytes = final_so.read_bytes()

            # A second host built the same module in its own staging directory
            # and publishes after the first host already did.
            staging_b = shared / "stage__host-b"
            staging_b.mkdir()
            staged_so = staging_b / "sgl_kernel_jit_racemod.so"
            staged_so.write_bytes(first_bytes)

            mock_tvm.load_module.reset_mock()
            result = jit_compile._publish_and_load(str(staged_so), final_so)

            self.assertEqual(final_so.stat().st_ino, first_inode)
            self.assertEqual(final_so.read_bytes(), first_bytes)
            mock_tvm.load_module.assert_called_once_with(str(final_so))
            self.assertEqual(result, ("loaded", str(final_so)))

    def test_library_published_during_the_compile_is_kept_and_loaded(self):
        """A library that appears mid compile is kept, and this host loads it.

        Two hosts start cold at the same time and the other host publishes
        while this host is still compiling. This host must leave the published
        library alone, because another host may already have loaded it, and it
        must load that published copy instead of its own staged one.
        """
        with _jit_test_env(hostname="host-a") as (cache_dir, mock_cpp, mock_tvm):
            shared = _shared_dir(cache_dir, "midflightmod")
            build_staged_so = mock_cpp.build_inline.side_effect

            def _build_then_other_host_publishes(name, **kwargs):
                staged_so = build_staged_so(name, **kwargs)
                shared.mkdir(parents=True, exist_ok=True)
                (shared / f"{name}.so").write_bytes(b"published-by-host-b")
                return staged_so

            mock_cpp.build_inline.side_effect = _build_then_other_host_publishes

            result = jit_compile.load_jit("midflightmod")

            final_so = shared / "sgl_kernel_jit_midflightmod.so"
            self.assertEqual(final_so.read_bytes(), b"published-by-host-b")
            mock_tvm.load_module.assert_called_once_with(str(final_so))
            self.assertEqual(result, ("loaded", str(final_so)))
            # The staged copy is cleaned up rather than left behind.
            staging = shared / "stage__host-a"
            self.assertEqual(
                sorted(entry.name for entry in staging.iterdir()), ["lock"]
            )

    def test_lost_race_publish_loads_the_published_target(self):
        """A rank whose staged file was already published loads the target.

        Several ranks on one host share a staging directory. The first rank
        moves the staged library to the shared target, so the next rank finds
        its staged file missing. The publish step must then load the already
        published target instead of failing on the missing staged path.
        """
        with _jit_test_env(hostname="host-a") as (cache_dir, _, mock_tvm):
            shared = _shared_dir(cache_dir, "lostmod")
            staging = shared / "stage__host-a"
            staging.mkdir(parents=True)
            final_so = shared / "sgl_kernel_jit_lostmod.so"
            final_so.write_bytes(b"fake-so")
            missing_staged = staging / "sgl_kernel_jit_lostmod.so"

            result = jit_compile._publish_and_load(str(missing_staged), final_so)

            mock_tvm.load_module.assert_called_once_with(str(final_so))
            self.assertEqual(result, ("loaded", str(final_so)))

    def test_publish_with_nothing_to_load_raises_clear_error(self):
        """If both the staged file and the target are missing, raise clearly."""
        with _jit_test_env(hostname="host-a") as (cache_dir, _, _):
            shared = _shared_dir(cache_dir, "gonemod")
            staging = shared / "stage__host-a"
            staging.mkdir(parents=True)
            missing_staged = staging / "sgl_kernel_jit_gonemod.so"
            final_so = shared / "sgl_kernel_jit_gonemod.so"

            with self.assertRaises(RuntimeError):
                jit_compile._publish_and_load(str(missing_staged), final_so)

    def test_bad_cached_library_is_rebuilt_and_replaced(self):
        """A cached .so that fails to load is rebuilt and overwritten."""
        with _jit_test_env(hostname="host-a") as (cache_dir, mock_cpp, mock_tvm):
            shared = _shared_dir(cache_dir, "badmod")
            shared.mkdir(parents=True)
            final_so = shared / "sgl_kernel_jit_badmod.so"
            final_so.write_bytes(b"corrupt-so")

            def _load(path):
                if pathlib.Path(path).read_bytes() == b"corrupt-so":
                    raise RuntimeError("bad library")
                return ("loaded", path)

            mock_tvm.load_module.side_effect = _load

            result = jit_compile.load_jit("badmod")

            mock_cpp.build_inline.assert_called_once()
            self.assertEqual(final_so.read_bytes(), b"fake-so")
            self.assertEqual(result, ("loaded", str(final_so)))

    def test_successful_publish_cleans_staging_directory(self):
        """After a successful publish only the lock file stays in staging.

        The lock file must be kept: deleting it while another process still
        holds it open would let two later builders lock different files and
        build concurrently in the same directory.
        """
        with _jit_test_env(hostname="host-a") as (cache_dir, _, _):
            jit_compile.load_jit("cleanmod")
            staging = _shared_dir(cache_dir, "cleanmod") / "stage__host-a"
            leftovers = sorted(entry.name for entry in staging.iterdir())
            self.assertEqual(leftovers, ["lock"])

    def test_host_tag_is_filesystem_safe(self):
        """Hostnames with path-hostile characters are sanitized."""
        with patch("socket.gethostname", return_value="node/1:gpu cluster"):
            self.assertEqual(jit_compile._host_tag(), "node_1_gpu_cluster")
        with patch("socket.gethostname", return_value="node-1.example.com"):
            self.assertEqual(jit_compile._host_tag(), "node-1.example.com")

    def test_host_tag_falls_back_when_hostname_is_unavailable(self):
        """A host name lookup that fails or is empty must not break the build.

        The fallback stays unique to this process on purpose. One fixed
        fallback name would put every host that cannot report a name back into
        a single shared staging directory, which is exactly the cross host
        build race this staging step exists to prevent.
        """
        for failure in (
            {"return_value": ""},
            {"side_effect": OSError("host name lookup is not permitted")},
        ):
            with self.subTest(failure=failure):
                with patch("socket.gethostname", **failure):
                    tag = jit_compile._host_tag()

                    # Recognizable, but not a value another host can share.
                    self.assertTrue(tag.startswith("unknown-host"))
                    self.assertNotEqual(tag, "unknown-host")
                    # Still a usable directory name.
                    self.assertIsNotNone(re.fullmatch(r"[0-9A-Za-z._-]+", tag))
                    # Stable within this process, so repeated loads on one
                    # host reuse a single staging directory.
                    self.assertEqual(tag, jit_compile._host_tag())


if __name__ == "__main__":
    unittest.main()
