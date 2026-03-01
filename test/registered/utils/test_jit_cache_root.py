"""
Unit tests for the unified JIT cache directory feature (SGLANG_JIT_CACHE_ROOT).

Tests that all cache consumers correctly derive their paths from the unified
root, backward-compat env vars are honoured, and external-library env vars
(TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR) are propagated.
"""

import os
import tempfile
import unittest

from sglang.srt.environ import (
    configure_jit_cache_env_vars,
    envs,
    get_jit_cache_subdir,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")


class TestGetJitCacheSubdir(unittest.TestCase):
    """Tests for the get_jit_cache_subdir() helper."""

    def setUp(self):
        # Save and clear relevant env vars before each test
        self._saved = {}
        for key in (
            "SGLANG_JIT_CACHE_ROOT",
            "SGLANG_CACHE_DIR",
            "SGLANG_DG_CACHE_DIR",
        ):
            self._saved[key] = os.environ.pop(key, None)

    def tearDown(self):
        # Restore env vars
        for key, val in self._saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_default_root(self):
        """Without any env var set, subdir is under ~/.cache/sglang."""
        result = get_jit_cache_subdir("deep_gemm")
        expected = os.path.join(os.path.expanduser("~/.cache/sglang"), "deep_gemm")
        self.assertEqual(result, expected)

    def test_custom_root(self):
        """SGLANG_JIT_CACHE_ROOT redirects all subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            envs.SGLANG_JIT_CACHE_ROOT.set(tmpdir)
            try:
                result = get_jit_cache_subdir("torch_compile_cache")
                self.assertEqual(result, os.path.join(tmpdir, "torch_compile_cache"))
            finally:
                envs.SGLANG_JIT_CACHE_ROOT.clear()

    def test_override_env_takes_precedence(self):
        """When an override env var is explicitly set, it wins."""
        custom_path = "/tmp/my_custom_dg_cache"
        os.environ["SGLANG_DG_CACHE_DIR"] = custom_path
        try:
            result = get_jit_cache_subdir(
                "deep_gemm", override_env="SGLANG_DG_CACHE_DIR"
            )
            self.assertEqual(result, custom_path)
        finally:
            del os.environ["SGLANG_DG_CACHE_DIR"]

    def test_override_env_not_set_falls_back_to_root(self):
        """When override env var is not set, falls back to SGLANG_JIT_CACHE_ROOT."""
        with tempfile.TemporaryDirectory() as tmpdir:
            envs.SGLANG_JIT_CACHE_ROOT.set(tmpdir)
            try:
                # SGLANG_DG_CACHE_DIR is NOT in os.environ
                result = get_jit_cache_subdir(
                    "deep_gemm", override_env="SGLANG_DG_CACHE_DIR"
                )
                self.assertEqual(result, os.path.join(tmpdir, "deep_gemm"))
            finally:
                envs.SGLANG_JIT_CACHE_ROOT.clear()

    def test_override_env_with_tilde(self):
        """Tilde in override env var is expanded."""
        os.environ["SGLANG_DG_CACHE_DIR"] = "~/my_cache/dg"
        try:
            result = get_jit_cache_subdir(
                "deep_gemm", override_env="SGLANG_DG_CACHE_DIR"
            )
            self.assertEqual(result, os.path.expanduser("~/my_cache/dg"))
        finally:
            del os.environ["SGLANG_DG_CACHE_DIR"]

    def test_multiple_subdirs_share_root(self):
        """Different subdirectories all live under the same root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            envs.SGLANG_JIT_CACHE_ROOT.set(tmpdir)
            try:
                dg = get_jit_cache_subdir("deep_gemm")
                tc = get_jit_cache_subdir("torch_compile_cache")
                self.assertTrue(dg.startswith(tmpdir))
                self.assertTrue(tc.startswith(tmpdir))
                self.assertNotEqual(dg, tc)
            finally:
                envs.SGLANG_JIT_CACHE_ROOT.clear()


class TestConfigureJitCacheEnvVars(unittest.TestCase):
    """Tests for configure_jit_cache_env_vars()."""

    def setUp(self):
        self._saved = {}
        for key in (
            "SGLANG_JIT_CACHE_ROOT",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
        ):
            self._saved[key] = os.environ.pop(key, None)

    def tearDown(self):
        for key, val in self._saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_sets_triton_cache_dir(self):
        """configure_jit_cache_env_vars sets TRITON_CACHE_DIR when absent."""
        configure_jit_cache_env_vars()
        root = envs.SGLANG_JIT_CACHE_ROOT.get()
        self.assertEqual(
            os.environ["TRITON_CACHE_DIR"],
            os.path.join(root, "triton"),
        )

    def test_sets_inductor_cache_dir(self):
        """configure_jit_cache_env_vars sets TORCHINDUCTOR_CACHE_DIR when absent."""
        configure_jit_cache_env_vars()
        root = envs.SGLANG_JIT_CACHE_ROOT.get()
        self.assertEqual(
            os.environ["TORCHINDUCTOR_CACHE_DIR"],
            os.path.join(root, "inductor"),
        )

    def test_does_not_override_existing_triton_cache_dir(self):
        """If user already set TRITON_CACHE_DIR, we don't overwrite it."""
        os.environ["TRITON_CACHE_DIR"] = "/my/triton"
        configure_jit_cache_env_vars()
        self.assertEqual(os.environ["TRITON_CACHE_DIR"], "/my/triton")

    def test_does_not_override_existing_inductor_cache_dir(self):
        """If user already set TORCHINDUCTOR_CACHE_DIR, we don't overwrite it."""
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/my/inductor"
        configure_jit_cache_env_vars()
        self.assertEqual(os.environ["TORCHINDUCTOR_CACHE_DIR"], "/my/inductor")

    def test_custom_root_propagates_to_external(self):
        """Custom SGLANG_JIT_CACHE_ROOT propagates to Triton/Inductor."""
        with tempfile.TemporaryDirectory() as tmpdir:
            envs.SGLANG_JIT_CACHE_ROOT.set(tmpdir)
            try:
                configure_jit_cache_env_vars()
                self.assertEqual(
                    os.environ["TRITON_CACHE_DIR"],
                    os.path.join(tmpdir, "triton"),
                )
                self.assertEqual(
                    os.environ["TORCHINDUCTOR_CACHE_DIR"],
                    os.path.join(tmpdir, "inductor"),
                )
            finally:
                envs.SGLANG_JIT_CACHE_ROOT.clear()


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure legacy env vars still work."""

    def setUp(self):
        self._saved = {}
        for key in (
            "SGLANG_JIT_CACHE_ROOT",
            "SGLANG_CACHE_DIR",
            "SGLANG_DG_CACHE_DIR",
        ):
            self._saved[key] = os.environ.pop(key, None)

    def tearDown(self):
        for key, val in self._saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_torch_compile_default_path_unchanged(self):
        """Default torch_compile cache path is still under ~/.cache/sglang/torch_compile_cache."""
        result = get_jit_cache_subdir("torch_compile_cache")
        expected = os.path.join(
            os.path.expanduser("~/.cache/sglang"), "torch_compile_cache"
        )
        self.assertEqual(result, expected)

    def test_gpu_p2p_default_path_unchanged(self):
        """Default GPU P2P cache is still under ~/.cache/sglang."""
        root = envs.SGLANG_JIT_CACHE_ROOT.get()
        self.assertEqual(root, os.path.expanduser("~/.cache/sglang"))

    def test_sglang_cache_dir_is_honored(self):
        """Legacy SGLANG_CACHE_DIR is remapped to SGLANG_JIT_CACHE_ROOT."""
        from sglang.srt.environ import _convert_SGL_to_SGLANG

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["SGLANG_CACHE_DIR"] = tmpdir
            try:
                _convert_SGL_to_SGLANG()
                self.assertEqual(os.environ.get("SGLANG_JIT_CACHE_ROOT"), tmpdir)
            finally:
                os.environ.pop("SGLANG_CACHE_DIR", None)
                os.environ.pop("SGLANG_JIT_CACHE_ROOT", None)


if __name__ == "__main__":
    unittest.main(verbosity=3)
