import os
import unittest

from sglang.srt.environ import envs, get_jit_cache_subdir
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")


class TestGetJitCacheSubdir(unittest.TestCase):
    """Verify the priority logic of get_jit_cache_subdir().

    Priority (highest to lowest):
      1. Per-component override env var (e.g. SGLANG_DG_CACHE_DIR)
      2. SGLANG_JIT_CACHE_ROOT / <subdir>
      3. Default root (~/.cache/sglang) / <subdir>
    """

    def setUp(self):
        self._env_backup = {}
        self._keys = [
            "SGLANG_JIT_CACHE_ROOT",
            "SGLANG_DG_CACHE_DIR",
            "SGLANG_CACHE_DIR",
            "XDG_CACHE_HOME",
            "_TEST_OVERRIDE",
        ]
        for k in self._keys:
            self._env_backup[k] = os.environ.pop(k, None)

    def tearDown(self):
        for k in self._keys:
            if self._env_backup[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = self._env_backup[k]

    def test_default_root(self):
        """No env vars set -> uses ~/.cache/sglang/<subdir>."""
        result = get_jit_cache_subdir("deep_gemm")
        expected = os.path.join(os.path.expanduser("~/.cache"), "sglang", "deep_gemm")
        self.assertEqual(result, expected)

    def test_custom_root(self):
        """SGLANG_JIT_CACHE_ROOT set -> uses that root."""
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/data/my_cache"
        result = get_jit_cache_subdir("deep_gemm")
        self.assertEqual(result, "/data/my_cache/deep_gemm")

    def test_xdg_cache_home(self):
        """XDG_CACHE_HOME set (without SGLANG_JIT_CACHE_ROOT) -> uses XDG path.

        Note: XDG_CACHE_HOME is read at import time for the default value,
        so this test verifies the SGLANG_JIT_CACHE_ROOT override instead.
        """
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/xdg_root/sglang"
        result = get_jit_cache_subdir("triton")
        self.assertEqual(result, "/xdg_root/sglang/triton")

    def test_override_env_takes_precedence(self):
        """Per-component override env var wins over SGLANG_JIT_CACHE_ROOT."""
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/data/root"
        os.environ["_TEST_OVERRIDE"] = "/custom/override_path"
        result = get_jit_cache_subdir("deep_gemm", override_env="_TEST_OVERRIDE")
        self.assertEqual(result, "/custom/override_path")

    def test_override_env_empty_falls_through(self):
        """Empty override env var is ignored -> falls through to root."""
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/data/root"
        os.environ["_TEST_OVERRIDE"] = ""
        result = get_jit_cache_subdir("deep_gemm", override_env="_TEST_OVERRIDE")
        self.assertEqual(result, "/data/root/deep_gemm")

    def test_override_env_not_set_falls_through(self):
        """Unset override env var -> falls through to root."""
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/data/root"
        os.environ.pop("_TEST_OVERRIDE", None)
        result = get_jit_cache_subdir("deep_gemm", override_env="_TEST_OVERRIDE")
        self.assertEqual(result, "/data/root/deep_gemm")

    def test_no_override_env_param(self):
        """override_env=None -> always uses root."""
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/data/root"
        result = get_jit_cache_subdir("torch_compile")
        self.assertEqual(result, "/data/root/torch_compile")

    def test_override_with_tilde(self):
        """Override env var with ~ is expanded."""
        os.environ["_TEST_OVERRIDE"] = "~/my_cache/dg"
        result = get_jit_cache_subdir("deep_gemm", override_env="_TEST_OVERRIDE")
        self.assertEqual(result, os.path.expanduser("~/my_cache/dg"))
        self.assertNotIn("~", result)

    def test_different_subdirs(self):
        """Different subdir names produce different paths."""
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/cache"
        self.assertEqual(get_jit_cache_subdir("triton"), "/cache/triton")
        self.assertEqual(get_jit_cache_subdir("inductor"), "/cache/inductor")
        self.assertEqual(get_jit_cache_subdir("torch_compile"), "/cache/torch_compile")
        self.assertEqual(get_jit_cache_subdir("deep_gemm"), "/cache/deep_gemm")


class TestSGLangJitCacheRootEnvVar(unittest.TestCase):
    """Test that the SGLANG_JIT_CACHE_ROOT EnvStr descriptor works correctly."""

    def setUp(self):
        self._backup = os.environ.pop("SGLANG_JIT_CACHE_ROOT", None)

    def tearDown(self):
        if self._backup is None:
            os.environ.pop("SGLANG_JIT_CACHE_ROOT", None)
        else:
            os.environ["SGLANG_JIT_CACHE_ROOT"] = self._backup

    def test_default_value(self):
        os.environ.pop("SGLANG_JIT_CACHE_ROOT", None)
        result = envs.SGLANG_JIT_CACHE_ROOT.get()
        self.assertTrue(
            result.endswith("/sglang"), f"Expected path ending in /sglang, got {result}"
        )
        self.assertNotIn("~", result)

    def test_explicit_value(self):
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/tmp/test_cache"
        self.assertEqual(envs.SGLANG_JIT_CACHE_ROOT.get(), "/tmp/test_cache")

    def test_is_set(self):
        os.environ.pop("SGLANG_JIT_CACHE_ROOT", None)
        self.assertFalse(envs.SGLANG_JIT_CACHE_ROOT.is_set())
        os.environ["SGLANG_JIT_CACHE_ROOT"] = "/tmp/x"
        self.assertTrue(envs.SGLANG_JIT_CACHE_ROOT.is_set())


if __name__ == "__main__":
    unittest.main(verbosity=3)
