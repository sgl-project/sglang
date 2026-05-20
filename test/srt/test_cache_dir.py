"""Test unified cache directory resolution.

Verifies that DeepGEMM cache defaults to a subdirectory under
SGLANG_CACHE_DIR, and explicit SGLANG_DG_CACHE_DIR overrides work.

Closes: https://github.com/sgl-project/sglang/issues/19612
"""

import importlib
import os
import unittest
from unittest.mock import patch


def _get_dg_cache_default(**env_overrides):
    """Re-import environ module with given env vars to get fresh default."""
    clean_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("SGLANG_CACHE_DIR", "SGLANG_DG_CACHE_DIR")
    }
    clean_env.update(env_overrides)

    with patch.dict(os.environ, clean_env, clear=True):
        import sglang.srt.environ as env_mod

        importlib.reload(env_mod)
        return env_mod.envs.SGLANG_DG_CACHE_DIR.get()


class TestUnifiedCacheDir(unittest.TestCase):
    """Test that JIT cache directories are unified under SGLANG_CACHE_DIR."""

    def test_dg_cache_default_under_sglang_cache(self):
        """Default DG cache should be under ~/.cache/sglang/deep_gemm_cache."""
        result = _get_dg_cache_default()
        expected_suffix = os.path.join(".cache", "sglang", "deep_gemm_cache")
        self.assertTrue(
            result.endswith(expected_suffix),
            f"Expected path ending with '{expected_suffix}', got '{result}'",
        )

    def test_dg_cache_follows_custom_sglang_cache_dir(self):
        """When SGLANG_CACHE_DIR is set, DG cache should be under it."""
        result = _get_dg_cache_default(SGLANG_CACHE_DIR="/tmp/my_sglang_cache")
        expected = os.path.join("/tmp/my_sglang_cache", "deep_gemm_cache")
        self.assertEqual(result, expected)

    def test_dg_cache_explicit_override(self):
        """Explicit SGLANG_DG_CACHE_DIR should override the default."""
        result = _get_dg_cache_default(SGLANG_DG_CACHE_DIR="/custom/dg_path")
        self.assertEqual(result, "/custom/dg_path")

    def test_dg_cache_explicit_override_wins_over_sglang_cache(self):
        """Explicit SGLANG_DG_CACHE_DIR takes precedence over SGLANG_CACHE_DIR."""
        result = _get_dg_cache_default(
            SGLANG_CACHE_DIR="/tmp/base",
            SGLANG_DG_CACHE_DIR="/tmp/explicit_dg",
        )
        self.assertEqual(result, "/tmp/explicit_dg")

    def test_dg_cache_with_empty_sglang_cache_dir(self):
        """When SGLANG_CACHE_DIR is empty, DG cache should use default."""
        result = _get_dg_cache_default(SGLANG_CACHE_DIR="")
        expected_suffix = os.path.join(".cache", "sglang", "deep_gemm_cache")
        self.assertTrue(
            result.endswith(expected_suffix),
            f"Expected path ending with '{expected_suffix}', got '{result}'",
        )

    def test_dg_cache_not_under_old_default(self):
        """Default DG cache should NOT be ~/.cache/deep_gemm anymore."""
        result = _get_dg_cache_default()
        old_default = os.path.expanduser("~/.cache/deep_gemm")
        self.assertNotEqual(
            result,
            old_default,
            "DG cache should no longer default to ~/.cache/deep_gemm",
        )


if __name__ == "__main__":
    unittest.main()
