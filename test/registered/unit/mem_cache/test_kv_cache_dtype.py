"""KV dtype resolution must honor explicit formats and draft overrides."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache import kv_cache_dtype
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKvCacheDtype(CustomTestCase):
    def _resolve(self, dtype, *, is_draft=False, draft_dtype=None):
        return kv_cache_dtype.configure_kv_cache_dtype(
            server_args_kv_cache_dtype=dtype,
            model=None,
            model_dtype=torch.bfloat16,
            is_draft_worker=is_draft,
            is_dflash=False,
            speculative_draft_attention_backend=None,
            speculative_draft_kv_cache_dtype=draft_dtype,
        )

    @patch.object(kv_cache_dtype, "_is_hip", True)
    @patch.object(kv_cache_dtype.current_platform, "is_cpu", return_value=False)
    def test_hip_rejects_explicit_e5m2(self, _is_cpu):
        """The effective target or draft dtype must not silently become E4M3."""
        for target_dtype, is_draft, draft_dtype in (
            ("fp8_e5m2", False, None),
            ("fp8_e5m2", True, None),
            ("auto", True, "fp8_e5m2"),
        ):
            with self.subTest(target=target_dtype, draft=draft_dtype):
                with self.assertRaisesRegex(ValueError, "E5M2.*HIP"):
                    self._resolve(
                        target_dtype, is_draft=is_draft, draft_dtype=draft_dtype
                    )

    @patch.object(kv_cache_dtype, "_is_hip", True)
    def test_supported_hip_dtypes_and_draft_override(self):
        for dtype, expected in (
            ("auto", torch.bfloat16),
            ("bfloat16", torch.bfloat16),
            ("fp8_e4m3", kv_cache_dtype.fp8_dtype),
        ):
            with self.subTest(dtype=dtype):
                self.assertEqual(self._resolve(dtype)[1], expected)
        self.assertEqual(
            self._resolve("fp8_e5m2", is_draft=True, draft_dtype="bfloat16"),
            ("bfloat16", torch.bfloat16),
        )

    @patch.object(kv_cache_dtype, "_is_hip", False)
    @patch.object(kv_cache_dtype.current_platform, "is_cpu", return_value=False)
    def test_non_hip_e5m2_is_preserved(self, _is_cpu):
        self.assertEqual(self._resolve("fp8_e5m2")[1], torch.float8_e5m2)


if __name__ == "__main__":
    unittest.main()
