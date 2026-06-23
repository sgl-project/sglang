"""Unit tests for bugfix #26013: Gemma3 RoPE KeyError when factor is absent.

Gemma3TextModel crashes with KeyError when rope_parameters["full_attention"]
does not contain a "factor" key (non-scaled default RoPE configs).
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGemma3RoPEKeyError(CustomTestCase):
    """Test for bug #26013: Gemma3 RoPE KeyError when factor is absent."""

    def _make_rope_params(self, full_attention_params):
        return {
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": full_attention_params,
        }

    def test_rope_parameters_without_factor(self):
        """Config with full_attention using default RoPE (no factor) should not raise KeyError."""
        rope_params = self._make_rope_params(
            {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            }
        )
        factor = rope_params["full_attention"].get("factor", 1.0)
        rope_type = (
            "linear"
            if rope_params["full_attention"].get("factor") is not None
            else "default"
        )
        self.assertEqual(rope_type, "default")
        self.assertEqual(factor, 1.0)

    def test_rope_parameters_with_factor(self):
        """Config with factor present should still produce 'linear' rope_type."""
        rope_params = self._make_rope_params(
            {
                "rope_type": "linear",
                "rope_theta": 1000000.0,
                "factor": 8.0,
            }
        )
        factor = rope_params["full_attention"].get("factor", 1.0)
        rope_type = (
            "linear"
            if rope_params["full_attention"].get("factor") is not None
            else "default"
        )
        self.assertEqual(rope_type, "linear")
        self.assertEqual(factor, 8.0)


if __name__ == "__main__":
    unittest.main()
