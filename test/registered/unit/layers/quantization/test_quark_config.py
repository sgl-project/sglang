"""Unit tests for QuarkConfig — CPU-only, no model loading."""

import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    QuarkW4A4MXFP4,
)
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe import (
    QuarkW4A4MXFp4MoE,
)
from sglang.srt.layers.quantization.quark_int4fp8_moe import (
    QuarkInt4Fp8MoEMethod,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_GET_CAP = "sglang.srt.layers.quantization.quark.quark.get_device_capability"


def _bare_config() -> QuarkConfig:
    """Skip __init__ — _check_scheme_supported reads no instance attributes."""
    return QuarkConfig.__new__(QuarkConfig)


class TestCheckSchemeSupportedError(CustomTestCase):
    """Regression for `RuntimeError("a", "b", "c")` being passed three args.

    Bug: `_check_scheme_supported` raised `RuntimeError` with three positional
    string fragments. `RuntimeError.__str__` formats `self.args` as a tuple
    when `len(args) != 1`, so the user saw
        ('Quantization scheme is not supported for ', 'the current GPU…', 'Current capability: 70.')
    instead of a sentence. Fix: pass one already-joined message.
    """

    def test_error_is_single_argument(self):
        # The structural assertion that catches the bug regardless of wording.
        with patch(_GET_CAP, return_value=(7, 0)):  # capability = 70 < 200
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        err = ctx.exception
        self.assertEqual(
            len(err.args),
            1,
            f"RuntimeError must carry a single joined message, got {err.args!r}",
        )

    def test_error_message_renders_as_sentence(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        msg = str(ctx.exception)
        # Tuple-repr leakage shows up as a leading '(' and quote-comma joins.
        self.assertFalse(
            msg.startswith("("),
            f"error message starts with '(' (tuple repr leaked): {msg!r}",
        )
        self.assertNotIn(
            "', '",
            msg,
            f"error message contains tuple-style fragment join: {msg!r}",
        )

    def test_error_message_content(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            with self.assertRaises(RuntimeError) as ctx:
                _bare_config()._check_scheme_supported(min_capability=200)
        msg = str(ctx.exception)
        self.assertIn("Quantization scheme is not supported", msg)
        self.assertIn("Min capability: 200", msg)
        self.assertIn("Current capability: 70", msg)

    # ---- Guardrails: unchanged code paths ---------------------------------

    def test_unsupported_returns_false_when_error_disabled(self):
        with patch(_GET_CAP, return_value=(7, 0)):
            ok = _bare_config()._check_scheme_supported(min_capability=200, error=False)
        self.assertFalse(ok)

    def test_supported_returns_true(self):
        with patch(_GET_CAP, return_value=(8, 0)):  # capability = 80 >= 70
            ok = _bare_config()._check_scheme_supported(min_capability=70)
        self.assertTrue(ok)

    def test_no_device_returns_false(self):
        with patch(_GET_CAP, return_value=None):
            ok = _bare_config()._check_scheme_supported(min_capability=70)
        self.assertFalse(ok)


class TestOptionalAiterErrors(CustomTestCase):
    def test_w4a4_mxfp4_requires_aiter_only_when_selected(self):
        module = "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4"
        with (
            patch(f"{module}._is_hip", True),
            patch(f"{module}._use_aiter", False),
            self.assertRaisesRegex(RuntimeError, "requires AITER"),
        ):
            QuarkW4A4MXFP4({}, {})

    def test_w4a4_mxfp4_moe_requires_aiter_only_when_selected(self):
        module = (
            "sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe"
        )
        with (
            patch(f"{module}._is_hip", True),
            patch(f"{module}._use_aiter", False),
            self.assertRaisesRegex(RuntimeError, "requires AITER"),
        ):
            QuarkW4A4MXFp4MoE({}, {})

    def test_int4fp8_moe_requires_aiter_only_when_selected(self):
        module = "sglang.srt.layers.quantization.quark_int4fp8_moe"
        with (
            patch(f"{module}._is_hip", True),
            patch(f"{module}._use_aiter", False),
            self.assertRaisesRegex(RuntimeError, "requires AITER"),
        ):
            QuarkInt4Fp8MoEMethod(object())


if __name__ == "__main__":
    unittest.main()
