"""Unit tests for the opt-in DSR o_proj MXFP4 ASM a4w4 routing.

CPU-only, no model loading. Validates the gate added by
`SGLANG_DSR_OPROJ_MXFP4_ASM`:

* `QuarkW4A4MXFP4.enable_asm()` flips `use_asm` on only when the aiter ASM
  a4w4 kernel is available, and otherwise falls back (warns, stays off).
* `QuarkConfig.get_linear_scheme()` only calls `enable_asm()` for `*.o_proj`
  layers when the env var is set to "1".
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.quark import quark as quark_mod
from sglang.srt.layers.quantization.quark.quark import QuarkConfig
from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4 as mxfp4_mod
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import QuarkW4A4MXFP4
from sglang.test.test_utils import CustomTestCase

_ENV = "SGLANG_DSR_OPROJ_MXFP4_ASM"


def _bare_scheme() -> QuarkW4A4MXFP4:
    """Build a scheme without running __init__ (needs no config for the gate)."""
    scheme = QuarkW4A4MXFP4.__new__(QuarkW4A4MXFP4)
    scheme.use_asm = False
    return scheme


def _bare_config() -> QuarkConfig:
    return QuarkConfig.__new__(QuarkConfig)


class TestEnableAsm(CustomTestCase):
    """`QuarkW4A4MXFP4.enable_asm()` availability handling."""

    def test_enabled_when_asm_available(self):
        scheme = _bare_scheme()
        with patch.object(mxfp4_mod, "_is_hip", True), patch.object(
            mxfp4_mod, "_HAS_ASM_A4W4", True
        ):
            scheme.enable_asm()
        self.assertTrue(scheme.use_asm)

    def test_falls_back_when_asm_missing(self):
        scheme = _bare_scheme()
        with patch.object(mxfp4_mod, "_is_hip", True), patch.object(
            mxfp4_mod, "_HAS_ASM_A4W4", False
        ):
            scheme.enable_asm()
        self.assertFalse(scheme.use_asm)

    def test_falls_back_when_not_hip(self):
        scheme = _bare_scheme()
        with patch.object(mxfp4_mod, "_is_hip", False), patch.object(
            mxfp4_mod, "_HAS_ASM_A4W4", True
        ):
            scheme.enable_asm()
        self.assertFalse(scheme.use_asm)


class TestGetLinearSchemeRouting(CustomTestCase):
    """The `get_linear_scheme` opt-in gate only fires for o_proj + env=1."""

    def _run(self, layer_name: str, env_value):
        """Return the scheme produced by get_linear_scheme for layer_name.

        The scheme-selection internals are mocked so the test isolates the
        env/o_proj gate; enable_asm() is exercised for real with ASM forced
        available.
        """
        config = _bare_config()
        scheme = _bare_scheme()

        env = {} if env_value is None else {_ENV: env_value}
        with patch.object(
            QuarkConfig, "_find_matched_config", return_value={}
        ), patch.object(
            QuarkConfig, "_get_scheme_from_config", return_value=scheme
        ), patch.object(
            QuarkConfig, "_check_scheme_supported", return_value=True
        ), patch.object(
            scheme, "get_min_capability", return_value=70
        ), patch.object(
            mxfp4_mod, "_is_hip", True
        ), patch.object(
            mxfp4_mod, "_HAS_ASM_A4W4", True
        ), patch.dict(
            quark_mod.os.environ, env, clear=False
        ):
            # Ensure a stale env var from the outer environment can't leak in.
            quark_mod.os.environ.pop(_ENV, None)
            if env_value is not None:
                quark_mod.os.environ[_ENV] = env_value
            return config.get_linear_scheme(layer=None, layer_name=layer_name)

    def test_oproj_with_env_enables_asm(self):
        scheme = self._run("model.layers.3.self_attn.o_proj", "1")
        self.assertTrue(scheme.use_asm)

    def test_oproj_without_env_keeps_triton(self):
        scheme = self._run("model.layers.3.self_attn.o_proj", None)
        self.assertFalse(scheme.use_asm)

    def test_oproj_env_zero_keeps_triton(self):
        scheme = self._run("model.layers.3.self_attn.o_proj", "0")
        self.assertFalse(scheme.use_asm)

    def test_non_oproj_with_env_keeps_triton(self):
        scheme = self._run("model.layers.3.self_attn.kv_b_proj", "1")
        self.assertFalse(scheme.use_asm)


if __name__ == "__main__":
    unittest.main()
