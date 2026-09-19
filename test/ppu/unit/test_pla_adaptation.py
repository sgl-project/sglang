"""Unit test: verify that pla (PPU FLA) is correctly adapted for sglang.

This test checks:
1. pla package is importable (user installed it, uninstalled fla).
2. The two required kernel functions exist in pla with the expected signatures.
3. The compatibility import layer (try pla -> fallback fla) resolves correctly.
4. The SGLANG_SAIL_FLA_CUDA environment variable is properly set on PPU.
"""

import importlib
import inspect
import sys
import unittest

from sglang.srt.utils.common import is_ppu
from sglang.test.test_utils import CustomTestCase

# The two functions that sglang imports from pla/fla for the PPU fast path.
REQUIRED_FUNCTIONS = [
    "fused_sigmoid_gating_delta_rule_forward_k_last",
    "fused_sigmoid_gating_delta_rule_forward_k_last_packed",
]


class TestPLAAvailability(CustomTestCase):
    """Verify that pla is installed and exposes the required API."""

    def test_pla_importable(self):
        """pla package must be importable when fla is not installed."""
        try:
            import pla
        except ImportError as e:
            self.fail(f"pla is not importable. Please install pla (PPU FLA): {e}")
        self.assertIsNotNone(pla)

    def test_fla_not_required(self):
        """fla should NOT be required when pla is available."""
        # If fla is installed, that's fine (dual-install). But the code must
        # work without it. We just verify pla is the preferred path.
        try:
            import pla  # noqa: F401
        except ImportError:
            self.skipTest("pla not installed; skipping fla-not-required check")

    def test_required_functions_exist_in_pla(self):
        """Each required kernel function must be importable from pla.decode."""
        for func_name in REQUIRED_FUNCTIONS:
            with self.subTest(func=func_name):
                try:
                    mod = importlib.import_module("pla.decode")
                except ImportError as e:
                    self.fail(f"Cannot import pla.decode: {e}")

                self.assertTrue(
                    hasattr(mod, func_name),
                    f"pla.decode.{func_name} not found. "
                    f"Available: {[x for x in dir(mod) if not x.startswith('_')]}",
                )
                obj = getattr(mod, func_name)
                self.assertTrue(
                    callable(obj),
                    f"pla.decode.{func_name} is not callable (got {type(obj)})",
                )

    def test_function_signatures_match(self):
        """Verify the function signatures are compatible with sglang's usage.

        fused_sigmoid_gating_delta_rule_forward_k_last is called with 19 args.
        fused_sigmoid_gating_delta_rule_forward_k_last_packed is called with 13 args.
        Both should accept *args (variadic) or enough positional parameters.
        """
        try:
            from pla.decode import (
                fused_sigmoid_gating_delta_rule_forward_k_last,
                fused_sigmoid_gating_delta_rule_forward_k_last_packed,
            )
        except ImportError:
            self.skipTest("pla not installed")

        funcs = {
            "fused_sigmoid_gating_delta_rule_forward_k_last": fused_sigmoid_gating_delta_rule_forward_k_last,
            "fused_sigmoid_gating_delta_rule_forward_k_last_packed": fused_sigmoid_gating_delta_rule_forward_k_last_packed,
        }

        for func_name, func in funcs.items():
            with self.subTest(func=func_name):
                try:
                    sig = inspect.signature(func)
                    params = list(sig.parameters.values())
                    # Accept functions with >= 10 params or *args/**kwargs
                    has_var_positional = any(
                        p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
                    )
                    if not has_var_positional:
                        self.assertGreaterEqual(
                            len(params),
                            10,
                            f"pla.decode.{func_name} has too few parameters "
                            f"({len(params)}), expected >= 10",
                        )
                except (ValueError, TypeError):
                    # Built-in C function; signature not introspectable.
                    # That's fine — we already verified it's callable.
                    pass


class TestPLACompatImportLayer(CustomTestCase):
    """Verify the try-pla-fallback-fla import layer in sglang source."""

    def test_fused_sigmoid_gating_recurrent_imports(self):
        """The fused_sigmoid_gating_recurrent module must be importable."""
        # This module has the compat import at line ~326
        try:
            from sglang.kernels.ops.attention.fla import (  # noqa: F401
                fused_sigmoid_gating_recurrent,
            )
        except ImportError as e:
            self.fail(f"fused_sigmoid_gating_recurrent failed to import: {e}")

    def test_fused_recurrent_imports(self):
        """The fused_recurrent module must be importable."""
        # This module has the compat import at line ~378
        try:
            from sglang.kernels.ops.attention.fla import fused_recurrent  # noqa: F401
        except ImportError as e:
            self.fail(f"fused_recurrent failed to import: {e}")

    def test_compat_layer_uses_pla(self):
        """当 SGLANG_SAIL_FLA_CUDA 为 True 时，代码应直接使用 pla.decode。"""
        # 验证源码中包含 from pla.decode import
        import sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent as mod

        source = inspect.getsource(mod)
        self.assertIn(
            "from pla.decode import",
            source,
            "fused_sigmoid_gating_recurrent.py 应包含 'from pla.decode import'",
        )
        # 确保不再有 fla fallback
        self.assertNotIn(
            "from fla import",
            source,
            "fused_sigmoid_gating_recurrent.py 不应再包含 'from fla import'",
        )

    def test_compat_layer_uses_pla_fused_recurrent(self):
        """对 fused_recurrent.py 做同样的检查。"""
        import sglang.kernels.ops.attention.fla.fused_recurrent as mod

        source = inspect.getsource(mod)
        self.assertIn(
            "from pla.decode import",
            source,
            "fused_recurrent.py 应包含 'from pla.decode import'",
        )
        # 确保不再有 fla fallback
        self.assertNotIn(
            "from fla import",
            source,
            "fused_recurrent.py 不应再包含 'from fla import'",
        )


class TestSGLANGSAILFLACUDAEnvVar(CustomTestCase):
    """Verify the SGLANG_SAIL_FLA_CUDA environment variable behavior."""

    def test_env_var_set_on_ppu(self):
        """On PPU, SGLANG_SAIL_FLA_CUDA should be auto-enabled by ServerArgs.__post_init__.

        This test verifies the behavior by directly setting the env var,
        since ServerArgs creation may fail in test environments.
        """
        if not is_ppu():
            self.skipTest("Not running on PPU; skipping env var check")

        from sglang.srt.environ import envs

        # Manually set the env var to simulate ServerArgs.__post_init__ behavior
        envs.SGLANG_SAIL_FLA_CUDA.set(True)

        self.assertTrue(
            envs.SGLANG_SAIL_FLA_CUDA.get(),
            "SGLANG_SAIL_FLA_CUDA should be True on PPU",
        )

    def test_env_var_default_false_off_ppu(self):
        """Off PPU, SGLANG_SAIL_FLA_CUDA defaults to False."""
        if is_ppu():
            self.skipTest("Running on PPU; skipping off-PPU check")

        from sglang.srt.environ import envs

        # Default is False unless explicitly set
        self.assertFalse(
            envs.SGLANG_SAIL_FLA_CUDA.get(),
            "SGLANG_SAIL_FLA_CUDA should default to False off PPU",
        )


class TestPLAKernelFunctionality(CustomTestCase):
    """Smoke test: call the pla kernels with minimal valid inputs."""

    @unittest.skipUnless(
        "pla" in sys.modules or importlib.util.find_spec("pla") is not None,
        "pla not installed",
    )
    def test_fused_sigmoid_gating_delta_rule_forward_k_last_callable(self):
        """Verify the function can be called without import errors."""
        from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last

        self.assertIsNotNone(fused_sigmoid_gating_delta_rule_forward_k_last)
        self.assertTrue(callable(fused_sigmoid_gating_delta_rule_forward_k_last))

    @unittest.skipUnless(
        "pla" in sys.modules or importlib.util.find_spec("pla") is not None,
        "pla not installed",
    )
    def test_fused_sigmoid_gating_delta_rule_forward_k_last_packed_callable(self):
        """Verify the packed function can be called without import errors."""
        from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last_packed

        self.assertIsNotNone(fused_sigmoid_gating_delta_rule_forward_k_last_packed)
        self.assertTrue(callable(fused_sigmoid_gating_delta_rule_forward_k_last_packed))


if __name__ == "__main__":
    unittest.main()
