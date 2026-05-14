"""Unit tests for MLX backend on-the-fly quantization.

Covers:
  - mlx_q4 / mlx_q8 quantize fp16 weights to QuantizedLinear in-place
  - active-memory drops after quantization
  - smoke /generate still works post-quantize
  - pre-quantized HF repos still load (regression guard for mlx_lm passthrough)
  - mlx_q4 flag on an already-quantized model is a no-op (skip + log)

Skips on non-Apple-Silicon platforms and when ``mlx`` / ``mlx_lm`` are missing.
"""

from __future__ import annotations

import gc
import importlib.util
import platform
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

# Registered with the CPU suite (runtime no-op marker, parsed via AST).
# On non-Apple-Silicon CI runners the entire TestCase class skips via the
# @skipUnless guard below, so this registration is the harmless "yes this
# test exists" signal the registry requires.
register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

_IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"
_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)

_SKIP_REASON = "Apple-Silicon-only test (requires Darwin/arm64 + mlx + mlx_lm)"

# Tiny model used across tests; ~0.6B fp16 = ~1.1 GB on disk after first download.
_TEST_MODEL = "Qwen/Qwen3-0.6B"
_TEST_MODEL_PREQUANT = "mlx-community/Qwen3-0.6B-4bit"


@unittest.skipUnless(_IS_APPLE_SILICON and _HAS_MLX, _SKIP_REASON)
class TestMlxQuantization(unittest.TestCase):
    """Smoke tests for --quantization mlx_q4 / mlx_q8 in MlxModelRunner."""

    # ---------- helpers ----------

    @staticmethod
    def _module_counts(model) -> tuple[int, int]:
        n_quant, n_linear = 0, 0
        for _, m in model.named_modules():
            cls = type(m).__name__
            if cls == "QuantizedLinear":
                n_quant += 1
            elif cls == "Linear":
                n_linear += 1
        return n_quant, n_linear

    @staticmethod
    def _reset_mlx_memory() -> None:
        import mlx.core as mx

        gc.collect()
        mx.clear_cache()

    def _build_runner(self, model_path: str, quantization: str | None):
        from sglang.srt.hardware_backend.mlx.model_runner import MlxModelRunner

        return MlxModelRunner(
            model_path=model_path,
            quantization=quantization,
            pool_size=1024,  # small pool — these tests don't drive generation depth
        )

    # ---------- tests ----------

    def test_mlx_q4_creates_quantized_linear_modules(self):
        """All Linear modules should be QuantizedLinear after mlx_q4 load."""
        self._reset_mlx_memory()
        runner = self._build_runner(_TEST_MODEL, "mlx_q4")
        try:
            n_quant, n_linear = self._module_counts(runner.model)
            self.assertGreater(
                n_quant, 0, "expected at least one QuantizedLinear module"
            )
            self.assertEqual(
                n_linear,
                0,
                f"all Linear modules should have been quantized, got {n_linear} remaining",
            )
        finally:
            del runner
            self._reset_mlx_memory()

    def test_mlx_q4_reduces_memory_vs_fp16(self):
        """mlx_q4 should use meaningfully less memory than the fp16 baseline."""
        import mlx.core as mx

        self._reset_mlx_memory()
        runner_fp = self._build_runner(_TEST_MODEL, None)
        mx.eval(runner_fp.model.parameters())
        mem_fp = mx.get_active_memory()
        del runner_fp
        self._reset_mlx_memory()

        runner_q4 = self._build_runner(_TEST_MODEL, "mlx_q4")
        mx.eval(runner_q4.model.parameters())
        mem_q4 = mx.get_active_memory()
        del runner_q4
        self._reset_mlx_memory()

        # Conservative: expect at least 40% reduction. On Qwen3-0.6B we measured ~72%;
        # 40% leaves headroom for different mlx_lm versions, model shapes, etc.
        reduction = 1 - (mem_q4 / max(mem_fp, 1))
        self.assertGreater(
            reduction,
            0.40,
            f"expected >40% memory reduction with mlx_q4, got {reduction*100:.1f}% "
            f"(fp16={mem_fp/1024**3:.2f} GB, q4={mem_q4/1024**3:.2f} GB)",
        )

    def test_mlx_q8_creates_quantized_linear_modules(self):
        """Same check for the 8-bit variant."""
        self._reset_mlx_memory()
        runner = self._build_runner(_TEST_MODEL, "mlx_q8")
        try:
            n_quant, n_linear = self._module_counts(runner.model)
            self.assertGreater(n_quant, 0)
            self.assertEqual(n_linear, 0)
        finally:
            del runner
            self._reset_mlx_memory()

    def test_mlx_q4_generates_text(self):
        """After on-the-fly quantization the model must still generate non-empty text."""
        from mlx_lm import generate
        from transformers import AutoTokenizer

        self._reset_mlx_memory()
        runner = self._build_runner(_TEST_MODEL, "mlx_q4")
        try:
            tok = AutoTokenizer.from_pretrained(_TEST_MODEL)
            output = generate(
                runner.model,
                tok,
                prompt="The capital of France is",
                max_tokens=5,
                verbose=False,
            )
            self.assertIsInstance(output, str)
            self.assertGreater(
                len(output.strip()), 0, "generation returned empty string"
            )
        finally:
            del runner
            self._reset_mlx_memory()

    def test_pre_quantized_hf_repo_passthrough(self):
        """Loading mlx-community/<model>-4bit must still work (mlx_lm passthrough,
        regression guard for the no-quantization-flag path).
        """
        self._reset_mlx_memory()
        runner = self._build_runner(_TEST_MODEL_PREQUANT, quantization=None)
        try:
            n_quant, n_linear = self._module_counts(runner.model)
            self.assertGreater(
                n_quant,
                0,
                "pre-quantized HF repo should load as QuantizedLinear without --quantization",
            )
        finally:
            del runner
            self._reset_mlx_memory()

    def test_quantize_flag_on_already_quantized_model_is_noop(self):
        """Passing --quantization mlx_q4 on a pre-quantized repo should NOT double-quantize."""
        self._reset_mlx_memory()
        # Using mlx_q4 against an already-q4 repo. The runner logs the skip and leaves
        # the existing QuantizedLinear modules untouched.
        runner = self._build_runner(_TEST_MODEL_PREQUANT, "mlx_q4")
        try:
            n_quant, n_linear = self._module_counts(runner.model)
            self.assertGreater(n_quant, 0)
            self.assertEqual(n_linear, 0)
        finally:
            del runner
            self._reset_mlx_memory()


if __name__ == "__main__":
    unittest.main()
