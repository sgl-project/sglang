"""Unified memory pool on a GDN-hybrid model, across the backend matrix.

Qwen3.5-4B is a gated-delta-net / linear-attention hybrid, which exercises the
path most prone to subtle bugs: the Mamba conv/SSM state stays a strided
envelope view (its kernels are stride-aware by design) while the
full-attention KV is per-layer views, which the fa3 / flashinfer cells read
through the translator's read tables.
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=261, stage="extra-a", runner_config="1-gpu-large")

_COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.85",
    "--linear-attn-backend",
    "triton",
    "--mamba-backend",
    "triton",
]
_UNIFIED_COMMON_ARGS = _COMMON_ARGS + ["--enable-unified-memory"]


class TestUnifiedQwenHybridTriton(DefaultServerBase):
    """Unified pool on Qwen3.5-4B (GDN-hybrid), Triton pinned: contiguous
    full-attention views + strided conv/SSM state through the reference
    backends."""

    model = DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST

    # Keep the accuracy gate while validating the sgl-eval thinking protocol
    # against the static-pool control below.
    gsm8k_threshold = 0.80
    num_gsm8k_questions = 200
    parallel = 32

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "triton"]

    def test_gsm8k(self):
        self.assertGreaterEqual(self._run_gsm8k(thinking=True), self.gsm8k_threshold)

    def _run_gsm8k(self, *, thinking):
        from sglang.test.run_eval import run_eval as run_gsm8k_eval

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            eval_name="gsm8k",
            num_examples=self.num_gsm8k_questions,
            max_tokens=16384 if thinking else 2048,
            sgl_eval_thinking=thinking,
            num_threads=self.parallel,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_gsm8k_eval(args)
        print(
            f"[{self.__class__.__name__}] thinking={thinking}, "
            f"GSM8K accuracy: {metrics['accuracy']:.3f} "
            f"(threshold: {self.gsm8k_threshold})"
        )
        return metrics["accuracy"]


class TestUnifiedQwenHybridFa3(TestUnifiedQwenHybridTriton):
    """fa3 pinned: read tables, eager direct-bind + captured fused copy."""

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "fa3"]


class TestUnifiedQwenHybridFlashinfer(TestUnifiedQwenHybridTriton):
    """flashinfer pinned: token ids reconstructed from the read table by the
    ENTRY_PAGE_SIZE CSR builder."""

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "flashinfer"]


class TestStaticQwenHybridFa3(TestUnifiedQwenHybridTriton):
    """Control for distinguishing evaluation settings from unified-pool errors."""

    other_args = _COMMON_ARGS + ["--attention-backend", "fa3"]

    def test_gsm8k(self):
        # Measure the previous protocol on static pools before the thinking run.
        self._run_gsm8k(thinking=False)
        super().test_gsm8k()


if __name__ == "__main__":
    unittest.main()
