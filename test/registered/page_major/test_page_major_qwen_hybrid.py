"""Unified memory pool on a GDN-hybrid model, across the backend matrix.

Qwen3.5-4B is a gated-delta-net / linear-attention hybrid, which exercises the
path most prone to subtle bugs: the Mamba conv/SSM state stays a strided
envelope view (its kernels are stride-aware by design) while the
full-attention KV is per-layer views, which the fa3 / flashinfer cells read
through the translator's read tables. The resolved-default cell pins the
no-pin path, since a pinned backend hides default-resolution breakage by
construction.

Registered to the label-gated ``run-ci-extra`` suite (opt-in, not per-commit).
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=1600, stage="extra-a", runner_config="1-gpu-large")

_UNIFIED_COMMON_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.85",
    "--enable-unified-memory",
    "--linear-attn-backend",
    "triton",
    "--mamba-backend",
    "triton",
]


class TestUnifiedQwenHybridTriton(DefaultServerBase):
    """Unified pool on Qwen3.5-4B (GDN-hybrid), Triton pinned: contiguous
    full-attention views + strided conv/SSM state through the reference
    backends."""

    model = DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST

    # Measured ~0.86 in this harness on both the static pools and the envelope
    # layout; 0.80 leaves noise margin and still catches a corrupted prefill
    # state, which reads ~0.61.
    gsm8k_threshold = 0.80
    num_gsm8k_questions = 200
    num_shots = 5
    parallel = 32

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "triton"]

    def test_gsm8k(self):
        from sglang.test.few_shot_gsm8k import run_eval as run_few_shot_gsm8k

        url = urlparse(self.base_url)
        args = SimpleNamespace(
            num_shots=self.num_shots,
            data_path=None,
            num_questions=self.num_gsm8k_questions,
            max_new_tokens=512,
            parallel=self.parallel,
            host=f"http://{url.hostname}",
            port=int(url.port),
        )
        metrics = run_few_shot_gsm8k(args)
        print(
            f"[{self.__class__.__name__}] GSM8K accuracy: {metrics['accuracy']:.3f} "
            f"(threshold: {self.gsm8k_threshold})"
        )
        self.assertGreaterEqual(metrics["accuracy"], self.gsm8k_threshold)


class TestUnifiedQwenHybridFa3(TestUnifiedQwenHybridTriton):
    """fa3 pinned: read tables, eager direct-bind + captured fused copy."""

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "fa3"]


class TestUnifiedQwenHybridFlashinfer(TestUnifiedQwenHybridTriton):
    """flashinfer pinned: token ids reconstructed from the read table by the
    ENTRY_PAGE_SIZE CSR builder."""

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "flashinfer"]


class TestUnifiedQwenHybridResolvedDefault(TestUnifiedQwenHybridTriton):
    """No backend pin: whatever the host resolves must be in the allow-list,
    or the server fails to boot under its own defaults."""

    other_args = _UNIFIED_COMMON_ARGS


if __name__ == "__main__":
    unittest.main()
