"""
End-to-end accuracy test for the page-major KV layout on a GDN-hybrid model.

Launches Qwen3.5-4B (a gated-delta-net / linear-attention hybrid) with
``--enable-page-major-kv-layout`` on the Triton attention + linear-attn + Mamba
backends and checks that GSM8K accuracy holds. This exercises the page-major
path most prone to subtle bugs: the Mamba conv/SSM state stored as a strided
envelope view, plus the full-attention KV pool, both read/written by the GDN
prefill and decode kernels.

Registered to the label-gated ``run-ci-extra`` suite (opt-in, not per-commit).

Usage:
    python3 -m unittest test_page_major_qwen_hybrid
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=85, stage="extra-a", runner_config="1-gpu-large")


class TestPageMajorQwenHybrid(DefaultServerBase):
    """Page-major KV layout on Qwen3.5-4B (GDN-hybrid), Triton backends."""

    model = DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST

    # Measured in this harness: baseline (no page-major) and page-major both
    # ~0.86; the 0.80 threshold leaves margin for run-to-run noise while still
    # catching the prefill-state corruption that page-major hit before the
    # gather/scatter fix in gdn_backend.forward_extend (which dropped it to ~0.61).
    gsm8k_threshold = 0.80
    num_gsm8k_questions = 200
    num_shots = 5
    parallel = 32

    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.85",
        "--enable-page-major-kv-layout",
        # Only the Triton attention / linear-attn / Mamba kernels read the
        # strided envelope K/V and conv/SSM state (enforced by the validator).
        "--attention-backend",
        "triton",
        "--linear-attn-backend",
        "triton",
        "--mamba-backend",
        "triton",
    ]

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


if __name__ == "__main__":
    unittest.main()
