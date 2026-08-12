"""
End-to-end accuracy tests for the page-major KV layout on a hybrid-SWA MoE model.

Launches gpt-oss-20b with ``--enable-page-major-kv-layout`` (Triton, strided
views) and with ``--enable-unified-memory`` across the dense-view backend
matrix, checking GSM8K accuracy on each. gpt-oss is a uniform-row hybrid-SWA
model, so under the unified pool its MHA + SWA sub-pools flip to DENSE
per-layer views and the fa3 cell reads them through the choke point's
canonical page tables; the resolved-default cell pins the no-pin path (the
pattern that caught both #32972 review defects — a pinned backend hides
default-resolution breakage by construction). flashinfer is absent on
purpose: gpt-oss uses attention sinks, which flashinfer does not support.

Registered to the label-gated ``run-ci-extra`` suite (opt-in, not per-commit).

Usage:
    python3 -m unittest test_page_major_gpt_oss
"""

import unittest
from types import SimpleNamespace
from urllib.parse import urlparse

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE

register_cuda_ci(est_time=1500, stage="extra-a", runner_config="1-gpu-large")


class TestPageMajorGptOss(DefaultServerBase):
    """Page-major KV layout on gpt-oss-20b (hybrid-SWA MoE), Triton backend."""

    model = DEFAULT_MODEL_NAME_FOR_TEST_MXFP4_WITH_MOE

    gsm8k_threshold = 0.45
    num_gsm8k_questions = 200
    num_shots = 5
    parallel = 32

    other_args = [
        "--enable-page-major-kv-layout",
        # The envelope's strided 4-D K/V views are only read by the Triton
        # attention kernels (the layout's validator enforces this).
        "--attention-backend",
        "triton",
        "--mem-fraction-static",
        "0.70",
        "--cuda-graph-backend-prefill=disabled",
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


_UNIFIED_COMMON_ARGS = [
    "--enable-unified-memory",
    "--mem-fraction-static",
    "0.70",
    "--cuda-graph-backend-prefill=disabled",
]


class TestUnifiedGptOssTriton(TestPageMajorGptOss):
    """Unified pool, Triton pinned: dense MHA/SWA views through the strided-
    capable reference backend."""

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "triton"]


class TestUnifiedGptOssFa3(TestPageMajorGptOss):
    """Unified pool, fa3 pinned: dense views read via the choke point's
    canonical page tables (eager direct-bind + captured fused copy)."""

    other_args = _UNIFIED_COMMON_ARGS + ["--attention-backend", "fa3"]


class TestUnifiedGptOssResolvedDefault(TestPageMajorGptOss):
    """Unified pool, NO backend pin: whatever the host resolves must be in the
    backend allow-list or the server fails to boot under its own defaults."""

    other_args = _UNIFIED_COMMON_ARGS


if __name__ == "__main__":
    unittest.main()
