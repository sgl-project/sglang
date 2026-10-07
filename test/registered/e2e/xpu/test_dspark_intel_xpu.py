"""DSPARK speculative decoding on Intel XPU."""

import os
import sys
import unittest

from sglang.srt.utils import is_xpu, kill_process_tree

# Put the `test/` root on sys.path so `registered.<...>` resolves regardless of
# cwd: CI runs each file as `python3 <full_path>` (only the file's own dir is on
# the path), and pytest inserts only the file's dir too. `test/` is three levels
# up from this file's dir (test/registered/e2e/xpu/<this>).
_TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

# Reference the shared mixin as a module attribute rather than importing the
# Test* names: pytest collects by class __name__, so importing the base module's
# CUDA TestBasicSanityDSpark here would re-collect it under the XPU file. Only the
# XPU subclasses below should run.
from registered.core import test_basic_sanity_dspark as _dspark_base

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.spec_server_kits import SpecLogprobKit
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=3000, suite="nightly-xpu-2-gpu", nightly=True)


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestBasicSanityDSparkXpuTriton(_dspark_base._DSparkSanityMixin, CustomTestCase):
    attention_backend = "triton"
    draft_attention_backend = "triton"
    max_running_requests = "2"
    mem_fraction_static = "0.75"
    chunked_prefill_size = "1024"
    extra_launch_args = ["--tp", "2", "--dtype", "bfloat16", "--base-gpu-id", "1"]


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestBasicSanityDSparkXpuIntelXpu(_dspark_base._DSparkSanityMixin, CustomTestCase):
    attention_backend = "intel_xpu"
    draft_attention_backend = "intel_xpu"
    page_size = "128"
    mem_fraction_static = "0.85"
    max_running_requests = "2"
    chunked_prefill_size = "512"
    extra_launch_args = ["--tp", "2", "--dtype", "bfloat16", "--base-gpu-id", "1"]

    def test_logprob_decode_match_prefill(self):
        # intel_xpu bf16 matmul is not batch-invariant, so the decode (verify,
        # M-rows) and prefill-rescore (full-length) paths drift in the deep logit
        # tail here (sampled / top-k tokens still match). Skipped on production
        # numerics; run under --enable-deterministic-inference to assert equality.
        self.skipTest("intel_xpu decode-vs-prefill deep-tail drift is batch-variance")


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestBasicSanityDSparkXpuIntelXpuDecodeGraph(TestBasicSanityDSparkXpuIntelXpu):
    extra_launch_args = TestBasicSanityDSparkXpuIntelXpu.extra_launch_args + [
        "--cuda-graph-backend-decode",
        "full",
    ]


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestDSparkXpuIntelXpuLogprobDeterministic(SpecLogprobKit, CustomTestCase):
    """Run only the shared logprob checks (SpecLogprobKit) on an intel_xpu DSPARK
    server launched with --enable-deterministic-inference.

    intel_xpu bf16 matmul is not batch-invariant, so decode (verify) vs
    prefill-rescore drift in the deep logit tail on production numerics -- which
    is why test_logprob_decode_match_prefill is skipped on the production classes.
    Deterministic inference pins the reductions so the two paths agree, letting the
    logprob suite (including the decode-vs-prefill check) run here.
    """

    model = _dspark_base.TARGET_MODEL
    process = None

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Reuse the production intel_xpu launch config; add only the deterministic
        # flag here so it stays scoped to this logprob-consistency class.
        src = TestBasicSanityDSparkXpuIntelXpu
        cls.process = popen_launch_server(
            _dspark_base.TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                src.attention_backend,
                "--speculative-draft-attention-backend",
                src.draft_attention_backend,
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                _dspark_base.DRAFT_MODEL,
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                src.mem_fraction_static,
                "--page-size",
                src.page_size,
                "--enable-metrics",
                "--cuda-graph-backend-prefill=disabled",
                "--max-running-requests",
                src.max_running_requests,
                "--chunked-prefill-size",
                src.chunked_prefill_size,
                "--enable-deterministic-inference",
                *src.extra_launch_args,
            ],
            env={
                "SGLANG_ENABLE_METRICS_DEVICE_TIMER": "1",
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
