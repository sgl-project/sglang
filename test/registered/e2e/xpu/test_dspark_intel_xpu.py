"""DSPARK speculative decoding on Intel XPU."""

import os
import sys
import unittest

from sglang.srt.utils import is_xpu

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
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=2400, suite="nightly-xpu-2-gpu", nightly=True)


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestBasicSanityDSparkXpuTriton(_dspark_base._DSparkSanityMixin, CustomTestCase):
    attention_backend = "triton"
    draft_attention_backend = "triton"
    max_running_requests = "2"
    mem_fraction_static = "0.75"
    chunked_prefill_size = "1024"
    extra_launch_args = ["--tp", "2", "--dtype", "bfloat16"]


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestBasicSanityDSparkXpuIntelXpu(_dspark_base._DSparkSanityMixin, CustomTestCase):
    attention_backend = "intel_xpu"
    draft_attention_backend = "intel_xpu"
    page_size = "128"
    mem_fraction_static = "0.85"
    max_running_requests = "2"
    chunked_prefill_size = "512"
    extra_launch_args = ["--tp", "2", "--dtype", "bfloat16"]


@unittest.skipUnless(is_xpu(), "Intel XPU required")
class TestBasicSanityDSparkXpuIntelXpuDecodeGraph(TestBasicSanityDSparkXpuIntelXpu):
    extra_launch_args = TestBasicSanityDSparkXpuIntelXpu.extra_launch_args + [
        "--cuda-graph-backend-decode",
        "full",
    ]


if __name__ == "__main__":
    unittest.main()
