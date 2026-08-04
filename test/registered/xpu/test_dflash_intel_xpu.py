"""DFLASH speculative decoding on Intel XPU.

Reuses the CUDA DFLASH suite (TestDFlashServerBase) unchanged, retargeted at
the Intel GPU: flashinfer is CUDA-only, so the device is forced to xpu and the
attention backend is one of the two XPU-capable options — `triton` (the XPU
default) or `intel_xpu` (the native XPUAttentionBackend). Each config below is
covered under both backends. mem-fraction-static is raised above the ~0.75
target+draft weight floor on a single Intel GPU (the base's trailing
other_launch_args win over its hardcoded 0.7).
"""

import os
import sys
import unittest

from sglang.srt.utils.common import is_xpu

# Put the `test/` root on sys.path so `registered.<...>` resolves regardless of
# cwd: CI runs each file as `python3 <full_path>` (only the file's own dir is on
# the path), and pytest inserts only the file's dir too. `test/` is two levels
# up from this file's dir (test/registered/xpu/<this>).
_TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

# Reference the base as a module attribute rather than importing the `Test*`
# name directly: pytest collects by class __name__, so a bare
# `from ... import TestDFlashServerBase` (even aliased) would make it re-collect
# the base's CUDA/flashinfer config here. Only the XPU subclass should run.
from registered.spec.dflash import test_dflash as _dflash_base
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=600, suite="nightly-xpu-1-gpu", nightly=True)

# Appended after the base launch_args by setUpClass: the trailing
# --mem-fraction-static overrides the base 0.7, and --device selects the Intel
# GPU. Variants that need extra flags must prepend these (the base does not
# merge other_launch_args — the subclass value replaces it wholesale).
_XPU_LAUNCH_ARGS = ["--device", "xpu", "--mem-fraction-static", "0.85"]


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPU(_dflash_base.TestDFlashServerBase):
    """Full DFLASH suite on device=xpu with the triton attention backend."""

    max_running_requests = 8
    attention_backend = "triton"
    other_launch_args = _XPU_LAUNCH_ARGS


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUPage256(_dflash_base.TestDFlashServerPage256):
    """page_size=256 + radix-attention smoke test on XPU."""

    max_running_requests = 8
    attention_backend = "triton"
    other_launch_args = _XPU_LAUNCH_ARGS


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUChunkedPrefill(_dflash_base.TestDFlashServerChunkedPrefill):
    """Chunked prefill (size 4) on XPU."""

    max_running_requests = 8
    attention_backend = "triton"
    # XPU args first, then the variant's own --chunked-prefill-size.
    other_launch_args = _XPU_LAUNCH_ARGS + ["--chunked-prefill-size", "4"]


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUNoCudaGraph(_dflash_base.TestDFlashServerNoCudaGraph):
    """CUDA-graph disabled on XPU."""

    max_running_requests = 8
    attention_backend = "triton"
    other_launch_args = _XPU_LAUNCH_ARGS + ["--disable-cuda-graph"]


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUSpecV2(_dflash_base.TestDFlashServerSpecV2):
    """SpecV2 (overlap schedule enabled) on XPU."""

    max_running_requests = 8
    attention_backend = "triton"
    other_launch_args = _XPU_LAUNCH_ARGS


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUSpecV2PlanStream(_dflash_base.TestDFlashServerSpecV2PlanStream):
    """SpecV2 with the overlap plan stream on XPU."""

    max_running_requests = 8
    attention_backend = "triton"
    other_launch_args = _XPU_LAUNCH_ARGS


# --- Native XPUAttentionBackend (intel_xpu) variants --------------------------
# Same configs as above, but exercising the native intel_xpu backend rather than
# triton. intel_xpu is not the XPU default (triton is), so it must be selected
# explicitly; these guard the DFLASH draft/verify path through XPUAttentionBackend.


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUBackend(_dflash_base.TestDFlashServerBase):
    """Full DFLASH suite on device=xpu with the intel_xpu attention backend."""

    max_running_requests = 8
    attention_backend = "intel_xpu"
    other_launch_args = _XPU_LAUNCH_ARGS


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUBackendPage256(_dflash_base.TestDFlashServerPage256):
    """page_size=256 + radix-attention smoke test on XPU (intel_xpu backend)."""

    max_running_requests = 8
    attention_backend = "intel_xpu"
    other_launch_args = _XPU_LAUNCH_ARGS


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUBackendChunkedPrefill(
    _dflash_base.TestDFlashServerChunkedPrefill
):
    """Chunked prefill (size 128) on XPU (intel_xpu backend)."""

    max_running_requests = 8
    attention_backend = "intel_xpu"
    other_launch_args = _XPU_LAUNCH_ARGS + ["--chunked-prefill-size", "128"]


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUBackendNoCudaGraph(_dflash_base.TestDFlashServerNoCudaGraph):
    """CUDA-graph disabled on XPU (intel_xpu backend)."""

    max_running_requests = 8
    attention_backend = "intel_xpu"
    other_launch_args = _XPU_LAUNCH_ARGS + ["--disable-cuda-graph"]


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUBackendSpecV2(_dflash_base.TestDFlashServerSpecV2):
    """SpecV2 (overlap schedule enabled) on XPU (intel_xpu backend)."""

    max_running_requests = 8
    attention_backend = "intel_xpu"
    other_launch_args = _XPU_LAUNCH_ARGS


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
class TestDFlashIntelXPUBackendSpecV2PlanStream(
    _dflash_base.TestDFlashServerSpecV2PlanStream
):
    """SpecV2 with the overlap plan stream on XPU (intel_xpu backend)."""

    max_running_requests = 8
    attention_backend = "intel_xpu"
    other_launch_args = _XPU_LAUNCH_ARGS


if __name__ == "__main__":
    unittest.main()
