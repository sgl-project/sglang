"""Lossless output parity: spec-decode greedy output == a non-spec reference.

The reference is a separate non-spec server, launched and torn down BEFORE the
spec server (sequential -- one model resident at a time; see SpecParityKit).
"""

import unittest

from sglang.srt.environ import envs
from sglang.srt.utils import is_xpu
from sglang.test.ci.ci_register import register_cuda_ci, register_xpu_ci
from sglang.test.kits.spec_server_kits import SpecParityKit
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=118, stage="base-b", runner_config="1-gpu-large")
register_xpu_ci(
    est_time=360,
    stage="stage-b",
    runner_config="1-gpu-xpu",
    disabled="EAGLE3 numerical parity mismatches on XPU",
)

_is_xpu = is_xpu()


class _Eagle3ParityBase(Eagle3Base):
    """Shared knobs for EAGLE3 parity variants; no test methods."""

    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


@unittest.skipIf(_is_xpu, "CUDA runner only")
class TestEagle3ParityCUDA(SpecParityKit, _Eagle3ParityBase):
    """EAGLE3 spec v2 (flashinfer, overlap) greedy output == non-spec reference.

    SpecParityKit is first so its setUpClass runs the reference server (and tears
    it down) before the fixture launches the spec server -- sequential, one model
    at a time.
    """

    disable_overlap = False


@unittest.skipUnless(_is_xpu, "XPU runner only")
class TestEagle3ParityXPU(SpecParityKit, _Eagle3ParityBase):
    """EAGLE3 parity on XPU (triton, no overlap, deterministic)."""

    disable_overlap = False
    attention_backend = "triton"
    # Decode full-graph was active by default when this test was added
    # (via XPUCudaGraphBackend). Opt in explicitly now that it is disabled
    # by default so the coverage is preserved.
    extra_args = ("--cuda-graph-config", '{"decode":{"backend":"full"}}')


if __name__ == "__main__":
    unittest.main()
