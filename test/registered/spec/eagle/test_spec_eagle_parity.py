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

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-large")
register_xpu_ci(est_time=360, stage="stage-b", runner_config="1-gpu-xpu")

_is_xpu = is_xpu()


class TestEagle3Parity(SpecParityKit, Eagle3Base):
    """EAGLE3 spec v2 (flashinfer) greedy output == non-spec reference.

    SpecParityKit is first so its setUpClass runs the reference server (and tears
    it down) before the fixture launches the spec server -- sequential, one model
    at a time.
    """

    if _is_xpu:
        disable_overlap = True
        attention_backend = "triton"
    else:
        disable_overlap = False
    env_overrides = ((envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY, 1),)


if __name__ == "__main__":
    unittest.main()
