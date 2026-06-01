"""Lossless output parity: spec decode output == non-spec reference.

Launches a second non-spec reference server, so it needs the large (Hopper)
runner for the memory headroom (two models on one GPU).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.spec_server_kits import SpecParityKit
from sglang.test.server_fixtures.spec_eagle_fixture import Eagle3Base

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-large")


class TestEagle3Parity(Eagle3Base, SpecParityKit):
    """EAGLE3 spec v2 (flashinfer) greedy output == non-spec reference."""

    disable_overlap = False
    # Main 0.5 + reference 0.35 (SpecParityKit) = 0.85, both fit on an 80GB GPU.
    mem_fraction_static = 0.5


if __name__ == "__main__":
    unittest.main()
