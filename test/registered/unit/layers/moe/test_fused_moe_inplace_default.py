"""FusedMoE must not overwrite its input unless the caller opts in.

Dual-stream MoE blocks read the MoE input for the shared experts on another stream
while the routed experts run; an in-place triton runner races that read.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import inspect
import unittest

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.test.test_utils import CustomTestCase


class FusedMoEInplaceDefaultTest(CustomTestCase):
    def test_fused_moe_defaults_to_out_of_place(self):
        default = inspect.signature(FusedMoE.__init__).parameters["inplace"].default
        self.assertFalse(default)

    def test_runner_config_defaults_to_out_of_place(self):
        self.assertFalse(MoeRunnerConfig().inplace)


if __name__ == "__main__":
    unittest.main()
