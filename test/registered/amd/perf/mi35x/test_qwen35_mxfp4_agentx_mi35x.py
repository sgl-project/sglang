"""MI35x InferenceX AgentX curve for Qwen3.5-397B-A17B MXFP4 with EAGLE MTP,
one point per run.

Runs the point AGENTX_POINT (``tp2-c1``, ``tp2-c16``, ...) of InferenceX's
``qwen3.5-fp4-mi355x-sglang-agentic-mtp`` row against this checkout; see
``sglang.test.agentx_common``. From c16 the row offloads KV to host DRAM with
HiCache. InferenceX runs this row on rocm720 and the nightly on rocm10, so its
published points are a reference, not a like-for-like comparison.

Registry: nightly-perf-8-gpu-mi35x-qwen35-mxfp4-agentx suite
"""

import unittest

from sglang.test.agentx_common import POINT, run_point
from sglang.test.ci.ci_register import register_amd_ci

# Register for AMD CI - one AgentX point: model load, then InferenceX's one-hour
# measurement window (this row has no warmup grace).
register_amd_ci(
    est_time=10800, suite="nightly-perf-8-gpu-mi35x-qwen35-mxfp4-agentx", nightly=True
)


# Not CustomTestCase: its CI retry would rerun a failed two-hour point and report
# the second attempt's numbers as a pass.
@unittest.skipUnless(POINT, "AGENTX_POINT is not set")
class TestQwen35MXFP4AgentXMI35x(unittest.TestCase):
    def test_agentx_point(self):
        run_point(
            self,
            config_key="qwen3.5-fp4-mi355x-sglang-agentic-mtp",
            title="Qwen3.5 MXFP4 MTP",
        )


if __name__ == "__main__":
    unittest.main()
