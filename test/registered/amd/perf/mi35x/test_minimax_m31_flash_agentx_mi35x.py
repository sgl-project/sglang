"""MI35x InferenceX AgentX curve for MiniMax-M3.1-Flash with DSpark, one point
per run.

Runs the point AGENTX_POINT (``tp4-c1``, ``tp4-c48``, ...) of InferenceX's
``minimaxm31flash-fp4-mi355x-sglang-agentic-dspark`` row against this checkout;
see ``sglang.test.agentx_common``. The row is the cookbook's MI355X TP4 cells:
GPU KV up to c32, HiCache host-DRAM offload at c48 and c64. InferenceX runs this
row on a MiniMax day-zero image and the nightly on rocm10, so its published
points are a reference, not a like-for-like comparison.

MiniMaxAI/MiniMax-M3.1-Flash (~250 GB) must already be in the node's HF cache:
a cold download through hf_xet does not finish within the server start timeout.

Registry: nightly-perf-8-gpu-mi35x-minimax-m31-flash-agentx suite
"""

import unittest

from sglang.test.agentx_common import POINT, run_point
from sglang.test.ci.ci_register import register_amd_ci

# Register for AMD CI - one AgentX point: model load, then InferenceX's one-hour
# measurement window (this row has no warmup grace).
register_amd_ci(
    est_time=9000,
    suite="nightly-perf-8-gpu-mi35x-minimax-m31-flash-agentx",
    nightly=True,
)


# Not CustomTestCase: its CI retry would rerun a failed two-hour point and report
# the second attempt's numbers as a pass.
@unittest.skipUnless(POINT, "AGENTX_POINT is not set")
class TestMiniMaxM31FlashAgentXMI35x(unittest.TestCase):
    def test_agentx_point(self):
        run_point(
            self,
            config_key="minimaxm31flash-fp4-mi355x-sglang-agentic-dspark",
            title="MiniMax-M3.1-Flash DSpark",
        )


if __name__ == "__main__":
    unittest.main()
