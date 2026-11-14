"""MI35x InferenceX AgentX curve for DeepSeek-V4.1-Flash, one point per run.

Runs the point AGENTX_POINT (``tp2-c1``, ``tp2-dpa-c64``, ...) of InferenceX's
``dsv41flash-fp4-mi355x-sglang-agentic-dspark`` row against this checkout; see
``sglang.test.agentx_common``. InferenceX runs this row on rocm10, as the
nightly does.

Registry: nightly-perf-8-gpu-mi35x-dsv41-flash-agentx suite
"""

import unittest

from sglang.test.agentx_common import POINT, run_point
from sglang.test.ci.ci_register import register_amd_ci

# Register for AMD CI - one AgentX point: model load, up to an hour of warmup,
# then InferenceX's one-hour measurement window.
register_amd_ci(
    est_time=10800, suite="nightly-perf-8-gpu-mi35x-dsv41-flash-agentx", nightly=True
)


# Not CustomTestCase: its CI retry would rerun a failed two-hour point and report
# the second attempt's numbers as a pass.
@unittest.skipUnless(POINT, "AGENTX_POINT is not set")
class TestDeepSeekV41FlashAgentXMI35x(unittest.TestCase):
    def test_agentx_point(self):
        run_point(
            self,
            config_key="dsv41flash-fp4-mi355x-sglang-agentic-dspark",
            title="DeepSeek-V4.1-Flash",
        )


if __name__ == "__main__":
    unittest.main()
