"""MI35x InferenceX AgentX curve for GLM-5.2 MXFP4 with EAGLE MTP, one point per
run.

Runs the point AGENTX_POINT (``tp8-c1``, ``tp4-c16``, ``tp4-dcp4-c48``, ...) of
InferenceX's ``glm5.2-fp4-mi355x-sglang-agentic-mtp`` row against this checkout;
see ``sglang.test.agentx_common``. The row has three arms: TP8 on GPU KV, TP4
with HiCache host-DRAM offload, and TP4 with DCP4. Both TP4 arms have a c16
point, so the DCP4 one is ``tp4-dcp4-c16``. InferenceX runs this row on rocm720
and the nightly on rocm10, so its published points are a reference, not a
like-for-like comparison.

amd/GLM-5.2-MXFP4 (~190 GB) must already be in the node's HF cache: a cold
download through hf_xet does not finish within the server start timeout.

Registry: nightly-perf-8-gpu-mi35x-glm52-mxfp4-agentx suite
"""

import unittest

from sglang.test.agentx_common import POINT, run_point
from sglang.test.ci.ci_register import register_amd_ci

# Register for AMD CI - one AgentX point: model load, then InferenceX's one-hour
# measurement window (this row has no warmup grace).
register_amd_ci(
    est_time=9000, suite="nightly-perf-8-gpu-mi35x-glm52-mxfp4-agentx", nightly=True
)


# Not CustomTestCase: its CI retry would rerun a failed two-hour point and report
# the second attempt's numbers as a pass.
@unittest.skipUnless(POINT, "AGENTX_POINT is not set")
class TestGLM52MXFP4AgentXMI35x(unittest.TestCase):
    def test_agentx_point(self):
        run_point(
            self,
            config_key="glm5.2-fp4-mi355x-sglang-agentic-mtp",
            title="GLM-5.2 MXFP4 MTP",
        )


if __name__ == "__main__":
    unittest.main()
