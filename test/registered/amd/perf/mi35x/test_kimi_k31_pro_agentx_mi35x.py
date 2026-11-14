"""MI35x InferenceX AgentX curve for Kimi-K3.1-Pro MXFP4 with DFlash speculative
decoding, one point per run.

Runs the point AGENTX_POINT (``tp8-c1``, ``tp8-c4``, ``tp8-dcp8-c8``, ...) of
InferenceX's ``kimik3.1-fp4-mi355x-sglang-agentic-dflash`` row against this
checkout; see ``sglang.test.agentx_common``. The row has two TP8 arms: GPU KV
with the AMD monokernel at c1 and c4, and DCP8 with HiCache host-DRAM offload
from c8 to c56. InferenceX runs this row on a Kimi-K3.1 dev image and the
nightly on the rocm10 CI image, so its published points are a reference, not a
like-for-like comparison: several env switches the recipe sets come from the
dev branch and are not read by main.

The row is pinned to its own InferenceX commit: the harness's commit has an
older version of it (no c24, other draft-token counts).

moonshotai/Kimi-K3.1-Pro (~1.56 TB) must already be in the node's HF cache: a
cold download does not finish within the server start timeout.

Registry: nightly-perf-8-gpu-mi35x-kimi-k31-pro-agentx suite
"""

import unittest

from sglang.test.agentx_common import POINT, run_point
from sglang.test.ci.ci_register import register_amd_ci

# Register for AMD CI - one AgentX point: model load, then InferenceX's one-hour
# measurement window (this row has no warmup grace).
register_amd_ci(
    est_time=9000, suite="nightly-perf-8-gpu-mi35x-kimi-k31-pro-agentx", nightly=True
)

INFERENCEX_COMMIT = "d40220b7bc455a6f146f202ea08b64597a9b27f4"


# Not CustomTestCase: its CI retry would rerun a failed two-hour point and report
# the second attempt's numbers as a pass.
@unittest.skipUnless(POINT, "AGENTX_POINT is not set")
class TestKimiK31ProAgentXMI35x(unittest.TestCase):
    def test_agentx_point(self):
        run_point(
            self,
            config_key="kimik3.1-fp4-mi355x-sglang-agentic-dflash",
            title="Kimi-K3.1-Pro MXFP4 DFlash",
            inferencex_commit=INFERENCEX_COMMIT,
        )


if __name__ == "__main__":
    unittest.main()
