"""MI35x InferenceX AgentX curve for Muse Spark Open 816B MXFP4 with DFlash2
speculative decoding, one point per run.

Runs the point AGENTX_POINT (``tp8-c6``, ``tp4-c28``, ...) of InferenceX's
``musespark-fp4-mi355x-sglang-agentic-dflash`` row against this checkout; see
``sglang.test.agentx_common``. The row has two arms: TP8/EP8 on GPU KV (c6 and
c10 with the AMD monokernel) and TP4/EP4 with HiCache host-DRAM offload. Both
arms have a c28 point. InferenceX runs this row on its Muse Spark day-zero image
and the nightly on the rocm10 CI image, so its published points are a
reference, not a like-for-like comparison.

The recipe serves with HF_HUB_OFFLINE=1, so amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4
(~450 GB, at the recipe's revision) and RadixArk/Muse-Spark-Open-DFlash2 must
already be in the node's HF cache.

Registry: nightly-perf-test-8-gpu-amd-mi35x-muse-spark-agentx suite
"""

import unittest

from sglang.test.agentx_common import POINT, run_point
from sglang.test.ci.ci_register import register_amd_ci

# The harness's pinned commit has this row with an older tp8-c6 stream interval.
INFERENCEX_COMMIT = "67c3ed26f9c196b3e8286053a44838b4f0024756"

# Register for AMD CI - one AgentX point: model load, then InferenceX's one-hour
# measurement window (this row has no warmup grace).
register_amd_ci(
    est_time=9000,
    suite="nightly-perf-test-8-gpu-amd-mi35x-muse-spark-agentx",
    nightly=True,
)


# Not CustomTestCase: its CI retry would rerun a failed two-hour point and report
# the second attempt's numbers as a pass.
@unittest.skipUnless(POINT, "AGENTX_POINT is not set")
class TestMuseSparkAgentXMI35x(unittest.TestCase):
    def test_agentx_point(self):
        run_point(
            self,
            config_key="musespark-fp4-mi355x-sglang-agentic-dflash",
            title="Muse Spark Open MXFP4 DFlash2",
            inferencex_commit=INFERENCEX_COMMIT,
        )


if __name__ == "__main__":
    unittest.main()
