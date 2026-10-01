"""Real PD samples written to a remote-only RDMA Store, then read after exit.

P/D serving share the local GPU and use TCP for their own KV handoff. The Store
segment must run on another physical node; this file verifies the RDMA storage
path, not a cross-node PD transfer. The Catalog remains a test double.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.training_capture_rdma import RDMACaptureRuntimeBase

register_cuda_ci(
    est_time=720,
    stage="base-b",
    runner_config="1-gpu-small",
    disabled="Requires a separate RDMA Store node and TRAINING_CAPTURE_RDMA_SETUP",
)


@unittest.skipUnless(
    os.environ.get("TRAINING_CAPTURE_RDMA_SETUP"), "dedicated RDMA Store required"
)
class TestRemoteRDMACapture(RDMACaptureRuntimeBase):
    def test_ar_eager(self):
        self.exercise_remote(replay=False)

    def test_ar_graph(self):
        self.exercise_remote(replay=True)

    def test_dspark_eager(self):
        self.exercise_remote(replay=False, draft_kind="target_kv")

    def test_dspark_graph(self):
        self.exercise_remote(replay=True, draft_kind="target_kv")


if __name__ == "__main__":
    unittest.main()
