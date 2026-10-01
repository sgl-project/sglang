"""Cross-node P -> D RDMA, then D -> remote Store RDMA and independent reads.

P is a separately supervised test fixture. The operator stops it after this
file exits and runs the retained-publications readback before stopping Store.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_remote import CrossNodePDCaptureRuntimeBase

register_cuda_ci(
    est_time=480,
    stage="base-b",
    runner_config="1-gpu-small",
    disabled="Requires a remote RDMA P fixture, Store and shared observation path",
)


@unittest.skipUnless(
    os.environ.get("TRAINING_CAPTURE_PD_PREFILL")
    and os.environ.get("TRAINING_CAPTURE_RDMA_SETUP"),
    "dedicated RDMA P and Store required",
)
class TestCrossNodePDRDMACapture(CrossNodePDCaptureRuntimeBase):
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
