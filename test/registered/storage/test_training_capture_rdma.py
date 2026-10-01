"""Real PD samples written to a remote-only RDMA Store, then read after exit.

P/D serving share the local GPU and use TCP for their own KV handoff. The Store
segment must run on another physical node; this file verifies the RDMA storage
path, not a cross-node PD transfer. The Catalog remains a test double.
"""

import json
import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.pd_capture_runtime import PDCaptureRuntimeBase
from sglang.test.training_capture_rdma import load_setup

register_cuda_ci(
    est_time=720,
    stage="base-b",
    runner_config="1-gpu-small",
    disabled="Requires a separate RDMA Store node and TRAINING_CAPTURE_RDMA_SETUP",
)


@unittest.skipUnless(
    os.environ.get("TRAINING_CAPTURE_RDMA_SETUP"), "dedicated RDMA Store required"
)
class TestRemoteRDMACapture(PDCaptureRuntimeBase):
    @classmethod
    def start_store(cls):
        cls.setup_path = os.environ["TRAINING_CAPTURE_RDMA_SETUP"]
        cls.store_setup = load_setup(cls.setup_path)

    def setUp(self):
        super().setUp()
        self.servers = []

    def launch(self, *args, **kwargs):
        process, url = super().launch(*args, **kwargs)
        self.servers.append(process)
        return process, url

    def exercise_remote(self, *, replay, draft_kind=None):
        self.exercise(replay=replay, draft_kind=draft_kind)
        for process in reversed(self.servers):
            self.stop_process(process)
            self.assertIsNotNone(process.poll())
        publications = self.root / "publications.json"
        publications.write_text(json.dumps(list(self.catalog.publications.values())))
        output = self.root / "readback.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "sglang.test.training_capture_rdma",
                "read",
                "--setup",
                self.setup_path,
                "--publications",
                str(publications),
                "--output",
                str(output),
            ],
            check=True,
            timeout=120,
        )
        snapshots = json.loads(output.read_text())["snapshots"]
        self.assertEqual(len(snapshots), len(self.catalog.publications))
        self.assertEqual(
            {(row["sample_id"], row["generation_id"]) for row in snapshots},
            {
                (row["sample_id"], row["generation_id"])
                for row in self.catalog.publications.values()
            },
        )
        self.assertTrue(all(row["tensor_bytes"] > 0 for row in snapshots))

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
