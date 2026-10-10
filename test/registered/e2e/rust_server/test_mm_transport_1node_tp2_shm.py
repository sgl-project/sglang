"""Rust server multimodal hand-off: one node, TP 2, ``--mm-feature-transport cpu``.

TP > 1 on a single node is the one layout that hands features over as POSIX
shm segments: the TP broadcast carries a ~100-byte stub per item and every
rank maps the segment in parallel. Guards that the stub path is selected there
(rather than pushing tens of MB per image through ``broadcast_pyobj``) and that
building the stub reads nothing.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.rust_mm_transport_fixture import (
    RustMmTransportServerBase,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")


class TestRustMmTransport1NodeTp2Shm(RustMmTransportServerBase):
    transport = "cpu"
    nnodes = 1
    tp = 2
    expected_handoff = "shm"


if __name__ == "__main__":
    unittest.main(verbosity=3)
