"""Rust server multimodal hand-off: two nodes, TP 2, ``--mm-feature-transport cpu``.

Two nodes emulated on one host (one GPU each). Across nodes POSIX shm cannot
be shared, so the tensor transport resolves to ``default`` and the Rust path
must hand features over inline on node 0, then let the cross-node broadcast
carry them. Guards the topology decision and that node 0's hand-off is still
zero-copy.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.rust_mm_transport_fixture import (
    RustMmTransportServerBase,
)

register_cuda_ci(est_time=150, stage="base-b", runner_config="2-gpu-large")


class TestRustMmTransport2NodeCpu(RustMmTransportServerBase):
    transport = "cpu"
    nnodes = 2
    tp = 2
    expected_handoff = "inline"


if __name__ == "__main__":
    unittest.main(verbosity=3)
