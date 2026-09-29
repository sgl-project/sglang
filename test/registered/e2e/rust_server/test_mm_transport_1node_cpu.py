"""Rust server multimodal hand-off: one node, TP 1, ``--mm-feature-transport cpu``.

The single-rank layout hands features over inline (a numpy array owning the
Rust vector, viewed by torch). Guards the hand-off staying zero-copy under the
default transport, which is the layout every 1-GPU VLM deployment runs.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.rust_mm_transport_fixture import (
    RustMmTransportServerBase,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


class TestRustMmTransport1NodeCpu(RustMmTransportServerBase):
    transport = "cpu"
    nnodes = 1
    tp = 1
    expected_handoff = "inline"


if __name__ == "__main__":
    unittest.main(verbosity=3)
