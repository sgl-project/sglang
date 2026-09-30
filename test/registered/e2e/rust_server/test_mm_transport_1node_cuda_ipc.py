"""Rust server multimodal hand-off: one node, TP 1, ``--mm-feature-transport cuda_ipc``.

``cuda_ipc`` reserves a GPU feature pool out of the KV budget and, on the
Python processor path, moves features through CUDA IPC. The Rust path produces
host features regardless, so the hand-off must stay the inline zero-copy shape
and the flag must not make the server allocate or copy on its behalf.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.rust_mm_transport_fixture import (
    RustMmTransportServerBase,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


class TestRustMmTransport1NodeCudaIpc(RustMmTransportServerBase):
    transport = "cuda_ipc"
    nnodes = 1
    tp = 1
    expected_handoff = "inline"


if __name__ == "__main__":
    unittest.main(verbosity=3)
