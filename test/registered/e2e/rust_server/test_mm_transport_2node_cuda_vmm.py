"""Rust server multimodal hand-off: two nodes, TP 2, ``--mm-feature-transport cuda_vmm``.

The layout ``cuda_vmm`` exists for: a multi-node GB200/GB300 deployment of a
model that declares ``supports_cuda_vmm_feature_transport``. Two nodes are
emulated on one GB300 host. The Rust path produces host features, so node 0
must still hand them over inline and zero-copy, and the scheduler's VMM
materialize step must pass the host tensors through untouched on both ranks.

Nightly on the GB300 pool, where the flag's driver checks are meaningful.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.rust_mm_transport_fixture import (
    CUDA_VMM_MODEL,
    RustMmTransportServerBase,
)

register_cuda_ci(est_time=180, stage="nightly", runner_config="4-gpu-gb300")


class TestRustMmTransport2NodeCudaVmm(RustMmTransportServerBase):
    model = CUDA_VMM_MODEL
    transport = "cuda_vmm"
    nnodes = 2
    tp = 2
    expected_handoff = "inline"


if __name__ == "__main__":
    unittest.main(verbosity=3)
