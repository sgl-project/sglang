"""Rust server multimodal hand-off: one node, TP 1, ``--mm-feature-transport cuda_vmm``.

``cuda_vmm`` is the multi-node GB200/GB300 transport; it is only defined for
models that declare ``supports_cuda_vmm_feature_transport``, so this cell runs
the smallest such model. With the Rust path producing host features, the
hand-off must stay inline and zero-copy, and the scheduler's VMM materialize
step must pass the host tensors through untouched.

Nightly on the GB300 pool: the flag's GPU pool reservation and the VMM driver
checks are only meaningful on that hardware.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.rust_mm_transport_fixture import (
    CUDA_VMM_MODEL,
    RustMmTransportServerBase,
)

register_cuda_ci(est_time=120, stage="nightly", runner_config="4-gpu-gb300")


class TestRustMmTransport1NodeCudaVmm(RustMmTransportServerBase):
    model = CUDA_VMM_MODEL
    transport = "cuda_vmm"
    nnodes = 1
    tp = 1
    expected_handoff = "inline"


if __name__ == "__main__":
    unittest.main(verbosity=3)
