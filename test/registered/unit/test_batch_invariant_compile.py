# SPDX-License-Identifier: Apache-2.0
"""Deterministic GEMM must survive Dynamo as an opaque dispatcher call."""

import torch

from sglang.srt.batch_invariant_ops.compiled import batch_invariant_linear
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def test_compiled_linear_keeps_dispatch_boundary():
    seen = []

    def backend(graph, example_inputs):
        seen.extend(node.target for node in graph.graph.nodes)
        return graph.forward

    compiled = torch.compile(batch_invariant_linear, backend=backend, fullgraph=True)
    generator = torch.Generator().manual_seed(20261127)
    weight = torch.randn(32, 16, generator=generator)
    bias = torch.randn(32, generator=generator)
    for count in (1, 8, 16):
        x = torch.randn(count, 16, generator=generator)
        expected = torch.nn.functional.linear(x, weight, bias)
        torch.testing.assert_close(compiled(x, weight, bias), expected, rtol=0, atol=0)
    assert torch.ops.sglang.batch_invariant_linear.default in seen
