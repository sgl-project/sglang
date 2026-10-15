"""Runtime residual bookkeeping must remain traceable in fullgraph mode."""

import pytest
import torch

from sglang.srt.layers.layer_boundary.exit import ExitDecision
from sglang.srt.layers.layer_boundary.residual.add_norm import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual.stream import Contribution
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("rows", [1, 4])
def test_contribution_fullgraph(rows):
    def forward(x):
        contribution = Contribution(x * 2, PLAIN_ADD)
        value = contribution.complete()
        contribution.release()
        return value + x

    torch._dynamo.reset()
    try:
        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
        for _ in range(2):
            x = torch.randn(rows, 16)
            torch.testing.assert_close(compiled(x), forward(x))
    finally:
        torch._dynamo.reset()


def test_exit_decision_fullgraph():
    def complete(x, residual):
        return x + residual, residual

    def forward(x):
        decision = ExitDecision(False, False, complete)
        return decision.complete(x, x)[0]

    torch._dynamo.reset()
    try:
        compiled = torch.compile(forward, backend="aot_eager", fullgraph=True)
        x = torch.randn(2, 16)
        torch.testing.assert_close(compiled(x), forward(x))
    finally:
        torch._dynamo.reset()
