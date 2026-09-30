"""A Step-3.5 dense layer must leave its all-reduce out when postprocess
reduce-scatters its output; otherwise the output is summed twice. The layer's
FFN exit publishes the decision while the MLP runs, then completes it."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers import layer_boundary as comm
from sglang.srt.layers.layer_boundary.residual.stream import ResidualStream
from sglang.srt.runtime_context import get_forward
from sglang.test.boundary_fixtures import stub_plan
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def dense_layer(*, reduce_scatter):
    step3p5 = importlib.import_module("sglang.srt.models.step3p5")
    layer = step3p5.Step3p5DecoderLayer.__new__(step3p5.Step3p5DecoderLayer)
    torch.nn.Module.__init__(layer)
    seen = {}

    def mlp(hidden_states):
        # RowParallelLinear skips its all-reduce when either flag is published.
        seen["mlp_reduce_scatter"] = get_forward().mlp_reduce_scatter
        seen["fuse_mlp_allreduce"] = get_forward().fuse_mlp_allreduce
        return hidden_states

    complete = MagicMock(side_effect=lambda h, r: (h, r))
    plan = stub_plan()
    plan.path_for = lambda fb: SimpleNamespace(
        output=comm.OutputContract(comm.Layout(frozenset())),
        returns_over_dp=False,
        output_move_completes_sum=False,
    )
    plan.output._decide = lambda fb, steps: comm.ExitDecision(
        defer_moe_finalize=False,
        fuse_mlp_allreduce=False,
        mlp_reduce_scatter=reduce_scatter,
        complete=complete,
    )

    def stage_exit(fb):
        return plan.output.ffn_exit(fb, stream=fb.residual_stream)

    object.__setattr__(
        layer,
        "attn_boundary",
        SimpleNamespace(
            prepare=lambda h, fb, **kwargs: h,
            finish=lambda h, fb: h,
        ),
    )
    object.__setattr__(
        layer,
        "ffn_boundary",
        SimpleNamespace(
            prepare=lambda h, fb: h,
            exit=stage_exit,
        ),
    )
    object.__setattr__(layer, "use_moe", False)
    object.__setattr__(layer, "mlp", mlp)
    object.__setattr__(layer, "self_attn", lambda **kwargs: kwargs["hidden_states"])
    return layer, seen, complete


class TestStep3p5DenseReduceScatter(CustomTestCase):
    def test_dense_mlp_sees_the_postprocess_reduce_scatter(self):
        for reduce_scatter in (False, True):
            with self.subTest(reduce_scatter=reduce_scatter):
                layer, seen, complete = dense_layer(reduce_scatter=reduce_scatter)
                layer.forward(
                    positions=None,
                    hidden_states=torch.ones(2, 4),
                    forward_batch=SimpleNamespace(
                        residual_stream=ResidualStream(torch.ones(2, 4))
                    ),
                )
                self.assertEqual(seen["mlp_reduce_scatter"], reduce_scatter)
                self.assertFalse(seen["fuse_mlp_allreduce"])
                complete.assert_called_once()


if __name__ == "__main__":
    unittest.main()
