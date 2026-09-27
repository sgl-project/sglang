"""A Step-3.5 dense layer must leave its all-reduce out when postprocess
reduce-scatters its output; otherwise the output is summed twice."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.runtime_context import get_forward
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

    communicator = SimpleNamespace(
        prepare_attn=lambda h, r, fb, **_: (h, h if r is None else r),
        prepare_mlp=lambda h, r, fb: (h, r),
        should_use_reduce_scatter=lambda fb: reduce_scatter,
        postprocess_layer=MagicMock(side_effect=lambda h, r, fb: (h, r)),
    )
    object.__setattr__(layer, "layer_communicator", communicator)
    object.__setattr__(layer, "use_moe", False)
    object.__setattr__(layer, "mlp", mlp)
    object.__setattr__(layer, "self_attn", lambda **kwargs: kwargs["hidden_states"])
    return layer, seen


class TestStep3p5DenseReduceScatter(CustomTestCase):
    def test_dense_mlp_sees_the_postprocess_reduce_scatter(self):
        for reduce_scatter in (False, True):
            with self.subTest(reduce_scatter=reduce_scatter):
                layer, seen = dense_layer(reduce_scatter=reduce_scatter)
                layer.forward(
                    positions=None,
                    hidden_states=torch.ones(2, 4),
                    forward_batch=None,
                    residual=None,
                )
                self.assertEqual(seen["mlp_reduce_scatter"], reduce_scatter)
                self.assertFalse(seen["fuse_mlp_allreduce"])
                layer.layer_communicator.postprocess_layer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
