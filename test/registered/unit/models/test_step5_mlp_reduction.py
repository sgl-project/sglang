"""Step5 FFN reductions: a dense MLP owns its TP reduction, so the last decoder layer's
output is summed; an MoE layer's routed + shared partial is summed unless the layer
exit moved the sum to the next layer."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.moe import utils as moe_utils
from sglang.srt.models import step5
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _down_proj_kwargs(role: str, use_dp_attention_reduce: bool = False) -> dict:
    row_parallel = MagicMock()
    with (
        patch.object(step5, "MergedColumnParallelLinear", MagicMock()),
        patch.object(step5, "RowParallelLinear", row_parallel),
        patch.object(step5, "SiluClampAndMulOp", MagicMock()),
    ):
        step5.Step5MLP(
            role=role,
            hidden_size=8,
            intermediate_size=16,
            swiglu_limit=7.0,
            tp_rank=0,
            tp_size=4,
            use_dp_attention_reduce=use_dp_attention_reduce,
        )
    return row_parallel.call_args.kwargs


class TestStep5MlpReduction(CustomTestCase):
    def test_dense_mlp_reduces_unless_the_boundary_defers_it(self):
        # RowParallelLinear skips the all-reduce only under the decoder's
        # fuse_mlp_allreduce scope; the terminal layer never sets it.
        self.assertTrue(_down_proj_kwargs("dense")["reduce_results"])

    def test_shared_expert_leaves_its_partial_to_the_moe_sum(self):
        self.assertFalse(_down_proj_kwargs("shared")["reduce_results"])
        self.assertTrue(
            _down_proj_kwargs("shared", use_dp_attention_reduce=True)["reduce_results"]
        )


def _moe_ffn_output(*, sum_moves_to_next_layer: bool) -> torch.Tensor:
    layer = step5.Step5DecoderLayer.__new__(step5.Step5DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.use_moe = True
    layer.use_dp_attention_reduce = False
    shared, routed = torch.ones(2, 4), torch.full((2, 4), 2.0)
    layer.moe = MagicMock(return_value=(shared, routed), use_mega_moe=False)
    with (
        patch.object(
            moe_utils, "get_parallel", return_value=SimpleNamespace(tp_size=2)
        ),
        patch.object(
            moe_utils,
            "should_skip_post_experts_all_reduce",
            return_value=sum_moves_to_next_layer,
        ),
        patch(
            "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
            lambda x: 2 * x,
        ),
    ):
        return layer._forward_ffn(
            hidden_states=torch.zeros(2, 4),
            forward_batch=SimpleNamespace(num_token_non_padded=None),
        )


class TestStep5MoeReduction(CustomTestCase):
    def test_partial_moves_to_the_next_layer_when_the_exit_defers_it(self):
        torch.testing.assert_close(
            _moe_ffn_output(sum_moves_to_next_layer=True), torch.full((2, 4), 3.0)
        )

    def test_the_moe_layer_sums_its_partial_otherwise(self):
        # e.g. under LoRA, where no fused consumer takes the deferred sum
        torch.testing.assert_close(
            _moe_ffn_output(sum_moves_to_next_layer=False), torch.full((2, 4), 6.0)
        )


if __name__ == "__main__":
    unittest.main()
