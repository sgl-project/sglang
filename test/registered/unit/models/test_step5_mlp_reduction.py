"""Step5 FFN reductions follow stage-boundary ownership."""

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
    def test_dense_mlp_leaves_its_partial_to_the_stage_boundary(self):
        self.assertFalse(_down_proj_kwargs("dense")["reduce_results"])

    def test_attention_dp_dense_mlp_reduces_without_stage_boundaries(self):
        self.assertTrue(
            _down_proj_kwargs("dense", use_dp_attention_reduce=True)["reduce_results"]
        )

    def test_shared_expert_leaves_its_partial_to_the_moe_sum(self):
        self.assertFalse(_down_proj_kwargs("shared")["reduce_results"])
        self.assertTrue(
            _down_proj_kwargs("shared", use_dp_attention_reduce=True)["reduce_results"]
        )


def _moe_ffn_output(*, tp_size: int, sum_moves_to_next_layer: bool) -> torch.Tensor:
    layer = step5.Step5DecoderLayer.__new__(step5.Step5DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.use_moe = True
    layer.use_dp_attention_reduce = False
    shared, routed = torch.ones(2, 4), torch.full((2, 4), 2.0)
    layer.moe = MagicMock(return_value=(shared, routed), use_mega_moe=False)
    with (
        patch.object(
            moe_utils, "get_parallel", return_value=SimpleNamespace(tp_size=tp_size)
        ),
        patch.object(
            moe_utils,
            "should_skip_post_experts_all_reduce",
            return_value=sum_moves_to_next_layer,
        ),
        patch(
            "sglang.srt.distributed.communication_op.tensor_model_parallel_all_reduce",
            lambda x: tp_size * x,
        ),
    ):
        return layer._forward_ffn(
            hidden_states=torch.zeros(2, 4),
            forward_batch=SimpleNamespace(num_token_non_padded=None),
        )


class TestStep5MoeReduction(CustomTestCase):
    def test_stage_boundary_owns_the_moe_sum_for_every_tp_size(self):
        # A producer always returns partials, whether the boundary completes
        # the sum now (LoRA/terminal layer) or carries it to the next input.
        for tp_size in (1, 2, 4, 8):
            for defer in (False, True):
                with self.subTest(tp_size=tp_size, defer=defer):
                    torch.testing.assert_close(
                        _moe_ffn_output(tp_size=tp_size, sum_moves_to_next_layer=defer),
                        torch.full((2, 4), 3.0),
                    )


if __name__ == "__main__":
    unittest.main()
