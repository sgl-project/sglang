"""Step5 FFN reductions follow stage-boundary ownership."""

import unittest
from unittest.mock import MagicMock, patch

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


if __name__ == "__main__":
    unittest.main()
