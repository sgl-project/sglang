from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import layernorm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Norm:
    variance_epsilon = 1e-6

    def forward(self, hidden_states, residual=None, post_residual_addition=None):
        return hidden_states + residual, residual


def test_flashinfer_runtime_decline_materializes_owned_reduction():
    hidden_states = torch.tensor([[1.0, 2.0]])
    residual = torch.tensor([[3.0, 4.0]])

    with (
        patch.object(layernorm, "_use_aiter", False),
        patch.object(
            layernorm,
            "get_parallel",
            return_value=SimpleNamespace(
                attn_tp_size=2,
                moe_ep_size=1,
                moe_tp_size=2,
            ),
        ),
        patch(
            "sglang.srt.layers.flashinfer_comm_fusion."
            "flashinfer_allreduce_residual_rmsnorm",
            return_value=(None, None),
        ),
        patch(
            "sglang.srt.distributed.tensor_model_parallel_all_reduce",
            return_value=hidden_states + 10,
        ) as all_reduce,
    ):
        output, output_residual = layernorm._forward_with_allreduce_fusion(
            _Norm(),
            hidden_states,
            residual,
            post_residual_addition=None,
            weight=torch.ones(2),
            use_attn_tp_group=False,
        )

    all_reduce.assert_called_once_with(hidden_states)
    torch.testing.assert_close(output, torch.tensor([[14.0, 16.0]]))
    torch.testing.assert_close(output_residual, residual)
