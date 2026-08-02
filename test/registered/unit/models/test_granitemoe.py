import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models import granitemoe
from sglang.srt.models.granitemoe import GraniteMoeDecoderLayer
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _IdentityNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class _AddOneAttention(nn.Module):
    def forward(
        self, positions, hidden_states: torch.Tensor, forward_batch
    ) -> torch.Tensor:
        return hidden_states + 1


class _DoubleMoE(nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states * 2


class TestGraniteMoeDecoderLayer(CustomTestCase):
    def test_scaled_residual_norm_uses_fused_cuda_path(self) -> None:
        hidden_size = 512
        dtype = torch.bfloat16
        scale = 0.22
        norm = RMSNorm(hidden_size, weight_dtype=dtype).cuda()
        hidden_states = torch.randn(2, hidden_size, device="cuda", dtype=dtype)
        residual = torch.randn_like(hidden_states)
        expected_residual = residual + hidden_states * scale
        expected_hidden_states = norm(expected_residual)

        with patch.object(
            granitemoe,
            "fused_scaled_add_rmsnorm",
            wraps=granitemoe.fused_scaled_add_rmsnorm,
        ) as fused:
            actual_hidden_states, actual_residual = granitemoe._scaled_residual_rmsnorm(
                norm, hidden_states, residual, scale
            )

        fused.assert_called_once()
        torch.testing.assert_close(
            actual_hidden_states,
            expected_hidden_states,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)

    def test_scaled_residual_norm_falls_back_for_aliases(self) -> None:
        hidden_size = 512
        dtype = torch.bfloat16
        scale = 0.22
        norm = RMSNorm(hidden_size, weight_dtype=dtype).cuda()
        storage = torch.randn(2, hidden_size, device="cuda", dtype=dtype)
        hidden_states = storage
        residual = storage.view_as(storage)
        expected_residual = residual + hidden_states * scale
        expected_hidden_states = norm(expected_residual)

        with patch.object(
            granitemoe,
            "fused_scaled_add_rmsnorm",
            wraps=granitemoe.fused_scaled_add_rmsnorm,
        ) as fused:
            actual_hidden_states, actual_residual = granitemoe._scaled_residual_rmsnorm(
                norm, hidden_states, residual, scale
            )

        fused.assert_not_called()
        torch.testing.assert_close(
            actual_hidden_states,
            expected_hidden_states,
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(actual_residual, expected_residual, atol=0, rtol=0)

    def test_defers_scaled_moe_residual_to_next_layer(self) -> None:
        layer = object.__new__(GraniteMoeDecoderLayer)
        nn.Module.__init__(layer)
        layer.input_layernorm = _IdentityNorm(4)
        layer.post_attention_layernorm = _IdentityNorm(4)
        layer.self_attn = _AddOneAttention()
        layer.block_sparse_moe = _DoubleMoE()
        layer.residual_multiplier = 0.25

        initial = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        hidden_states, residual = layer(None, initial, None)
        torch.testing.assert_close(residual, initial * 1.25 + 0.25)
        torch.testing.assert_close(hidden_states, initial * 2.5 + 0.5)

        hidden_states, residual = layer(None, hidden_states, None, residual)
        torch.testing.assert_close(residual, initial * 2.34375 + 0.71875)
        torch.testing.assert_close(hidden_states, initial * 4.6875 + 1.4375)


if __name__ == "__main__":
    unittest.main()
