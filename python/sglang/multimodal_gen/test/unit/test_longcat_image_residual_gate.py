from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.models.dits import longcat_image


class _JointAdaNorm(nn.Module):
    def __init__(self, attention_gate: float, mlp_gate: float):
        super().__init__()
        self.attention_gate = attention_gate
        self.mlp_gate = mlp_gate

    def forward(self, hidden_states, emb):
        batch, width = hidden_states.shape[0], hidden_states.shape[-1]
        shape = (batch, width)
        zeros = hidden_states.new_zeros(shape)
        return (
            hidden_states,
            hidden_states.new_full(shape, self.attention_gate),
            zeros,
            zeros,
            hidden_states.new_full(shape, self.mlp_gate),
        )


class _SingleAdaNorm(nn.Module):
    def forward(self, hidden_states, emb):
        gate = hidden_states.new_full(
            (hidden_states.shape[0], hidden_states.shape[-1]), 0.25
        )
        return hidden_states, gate


class _JointAttention(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states, **kwargs):
        return hidden_states * 0.1, encoder_hidden_states * 0.2


class _SingleAttention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _TupleIdentity(nn.Module):
    def forward(self, hidden_states):
        return hidden_states, None


class _MergeAttentionAndMlp(nn.Module):
    def forward(self, hidden_states):
        half = hidden_states.shape[-1] // 2
        return hidden_states[..., :half] + hidden_states[..., half:], None


def _reference_residual_gate_add(residual, update, gate):
    return residual + update * gate


def test_joint_block_uses_shared_residual_gate_add_for_all_four_updates():
    block = longcat_image._TransformerBlock.__new__(longcat_image._TransformerBlock)
    nn.Module.__init__(block)
    block.norm1 = _JointAdaNorm(attention_gate=0.25, mlp_gate=0.5)
    block.norm1_context = _JointAdaNorm(attention_gate=0.75, mlp_gate=1.0)
    block.attn = _JointAttention()
    block.norm2 = nn.Identity()
    block.ff = nn.Identity()
    block.norm2_context = nn.Identity()
    block.ff_context = nn.Identity()

    hidden_states = torch.randn(1, 3, 4)
    encoder_hidden_states = torch.randn(1, 2, 4)

    with patch.object(
        longcat_image,
        "residual_gate_add",
        side_effect=_reference_residual_gate_add,
    ) as residual_gate_add:
        encoder_out, hidden_out = block(
            hidden_states,
            encoder_hidden_states,
            temb=torch.zeros(1, 4),
        )

    assert residual_gate_add.call_count == 4
    assert torch.allclose(hidden_out, hidden_states * 1.5375)
    assert torch.allclose(encoder_out, encoder_hidden_states * 2.3)
    assert all(
        call.args[2].shape == (1, 1, 4) for call in residual_gate_add.call_args_list
    )


def test_single_block_uses_shared_residual_gate_add_before_stream_split():
    block = longcat_image._SingleTransformerBlock.__new__(
        longcat_image._SingleTransformerBlock
    )
    nn.Module.__init__(block)
    block.norm = _SingleAdaNorm()
    block.proj_mlp = _TupleIdentity()
    block.act_mlp = nn.Identity()
    block.attn = _SingleAttention()
    block.proj_out = _MergeAttentionAndMlp()

    hidden_states = torch.randn(1, 3, 4)
    encoder_hidden_states = torch.randn(1, 2, 4)
    expected = torch.cat([encoder_hidden_states, hidden_states], dim=1) * 1.5

    with patch.object(
        longcat_image,
        "residual_gate_add",
        side_effect=_reference_residual_gate_add,
    ) as residual_gate_add:
        encoder_out, hidden_out = block(
            hidden_states,
            encoder_hidden_states,
            temb=torch.zeros(1, 4),
        )

    residual_gate_add.assert_called_once()
    assert residual_gate_add.call_args.args[2].shape == (1, 1, 4)
    assert torch.allclose(encoder_out, expected[:, :2])
    assert torch.allclose(hidden_out, expected[:, 2:])
