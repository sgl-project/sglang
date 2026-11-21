from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.models.minicpm5 import (
    MiniCPM5Attention,
    MiniCPM5DecoderLayer,
    MiniCPM5ForCausalLM,
    MiniCPM5Gate,
    MiniCPM5Model,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def test_router_computes_logits_in_fp32_with_bf16_weights():
    config = SimpleNamespace(n_routed_experts=1, hidden_size=2)
    with set_default_torch_dtype(torch.bfloat16):
        gate = MiniCPM5Gate(config)
    with torch.no_grad():
        gate.weight.fill_(1.0)

    hidden_states = torch.tensor([[1.0, 1.0 / 256]], dtype=torch.bfloat16)
    logits = gate(hidden_states)

    assert gate.weight.dtype == torch.bfloat16
    assert gate.e_score_correction_bias.dtype == torch.float32
    # This sum is exactly between two BF16 values.
    expected = torch.tensor([[1.0 + 1.0 / 256]], dtype=torch.float32)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)


def test_dspark_maps_to_the_next_attention_boundary():
    model = MiniCPM5Model.__new__(MiniCPM5Model)
    nn.Module.__init__(model)
    model.layers_to_capture = []
    target = MiniCPM5ForCausalLM.__new__(MiniCPM5ForCausalLM)
    nn.Module.__init__(target)
    target.pp_group = SimpleNamespace(is_last_rank=True)
    target.model = model
    target.capture_aux_hidden_states = False

    def configure(layer_ids):
        configure_aux_hidden_state_capture(
            model=target,
            eagle_use_aux_hidden_state=False,
            eagle_aux_hidden_state_layer_ids=None,
            dflash_use_aux_hidden_state=True,
            dflash_target_layer_ids=layer_ids,
            is_dspark=True,
        )

    assert not target.capture_aux_hidden_states
    with pytest.raises(ValueError, match="explicit layer_ids"):
        configure(None)
    configure([1, 7, 13, 19, 25])
    assert target.capture_aux_hidden_states
    assert model.layers_to_capture == [2, 8, 14, 20, 26]


def test_decoder_captures_residual_before_attention():
    layer = MiniCPM5DecoderLayer.__new__(MiniCPM5DecoderLayer)
    nn.Module.__init__(layer)
    captured = []

    def prepare_attention(hidden_states, batch, *, capture):
        capture(hidden_states)
        return hidden_states + 1

    layer.attn_boundary = SimpleNamespace(
        prepare=prepare_attention,
        finish=lambda output, batch: output + 3,
    )
    layer.self_attn = lambda positions, hidden_states, batch: hidden_states * 2
    layer.ffn_boundary = SimpleNamespace(
        prepare=lambda hidden_states, batch: hidden_states + 5,
        finish=lambda output, batch: output + 7,
    )
    layer.mlp = lambda hidden_states, batch: hidden_states * 11

    hidden_states = torch.tensor([[2.0]])
    output = layer(torch.arange(1), hidden_states, None, capture_output=captured.append)

    torch.testing.assert_close(output, torch.tensor([[161.0]]))
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], hidden_states)


def test_attention_applies_the_gate_from_each_query_head():
    layer = MiniCPM5Attention.__new__(MiniCPM5Attention)
    nn.Module.__init__(layer)
    layer.use_gated_attention = True
    layer.num_heads, layer.head_dim = 2, 2
    layer.q_size, layer.kv_size = 4, 2

    # q0, gate0, q1, gate1, k, v
    packed = torch.tensor(
        [[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 1.0, 1.0, 5.0, 6.0, 7.0, 8.0]]
    )
    layer.qkv_proj = lambda hidden: (packed, None)
    layer.rotary_emb = lambda positions, q, k: (q, k)
    layer.o_proj = lambda hidden: (hidden, None)

    def attention(query, key, value, batch):
        torch.testing.assert_close(query, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        torch.testing.assert_close(key, torch.tensor([[5.0, 6.0]]))
        torch.testing.assert_close(value, torch.tensor([[7.0, 8.0]]))
        return query

    layer.attn = attention
    output = layer(torch.tensor([0]), torch.zeros((1, 3)), None)
    torch.testing.assert_close(
        output,
        torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        * torch.sigmoid(torch.tensor([[0.0, 0.0, 1.0, 1.0]])),
    )
