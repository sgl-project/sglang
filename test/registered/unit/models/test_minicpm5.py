from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sglang.srt.layers.aux_hidden_states import AuxHiddenStateList
from sglang.srt.layers.layer_boundary import PLAIN_ADD
from sglang.srt.layers.layer_boundary.residual import batch as residual_batch
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.model_executor.forward_mode import ForwardMode
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
from sglang.test.boundary_fixtures import make_test_stages, parallel_of, planning
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


@pytest.fixture
def dspark_target():
    model = MiniCPM5Model.__new__(MiniCPM5Model)
    nn.Module.__init__(model)
    model.layers_to_capture = []
    target = MiniCPM5ForCausalLM.__new__(MiniCPM5ForCausalLM)
    nn.Module.__init__(target)
    target.pp_group = SimpleNamespace(world_size=1, is_last_rank=True)
    target.model = model
    target.capture_aux_hidden_states = False
    return target


def configure_dspark_capture(target, layer_ids):
    configure_aux_hidden_state_capture(
        model=target,
        eagle_use_aux_hidden_state=False,
        eagle_aux_hidden_state_layer_ids=None,
        dflash_use_aux_hidden_state=True,
        dflash_target_layer_ids=layer_ids,
        is_dspark=True,
    )


def test_dspark_maps_to_the_next_attention_boundary(dspark_target):
    with pytest.raises(ValueError, match="explicit layer_ids"):
        configure_dspark_capture(dspark_target, None)
    configure_dspark_capture(dspark_target, [1, 7, 13, 19, 25])
    assert dspark_target.capture_aux_hidden_states
    assert dspark_target.model.layers_to_capture == [2, 8, 14, 20, 26]


@pytest.mark.parametrize("is_last_rank", [False, True], ids=["first", "last"])
def test_dspark_rejects_pipeline_parallel_capture(dspark_target, is_last_rank):
    """Earlier pipeline stages cannot forward their captured layers to the draft."""
    dspark_target.pp_group = SimpleNamespace(world_size=2, is_last_rank=is_last_rank)
    with pytest.raises(ValueError, match="DSpark.*pp_size=1"):
        configure_dspark_capture(dspark_target, [1, 7, 13, 19, 25])
    assert not dspark_target.capture_aux_hidden_states
    assert dspark_target.model.layers_to_capture == []


@pytest.mark.parametrize("layer_ids,tokens", [([27], 0), ([26, 27], 1)])
def test_dspark_captures_last_residual(layer_ids, tokens):
    model = MiniCPM5Model.__new__(MiniCPM5Model)
    nn.Module.__init__(model)
    model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    model.start_layer, model.end_layer = 26, 28
    model.norm = RMSNorm(2, force_native=True)

    def layer(positions, hidden, batch, capture_output):
        residual = residual_batch.fold(hidden, batch)
        if capture_output is not None:
            capture_output(residual)
        return residual_batch.stream_of(batch).record(
            torch.ones_like(residual), PLAIN_ADD
        )

    model.layers = {26: layer, 27: layer}
    target = SimpleNamespace(pp_group=model.pp_group, model=model)
    MiniCPM5ForCausalLM.set_dflash_layers_to_capture(target, layer_ids)
    embeds = torch.tensor([[2.0, 4.0]])[:tokens]
    normed, captured = model(None, None, SimpleNamespace(), input_embeds=embeds)
    assert len(captured) == len(layer_ids)
    for hidden, layer_id in zip(captured, layer_ids):
        torch.testing.assert_close(hidden, embeds + (layer_id - 25))
    if tokens:
        torch.testing.assert_close(normed, model.norm(embeds + 2))
        normed.zero_()
        torch.testing.assert_close(captured[-1], embeds + 2)


@pytest.mark.parametrize(
    "tied,tbo,lora,paths,message",
    [
        (True, False, False, None, "tie_word_embeddings=False"),
        (False, True, False, None, "--enable-two-batch-overlap"),
        (False, False, True, None, "LoRA"),
        (False, False, False, ["adapter"], "LoRA"),
    ],
)
def test_unsupported_options_fail_at_init(monkeypatch, tied, tbo, lora, paths, message):
    from sglang.srt.models import minicpm5

    monkeypatch.setattr(
        minicpm5,
        "get_exec",
        lambda: SimpleNamespace(overlap=SimpleNamespace(enable_two_batch_overlap=tbo)),
    )
    monkeypatch.setattr(
        minicpm5,
        "get_lora",
        lambda: SimpleNamespace(enable_lora=lora, lora_paths=paths),
    )
    with pytest.raises(ValueError, match=message):
        MiniCPM5ForCausalLM(SimpleNamespace(tie_word_embeddings=tied))


def test_decoder_retains_pre_norm_residual_before_attention():
    layer = MiniCPM5DecoderLayer.__new__(MiniCPM5DecoderLayer)
    nn.Module.__init__(layer)
    norm = RMSNorm(2, force_native=True)
    batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
    residual = torch.tensor([[2.0, 4.0]])
    update = torch.tensor([[1.0, -1.0]])
    expected = residual + update
    captured = AuxHiddenStateList()

    def attention(positions, hidden, batch):
        torch.testing.assert_close(hidden, norm(expected))
        # Reuse the boundary's residual buffer after capture.
        residual_batch.written_residual(batch).zero_()
        return hidden

    layer.self_attn = attention
    layer.mlp = lambda hidden, batch: hidden
    with planning(parallel_of(attn_dp=1, attn_tp=1)):
        stages = make_test_stages(attention_norm=norm, ffn_norm=norm, last=True)
        layer.attn_boundary, layer.ffn_boundary = stages.attn, stages.ffn
        residual_batch.set_written(residual, batch)
        hidden = residual_batch.stream_of(batch).record(update, PLAIN_ADD)
        layer(torch.arange(1), hidden, batch, capture_output=captured.capture)

    assert len(captured) == 1
    torch.testing.assert_close(captured[0], expected)


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


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
