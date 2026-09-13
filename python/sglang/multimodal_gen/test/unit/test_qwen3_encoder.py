from types import SimpleNamespace

import torch

import sglang.multimodal_gen.runtime.models.encoders.qwen3 as qwen3
import sglang.srt.layers.activation as srt_activation
from sglang.multimodal_gen.runtime.models.encoders.qwen3 import Qwen3ForCausalLM
from sglang.srt.layers.activation import SiluAndMul


class _CaptureLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.position_ids = None
        self.attention_lengths = None

    def forward(self, position_ids, hidden_states, residual, attention_lengths):
        self.position_ids = position_ids
        self.attention_lengths = attention_lengths
        if residual is None:
            residual = torch.zeros_like(hidden_states)
        return hidden_states, residual


class _IdentityNorm(torch.nn.Module):
    def forward(self, hidden_states, residual):
        if residual is not None:
            hidden_states = hidden_states + residual
        return hidden_states, None


def test_mlp_reuses_srt_activation_without_server_context(monkeypatch):
    def fail_get_exec():
        raise AssertionError("SiluAndMul must not read an unpublished context")

    monkeypatch.setattr(srt_activation, "publish_role", lambda: None)
    monkeypatch.setattr(srt_activation, "get_exec", fail_get_exec)

    def make_linear(*_args, **_kwargs):
        return torch.nn.Identity()

    monkeypatch.setattr(qwen3, "MergedColumnParallelLinear", make_linear)
    monkeypatch.setattr(qwen3, "RowParallelLinear", make_linear)

    mlp = qwen3.Qwen3MLP(16, 24, "silu")

    assert isinstance(mlp.act_fn, SiluAndMul)


def test_attention_keeps_diffusion_one_pass_qk_norm(monkeypatch):
    monkeypatch.setattr(qwen3, "get_tp_world_size", lambda: 1)
    monkeypatch.setattr(
        qwen3, "QKVParallelLinear", lambda **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(
        qwen3, "RowParallelLinear", lambda **kwargs: torch.nn.Identity()
    )
    monkeypatch.setattr(qwen3, "get_rope", lambda *args, **kwargs: torch.nn.Identity())
    monkeypatch.setattr(
        qwen3, "LocalAttention", lambda *args, **kwargs: torch.nn.Identity()
    )
    config = SimpleNamespace(
        head_dim=128,
        rms_norm_eps=1e-6,
        _supported_attention_backends=(),
    )

    attention = qwen3.Qwen3Attention(
        config,
        hidden_size=256,
        num_heads=2,
        num_kv_heads=1,
    )

    assert isinstance(attention.q_norm, qwen3.MMGenRMSNorm)
    assert isinstance(attention.k_norm, qwen3.MMGenRMSNorm)


def test_default_position_ids_batch_shape():
    model = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    layer = _CaptureLayer()
    model.config = SimpleNamespace(output_hidden_states=False)
    model.layers = torch.nn.ModuleList([layer])
    model.norm = _IdentityNorm()

    def get_input_embeddings(input_ids):
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)

    model.get_input_embeddings = get_input_embeddings

    input_ids = torch.zeros(2, 4, dtype=torch.long)
    attention_mask = torch.ones(2, 4, dtype=torch.long)

    model(input_ids=input_ids, attention_mask=attention_mask)

    assert layer.position_ids.shape == input_ids.shape
    assert torch.equal(layer.position_ids[0], torch.arange(4))
    assert torch.equal(layer.position_ids[1], torch.arange(4))
    assert layer.attention_lengths == (4, 4)


def test_fp8_qkv_scale_uses_the_packed_parameter_loader():
    model = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    layer = torch.nn.Module()
    layer.self_attn = torch.nn.Module()
    layer.self_attn.qkv_proj = torch.nn.Module()
    scale = torch.nn.Parameter(torch.zeros(3, 1), requires_grad=False)

    def load_scale(param, loaded_scale, shard_id):
        param.data[{"q": 0, "k": 1, "v": 2}[shard_id]].copy_(loaded_scale)

    scale.weight_loader = load_scale
    layer.self_attn.qkv_proj.register_parameter("weight_scale_inv", scale)
    model.layers = torch.nn.ModuleList([layer])
    model.config = SimpleNamespace(
        arch_config=SimpleNamespace(
            stacked_params_mapping=[(".qkv_proj", ".q_proj", "q")]
        )
    )

    loaded = model.load_weights(
        [("model.layers.0.self_attn.q_proj.weight_scale_inv", torch.tensor([2.0]))]
    )

    assert loaded == {"layers.0.self_attn.qkv_proj.weight_scale_inv"}
    torch.testing.assert_close(scale[:, 0], torch.tensor([2.0, 0.0, 0.0]))
