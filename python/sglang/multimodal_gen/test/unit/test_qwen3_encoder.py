from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RotaryEmbedding,
    apply_rotary_pos_emb,
)

import sglang.multimodal_gen.runtime.models.encoders.qwen3 as qwen3
import sglang.srt.layers.activation as srt_activation
from sglang.multimodal_gen.configs.models.encoders.qwen3 import Qwen3TextConfig
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders.qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3HfRowParallelLinear,
)
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


def test_default_position_ids_batch_shape():
    model = Qwen3ForCausalLM.__new__(Qwen3ForCausalLM)
    torch.nn.Module.__init__(model)
    layer = _CaptureLayer()
    model.config = SimpleNamespace(output_hidden_states=False)
    model.preserve_hf_numerics = False
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


def _hf_numerics_config(reference_config):
    config = Qwen3TextConfig(preserve_hf_numerics=True)
    config.update_model_arch(reference_config.to_dict())
    return config


def _tp_group(rank, world_size=2):
    group = SimpleNamespace(world_size=world_size, rank_in_group=rank)
    return group, patch(
        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group",
        return_value=group,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("case", ["padded", "all_valid", "implicit_causal"])
@torch.no_grad()
def test_hf_numerics_match_transformers_qwen3(
    single_process_model_parallel, dtype, case
):
    """preserve_hf_numerics must reproduce transformers' Qwen3Model, including
    learned norm scales in BF16, padded prompts, and an omitted mask."""
    torch.manual_seed(17)
    reference_config = Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
        attention_bias=False,
        rope_theta=1000000.0,
    )
    reference_config._attn_implementation = "sdpa"
    reference = Qwen3Model(reference_config).cuda().to(dtype).eval()
    # Learned norm scales expose casting before or after the scale multiply.
    for name, parameter in reference.named_parameters():
        if name.endswith("norm.weight"):
            parameter.uniform_(0.5, 1.5)
    native = Qwen3ForCausalLM(_hf_numerics_config(reference_config))
    native = native.cuda().to(dtype).eval()
    native.load_weights(reference.state_dict().items())

    length = 33 if case == "padded" else 65
    ids = torch.randint(1, 128, (2, length), device="cuda")
    mask = None if case == "implicit_causal" else torch.ones_like(ids)
    if case == "padded":
        mask[0, 31:] = 0
        mask[1, 28:] = 0
        ids[mask == 0] = 0
    with set_forward_context(current_timestep=0, attn_metadata=None):
        actual = native(ids, attention_mask=mask).last_hidden_state
    expected = reference(ids, attention_mask=mask, use_cache=False).last_hidden_state
    if mask is not None:
        actual = actual * mask.unsqueeze(-1)
        expected = expected * mask.unsqueeze(-1)
    atol, rtol = (1e-4, 1e-4) if dtype == torch.float32 else (0.05, 0.02)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("input_is_parallel", [False, True])
@pytest.mark.parametrize("skip_bias_add", [False, True])
@torch.no_grad()
def test_hf_tp_row_projection_matches_full_gemm(rank, input_is_parallel, skip_bias_add):
    """TP shards must reconstruct the HF projection without split-K rounding;
    the summands below lose the small terms if partials are rounded to BF16."""
    inputs = torch.tensor([[256, 1, -256, 0], [512, 2, -512, 0]], dtype=torch.bfloat16)
    weight = torch.tensor([[1, 1, 1, 1], [2, 1, 2, 1]], dtype=torch.bfloat16)
    bias = torch.tensor([0.5, -0.5], dtype=torch.bfloat16)
    group, group_patch = _tp_group(rank)
    with group_patch, set_default_torch_dtype(torch.bfloat16):
        layer = Qwen3HfRowParallelLinear(
            4, 2, input_is_parallel=input_is_parallel, skip_bias_add=skip_bias_add
        )
    layer.weight.weight_loader(layer.weight, weight)
    layer.bias.weight_loader(layer.bias, bias)
    collective_inputs = []

    def gather_shards(value, dim=-1, tp_group=None):
        assert tp_group is group
        complete = inputs if dim == -1 else weight
        torch.testing.assert_close(
            value, complete[:, rank * 2 : (rank + 1) * 2], atol=0, rtol=0
        )
        collective_inputs.append(value.clone())
        return complete.clone()

    local_input = inputs[:, rank * 2 : (rank + 1) * 2] if input_is_parallel else inputs
    with (
        patch.object(qwen3, "tensor_model_parallel_all_gather", gather_shards),
        torch.autocast("cpu", dtype=torch.bfloat16),
    ):
        actual, output_bias = layer(local_input)

    expected = F.linear(inputs, weight, None if skip_bias_add else bias)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(x.dtype == torch.bfloat16 for x in collective_inputs)
    if skip_bias_add:
        torch.testing.assert_close(output_bias, bias, atol=0, rtol=0)
    else:
        assert output_bias is None


@pytest.mark.parametrize("rank", [0, 1])
@torch.no_grad()
def test_hf_tp_row_projection_keeps_unreduced_partials(rank):
    """reduce_results=False callers keep the plain row-parallel contract."""
    inputs = torch.tensor([[256, 1, -256, 0]], dtype=torch.bfloat16)
    weight = torch.tensor([[1, 1, 1, 1]], dtype=torch.bfloat16)
    _, group_patch = _tp_group(rank)
    with group_patch, set_default_torch_dtype(torch.bfloat16):
        layer = Qwen3HfRowParallelLinear(4, 1, bias=False, reduce_results=False)
    layer.weight.weight_loader(layer.weight, weight)
    local = inputs[:, rank * 2 : (rank + 1) * 2]

    actual, bias = layer(local)

    expected = F.linear(local, weight[:, rank * 2 : (rank + 1) * 2])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert bias is None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_hf_rope_retains_linear_scaling(dtype):
    """The HF RoPE path must keep linear rope scaling and HF's rounding order."""
    reference_config = Qwen3Config(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_parameters={
            "rope_type": "linear",
            "rope_theta": 1000000.0,
            "factor": 2.0,
        },
    )
    _, group_patch = _tp_group(0, world_size=1)
    with (
        group_patch,
        patch.object(qwen3, "get_tp_world_size", return_value=1),
        patch.object(qwen3, "LocalAttention", return_value=torch.nn.Identity()),
    ):
        native = Qwen3Attention(
            _hf_numerics_config(reference_config),
            hidden_size=32,
            num_heads=4,
            num_kv_heads=2,
            rope_theta=1000000.0,
            rope_scaling=reference_config.rope_parameters,
        )
    reference = Qwen3RotaryEmbedding(reference_config)
    positions = torch.arange(17).expand(2, -1)
    torch.manual_seed(123)
    query = torch.randn(2, 17, 32).to(dtype)
    key = torch.randn(2, 17, 16).to(dtype)
    cos, sin = reference(query, positions)
    expected_query, expected_key = apply_rotary_pos_emb(
        query.unflatten(-1, (4, 8)).transpose(1, 2),
        key.unflatten(-1, (2, 8)).transpose(1, 2),
        cos,
        sin,
    )

    actual_query, actual_key = native._apply_hf_rope(positions, query, key)

    torch.testing.assert_close(
        actual_query, expected_query.transpose(1, 2).flatten(2), atol=0, rtol=0
    )
    torch.testing.assert_close(
        actual_key, expected_key.transpose(1, 2).flatten(2), atol=0, rtol=0
    )
