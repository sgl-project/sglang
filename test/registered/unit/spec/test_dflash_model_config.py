"""MiniMax checkpoints use the generic DFlash2 layers with different math."""

import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.srt.models import dflash
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def single_rank_native():
    # Exercise the real layers without process groups or device-specific kernels.
    previous_backend = get_fused_op_backend()
    set_fused_op_backend(KernelBackend.TORCH)
    try:
        with (
            get_context().override_server_args(),
            get_parallel().override(tp_size=1, tp_rank=0, tp_group=None),
            # CPU runners have no CUDA rope kernel; keep RoPE on the native path.
            patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True),
        ):
            yield
    finally:
        set_fused_op_backend(previous_backend)


def _config(**overrides):
    fields = dict(
        hidden_size=8,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        vocab_size=16,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=32,
        rope_theta=10000.0,
        partial_rotary_factor=0.5,
        dflash_config=dict(
            block_size=4,
            selector_rank=2,
            selector_top_k=2,
            target_layer_ids=[0, 2],
            conv_kernel_size=2,
            conv_group_size=2,
        ),
    )
    fields.update(overrides)
    return PretrainedConfig(**fields)


def _load_norm(norm):
    weight = torch.linspace(-0.4, 0.6, norm.weight.numel())
    loader = getattr(norm.weight, "weight_loader", dflash.default_weight_loader)
    loader(norm.weight, weight)


def _reference_norm(x, norm, gemma):
    scale = norm.weight.float() + (1.0 if gemma else 0.0)
    normalized = x.float() * torch.rsqrt(
        x.float().square().mean(dim=-1, keepdim=True) + norm.variance_epsilon
    )
    return (normalized * scale).to(x.dtype)


@pytest.mark.parametrize("gemma", [False, True])
def test_every_backbone_norm_uses_checkpoint_convention(gemma):
    # Omit the new flag entirely for the existing Qwen-style default.
    model = dflash.DFlash2DraftModel(
        _config(
            partial_rotary_factor=1.0,
            **({"use_gemma_norm": True} if gemma else {}),
        )
    )
    norms = [model.norm, model.hidden_norm]
    for layer in model.layers:
        norms.extend(
            [
                layer.input_layernorm,
                layer.post_attention_layernorm,
                layer.self_attn.q_norm,
                layer.self_attn.k_norm,
            ]
        )
    for norm in norms:
        assert type(norm) is (GemmaRMSNorm if gemma else RMSNorm)
        _load_norm(norm)
        x = torch.arange(1, 3 * norm.weight.numel() + 1, dtype=torch.float32).reshape(
            3, -1
        )
        torch.testing.assert_close(norm(x), _reference_norm(x, norm, gemma))
    assert model.supports_fused_context_kv is (not gemma)
    # Full RoPE would otherwise admit the RMSNorm-only fused table path.
    assert all(
        layer.self_attn.use_table_qk_norm_rope is (not gemma) for layer in model.layers
    )


@pytest.mark.parametrize("gemma", [False, True])
def test_context_projection_uses_the_same_norm_convention(gemma):
    model = dflash.DFlash2DraftModel(_config(use_gemma_norm=gemma))
    _load_norm(model.hidden_norm)
    with torch.no_grad():
        model.fc.weight.copy_(torch.arange(128).reshape(8, 16) / 128)
    context = torch.linspace(-2, 3, 48).reshape(3, 16)
    expected = _reference_norm(
        F.linear(context, model.fc.weight), model.hidden_norm, gemma
    )
    torch.testing.assert_close(model.project_target_hidden(context), expected)


@pytest.mark.parametrize(
    "activation,alpha,limit",
    [("silu", 1.0, 7.0), ("swigluoai", 1.702, 7.0), ("swigluoai", 0.75, 2.5)],
)
def test_mlp_matches_activation_formula_including_saturated_inputs(
    activation, alpha, limit
):
    mlp = dflash.DFlashMLP(
        _config(hidden_act=activation, swiglu_alpha=alpha, swiglu_limit=limit)
    )
    with torch.no_grad():
        mlp.gate_up_proj.weight.copy_(torch.eye(8))
        mlp.down_proj.weight.copy_(torch.arange(32).reshape(8, 4) / 32)
    x = torch.tensor(
        [
            [-12.0, -2.0, 2.0, 12.0, -12.0, -1.0, 1.0, 12.0],
            [12.0, 0.2, -0.2, -12.0, 12.0, -12.0, 12.0, -12.0],
        ]
    )
    gate, up = x.chunk(2, dim=-1)
    if activation == "swigluoai":
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        activated = gate * torch.sigmoid(alpha * gate) * (up + 1)
    else:
        activated = F.silu(gate) * up
    expected = F.linear(activated, mlp.down_proj.weight)
    torch.testing.assert_close(mlp(x), expected)


def _reference_partial_rope(x, positions, head_dim=4):
    # This config rotates the first two coordinates of each four-wide head.
    heads = x.reshape(x.shape[0], -1, head_dim)
    first, second = heads[..., 0], heads[..., 1]
    angle = positions.float()[:, None]
    rotated = torch.stack(
        (
            first * angle.cos() - second * angle.sin(),
            second * angle.cos() + first * angle.sin(),
        ),
        dim=-1,
    )
    return torch.cat((rotated, heads[..., 2:]), dim=-1).reshape_as(x)


def test_partial_rope_preserves_the_unrotated_half_of_each_head():
    attention = dflash.DFlashAttention(_config(use_gemma_norm=True), layer_id=0)
    assert attention.rotary_emb.rotary_dim == 2
    positions = torch.tensor([1, 3])
    q = torch.arange(1, 17, dtype=torch.float32).reshape(2, 8)
    k = torch.arange(1, 9, dtype=torch.float32).reshape(2, 4)
    actual_q, actual_k = attention.rotary_emb(positions, q.clone(), k.clone())
    torch.testing.assert_close(actual_q, _reference_partial_rope(q, positions))
    torch.testing.assert_close(actual_k, _reference_partial_rope(k, positions))


def test_gemma_attention_and_npu_context_preparation_share_correct_qk_math(monkeypatch):
    attention = dflash.DFlashAttention(_config(use_gemma_norm=True), layer_id=0)
    _load_norm(attention.q_norm)
    _load_norm(attention.k_norm)
    with torch.no_grad():
        attention.qkv_proj.weight.copy_(torch.arange(128).reshape(16, 8) / 128)
        attention.o_proj.weight.copy_(torch.eye(8))
    hidden = torch.linspace(-1, 2, 16).reshape(2, 8)
    positions = torch.tensor([1, 3])
    q, k, v = F.linear(hidden, attention.qkv_proj.weight).split([8, 4, 4], dim=-1)
    expected_q = _reference_partial_rope(
        _reference_norm(q.reshape(-1, 4), attention.q_norm, True).reshape_as(q),
        positions,
    )
    expected_k = _reference_partial_rope(
        _reference_norm(k, attention.k_norm, True), positions
    )
    actual_q, actual_k, actual_v = attention.forward_prepare_npu(positions, hidden)
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, v)
    torch.testing.assert_close(
        attention.apply_k_norm(k), _reference_norm(k, attention.k_norm, True)
    )

    # Only replace the serving attention backend; normalization, RoPE and all
    # projections run through the actual DFlashAttention.forward implementation.
    def capture_qkv(actual_q, actual_k, actual_v, forward_batch):
        torch.testing.assert_close(actual_q, expected_q)
        torch.testing.assert_close(actual_k, expected_k)
        torch.testing.assert_close(actual_v, v)
        return actual_q

    monkeypatch.setattr(attention.attn, "forward", capture_qkv)
    torch.testing.assert_close(attention(positions, hidden, None), expected_q)


def test_legacy_architecture_name_loads_the_generic_dflash2_state():
    config = _config(
        use_gemma_norm=True,
        hidden_act="swigluoai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
    )
    generic = dflash.DFlash2DraftModel(config)
    legacy = dflash.MiniMaxDFlash2DraftModel(config)
    with torch.no_grad():
        for parameter in generic.parameters():
            parameter.copy_(
                torch.linspace(-0.2, 0.3, parameter.numel()).reshape_as(parameter)
            )
    # Serving exports store separate Q/K/V and gate/up projections. Exercise
    # their real weight loaders, including Gemma's offset-weight buffer update.
    checkpoint = []
    for name, weight in generic.named_parameters():
        if ".qkv_proj." in name:
            for projection, shard in zip(("q", "k", "v"), weight.split([8, 4, 4])):
                checkpoint.append(
                    (name.replace("qkv_proj", projection + "_proj"), shard)
                )
        elif ".gate_up_proj." in name:
            for projection, shard in zip(("gate", "up"), weight.chunk(2)):
                checkpoint.append(
                    (name.replace("gate_up_proj", projection + "_proj"), shard)
                )
        else:
            checkpoint.append((name, weight))
    legacy.load_weights(checkpoint)
    for name, weight in legacy.named_parameters():
        torch.testing.assert_close(weight, dict(generic.named_parameters())[name])
    context = torch.linspace(-1, 1, 48).reshape(3, 16)
    hidden = torch.linspace(-1, 1, 24).reshape(3, 8)
    torch.testing.assert_close(
        legacy.project_target_hidden(context), generic.project_target_hidden(context)
    )
    torch.testing.assert_close(
        legacy.layers[0].mlp(hidden), generic.layers[0].mlp(hidden)
    )
    assert dflash.MiniMaxDFlash2DraftModel in dflash.EntryClass


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
