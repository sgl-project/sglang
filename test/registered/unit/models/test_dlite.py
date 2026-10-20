import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

import sglang.kernels as kernels
from sglang.kernels.spec import KernelBackend
from sglang.srt.models.dlite import DLiteDraftModel
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from sglang.srt.runtime_context import SpawnRanks, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.parallel_groups import publish

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _config(architecture_version: str = "dlite_v1") -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        attention_bias=False,
    )
    config.architectures = ["DLiteDraftModel"]
    config.block_size = 4
    config.num_target_layers = 4
    config.dlite_config = {
        "architecture_version": architecture_version,
        "chs_num_layers": 2,
        "mask_token_id": 31,
        "model_role": "student",
        "sequential_head": "rnn",
        "sequential_rank": 8,
        "target_layer_ids": [0, 3],
    }
    return config


def _checkpoint_weights(config: Qwen3Config) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)

    def randn(*shape):
        return torch.randn(*shape, generator=generator) * 0.1

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    q_width = config.num_attention_heads * config.head_dim
    kv_width = config.num_key_value_heads * config.head_dim
    rank = config.dlite_config["sequential_rank"]
    weights = {
        "context_norm.weight": torch.rand(hidden, generator=generator) + 0.5,
        "layer_depth_embedding.weight": randn(config.num_target_layers, hidden),
        "layers.0.input_layernorm.weight": torch.rand(hidden, generator=generator)
        + 0.5,
        "layers.0.mlp.down_proj.weight": randn(hidden, intermediate),
        "layers.0.mlp.gate_proj.weight": randn(intermediate, hidden),
        "layers.0.mlp.up_proj.weight": randn(intermediate, hidden),
        "layers.0.post_attention_layernorm.weight": torch.rand(
            hidden, generator=generator
        )
        + 0.5,
        "layers.0.self_attn.k_norm.weight": torch.rand(
            config.head_dim, generator=generator
        )
        + 0.5,
        "layers.0.self_attn.k_proj.weight": randn(kv_width, hidden),
        "layers.0.self_attn.o_proj.weight": randn(hidden, q_width),
        "layers.0.self_attn.q_norm.weight": torch.rand(
            config.head_dim, generator=generator
        )
        + 0.5,
        "layers.0.self_attn.q_proj.weight": randn(q_width, hidden),
        "layers.0.self_attn.v_proj.weight": randn(kv_width, hidden),
        "norm.weight": torch.rand(hidden, generator=generator) + 0.5,
        "sequential_head.output_proj.weight": randn(config.vocab_size, rank),
        "sequential_head.prev_token_embedding.weight": randn(config.vocab_size, rank),
        "sequential_head.state_hidden_mlp.bias": randn(rank),
        "sequential_head.state_hidden_mlp.weight": randn(rank, rank + hidden),
        "sequential_head.state_proj.bias": randn(2 * rank),
        "sequential_head.state_proj.weight": randn(2 * rank, 2 * rank),
    }
    return weights


def _rms_norm(
    hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    variance = hidden_states.square().mean(dim=-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + eps) * weight


def _apply_neox_rope(
    states: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> torch.Tensor:
    cache = cos_sin_cache.index_select(0, positions.reshape(-1))
    cos, sin = cache.chunk(2, dim=-1)
    half_dim = states.shape[-1] // 2
    cos = cos.view(*positions.shape, 1, half_dim)
    sin = sin.view(*positions.shape, 1, half_dim)
    first, second = states.chunk(2, dim=-1)
    return torch.cat(
        [first * cos - second * sin, second * cos + first * sin],
        dim=-1,
    )


def _reference_forward(
    *,
    config: Qwen3Config,
    model: DLiteDraftModel,
    weights: dict[str, torch.Tensor],
    query_embeddings: torch.Tensor,
    target_hidden: torch.Tensor,
) -> torch.Tensor:
    eps = config.rms_norm_eps
    target_ids = torch.tensor(config.dlite_config["target_layer_ids"])
    context = _rms_norm(
        target_hidden + weights["layer_depth_embedding.weight"][target_ids],
        weights["context_norm.weight"],
        eps,
    )
    attention_input = _rms_norm(
        query_embeddings,
        weights["layers.0.input_layernorm.weight"],
        eps,
    )
    context_and_query = torch.cat([context, attention_input], dim=1)

    batch_size, query_length = query_embeddings.shape[:2]
    q = F.linear(attention_input, weights["layers.0.self_attn.q_proj.weight"]).view(
        batch_size, query_length, config.num_attention_heads, config.head_dim
    )
    k = F.linear(context_and_query, weights["layers.0.self_attn.k_proj.weight"]).view(
        batch_size,
        context_and_query.shape[1],
        config.num_key_value_heads,
        config.head_dim,
    )
    v = F.linear(
        context_and_query, weights["layers.0.self_attn.v_proj.weight"]
    ).view_as(k)
    q = _rms_norm(q, weights["layers.0.self_attn.q_norm.weight"], eps)
    k = _rms_norm(k, weights["layers.0.self_attn.k_norm.weight"], eps)

    query_position_offset = (
        0 if config.dlite_config["architecture_version"] == "dlite_v1" else 1
    )
    query_positions = (
        torch.arange(query_length).unsqueeze(0).expand(batch_size, -1)
        + query_position_offset
    )
    context_positions = torch.zeros((batch_size, context.shape[1]), dtype=torch.long)
    key_positions = torch.cat([context_positions, query_positions], dim=1)
    rope_cache = model.layers[0].self_attn.rotary_emb.cos_sin_cache
    q = _apply_neox_rope(q, query_positions, rope_cache)
    k = _apply_neox_rope(k, key_positions, rope_cache)
    groups = config.num_attention_heads // config.num_key_value_heads
    k = k.repeat_interleave(groups, dim=2)
    v = v.repeat_interleave(groups, dim=2)

    attention = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    )
    attention = attention.transpose(1, 2).reshape(batch_size, query_length, -1)
    hidden_states = query_embeddings + F.linear(
        attention, weights["layers.0.self_attn.o_proj.weight"]
    )
    mlp_input = _rms_norm(
        hidden_states,
        weights["layers.0.post_attention_layernorm.weight"],
        eps,
    )
    gate = F.linear(mlp_input, weights["layers.0.mlp.gate_proj.weight"])
    up = F.linear(mlp_input, weights["layers.0.mlp.up_proj.weight"])
    hidden_states = hidden_states + F.linear(
        F.silu(gate) * up, weights["layers.0.mlp.down_proj.weight"]
    )
    hidden_states = _rms_norm(hidden_states, weights["norm.weight"], eps)
    return hidden_states[:, -(config.block_size - 1) :, :]


def _reference_sequential_tokens(
    *,
    weights: dict[str, torch.Tensor],
    hidden_states: torch.Tensor,
    anchor_token_ids: torch.Tensor,
) -> torch.Tensor:
    rank = weights["sequential_head.state_hidden_mlp.bias"].numel()
    state = hidden_states.new_zeros(hidden_states.shape[0], rank)
    previous_ids = anchor_token_ids
    outputs = []
    for position in range(hidden_states.shape[1]):
        previous = F.embedding(
            previous_ids, weights["sequential_head.prev_token_embedding.weight"]
        )
        gate_raw, candidate_raw = F.linear(
            torch.cat([state, previous], dim=-1),
            weights["sequential_head.state_proj.weight"],
            weights["sequential_head.state_proj.bias"],
        ).chunk(2, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        latent = F.linear(
            torch.cat([state, hidden_states[:, position, :]], dim=-1),
            weights["sequential_head.state_hidden_mlp.weight"],
            weights["sequential_head.state_hidden_mlp.bias"],
        )
        previous_ids = F.linear(
            latent, weights["sequential_head.output_proj.weight"]
        ).argmax(dim=-1)
        outputs.append(previous_ids.unsqueeze(1))
    return torch.cat(outputs, dim=1)


class TestDLiteDraftModel(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        kernels.set_fused_op_backend(KernelBackend.TORCH)
        self.addCleanup(kernels.set_fused_op_backend, None)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=1),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )

    def test_checkpoint_loading_and_forward_match_reference(self):
        for architecture_version in ("dlite_v1", "dlite_v2"):
            with self.subTest(architecture_version=architecture_version):
                config = _config(architecture_version)
                model = DLiteDraftModel(config)
                weights = _checkpoint_weights(config)
                loaded = model.load_weights(weights.items())

                self.assertEqual(loaded, set(dict(model.named_parameters())))

                generator = torch.Generator().manual_seed(29)
                query_embeddings = torch.randn(
                    2, config.block_size, config.hidden_size, generator=generator
                )
                target_hidden = torch.randn(
                    2,
                    config.dlite_config["chs_num_layers"],
                    config.hidden_size,
                    generator=generator,
                )
                actual_hidden = model.forward_draft(
                    query_embeddings=query_embeddings,
                    target_hidden=target_hidden,
                )
                flat_hidden = model.forward_draft(
                    query_embeddings=query_embeddings,
                    target_hidden=target_hidden.flatten(1),
                )
                torch.testing.assert_close(flat_hidden, actual_hidden, rtol=0, atol=0)
                expected_hidden = _reference_forward(
                    config=config,
                    model=model,
                    weights=weights,
                    query_embeddings=query_embeddings,
                    target_hidden=target_hidden,
                )
                torch.testing.assert_close(
                    actual_hidden, expected_hidden, rtol=1e-5, atol=1e-6
                )

                anchor_token_ids = torch.tensor([3, 11])
                actual_tokens = model.sample_greedy(
                    draft_hidden=actual_hidden,
                    anchor_token_ids=anchor_token_ids,
                )
                expected_tokens = _reference_sequential_tokens(
                    weights=weights,
                    hidden_states=expected_hidden,
                    anchor_token_ids=anchor_token_ids,
                )
                torch.testing.assert_close(
                    actual_tokens, expected_tokens, rtol=0, atol=0
                )

    def test_qwen3_capture_uses_hf_layer_output_ids(self):
        target = SimpleNamespace(
            pp_group=SimpleNamespace(is_last_rank=True),
            config=SimpleNamespace(num_hidden_layers=4),
            capture_aux_hidden_states=False,
            model=SimpleNamespace(layers_to_capture=[]),
        )

        Qwen3ForCausalLM.set_dlite_layers_to_capture(target, [0, 3])

        self.assertTrue(target.capture_aux_hidden_states)
        self.assertEqual(target.model.layers_to_capture, [1, 4])


if __name__ == "__main__":
    unittest.main()
