"""CPU checks for AliceAI's AttnRes topology and checkpoint mappings."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.alice_ai import AliceAIConfig
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    speculative_moe_a2a_backend_context,
)
from sglang.srt.models import alice_ai, alice_ai_mtp
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Linear(nn.Module):
    def __init__(self, input_size, output_size, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size) * 0.2)

    def forward(self, value):
        return F.linear(value, self.weight), None


class _Norm(nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, value):
        return (
            value
            * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
            * self.weight
        )


class _Transform(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(config.hidden_size, config.hidden_size) * 0.1
        )

    def forward(self, value, *args, **kwargs):
        return F.linear(value, self.weight)


class TestAliceAI(CustomTestCase):
    def test_standard_ep_partitions_routed_and_shared_experts(self):
        """EP partitions routed experts while the shared MLP stays TP-sharded."""
        config = AliceAIConfig(
            hidden_size=128,
            vocab_size=256,
            num_hidden_layers=1,
            layer_types=["full_attention"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
            mtp_num_hidden_layers=1,
        )
        for tp_size in (2, 4):
            for model_class in (
                alice_ai.AliceAIForCausalLM,
                alice_ai_mtp.AliceAIForCausalLMMTP,
            ):
                with (
                    self.subTest(tp_size=tp_size, model=model_class.__name__),
                    get_context().override_server_args(tp_size=tp_size, ep_size=2),
                    get_parallel().override(
                        tp_rank=0,
                        attn_tp_rank=0,
                        moe_ep_rank=0,
                        moe_tp_rank=0,
                        # Construction only: no collectives or GPU allocations.
                        tp_group=SimpleNamespace(world_size=tp_size, cpu_group=None),
                    ),
                    get_flags().moe.override(
                        a2a_backend=MoeA2ABackend.NONE,
                        runner_backend=MoeRunnerBackend.TRITON,
                    ),
                    # Skip optional RoPE kernel setup; this checks MoE partitioning.
                    patch.object(alice_ai, "get_rope", return_value=nn.Identity()),
                    torch.device("meta"),
                ):
                    model = model_class(config)
                    moe = model.model.layers[0].mlp
                    self.assertEqual(moe.experts.num_local_experts, 2)
                    self.assertEqual(
                        moe.experts.w13_weight.shape,
                        (2, 2 * config.moe_intermediate_size // (tp_size // 2), 128),
                    )
                    self.assertEqual(
                        moe.shared_expert.down_proj.weight.shape,
                        (128, config.shared_expert_intermediate_size // tp_size),
                    )

    def test_unsupported_runtime_modes_are_rejected(self):
        config = AliceAIConfig()
        for options, error in (
            ({"enable_eplb": True}, "expert-location trivial"),
            ({"ep_num_redundant_experts": 2}, "expert-location trivial"),
            ({"init_expert_location": "random"}, "expert-location trivial"),
            ({"cpu_offload_gb": 1}, "does not support --cpu-offload-gb"),
        ):
            with (
                self.subTest(options=options),
                get_context().override_server_args(tp_size=2, ep_size=2, **options),
                get_flags().moe.override(a2a_backend=MoeA2ABackend.NONE),
                self.assertRaisesRegex(NotImplementedError, error),
            ):
                alice_ai.AliceAISparseMoeBlock(config, 0, None, "model.layers.0.mlp")

        with (
            get_context().override_server_args(tp_size=2, ep_size=2),
            get_flags().moe.override(
                a2a_backend=MoeA2ABackend.NONE,
                speculative_a2a_backend=MoeA2ABackend.DEEPEP,
            ),
            speculative_moe_a2a_backend_context(),
            self.assertRaisesRegex(NotImplementedError, "MoE A2A backend 'none'"),
        ):
            alice_ai.AliceAISparseMoeBlock(config, 0, None, "model.layers.0.mlp")

    def test_shared_expert_uses_input_before_routed_expert_overwrites_it(self):
        """An in-place routed expert must not change the shared expert's input."""
        block = alice_ai.AliceAISparseMoeBlock.__new__(alice_ai.AliceAISparseMoeBlock)
        nn.Module.__init__(block)
        block.tp_size = 1
        block.alt_stream = None
        block.gate = lambda value: (value.new_zeros((value.shape[0], 1)), None)
        block.topk = lambda value, logits: None

        def inplace_experts(value, topk_output):
            value.mul_(2).add_(3)
            return value

        block.experts = inplace_experts
        block.shared_expert = lambda value: value.square()
        block.shared_expert_gate = lambda value: (value.sum(-1, keepdim=True), None)
        hidden = torch.tensor([[-0.5, 1.25], [2.0, -1.0]])
        original = hidden.clone()
        routed = 2 * original + 3
        expected = routed + original.square() * original.sum(-1, keepdim=True).sigmoid()

        actual = block(hidden)

        torch.testing.assert_close(actual, expected)
        # Combining the outputs must leave the expert's aliased routed result intact.
        torch.testing.assert_close(hidden, routed)

    def test_attnres_matches_explicit_block_reference(self):
        # Compare to a list-of-sources implementation with learned, nonuniform
        # scores. Cover completed blocks, a partial tail, and both attention kinds.
        for block_size in (1, 2, 4):
            with self.subTest(block_size=block_size):
                torch.manual_seed(7)
                config = AliceAIConfig(
                    hidden_size=4,
                    num_hidden_layers=5,
                    layer_types=[
                        "linear_attention",
                        "full_attention",
                        "linear_attention",
                        "full_attention",
                        "linear_attention",
                    ],
                    block_attn_res_block_size=block_size,
                )
                with (
                    patch.object(alice_ai, "ReplicatedLinear", _Linear),
                    patch.object(alice_ai, "RMSNorm", _Norm),
                    patch.object(alice_ai, "GemmaRMSNorm", _Norm),
                    patch.object(alice_ai, "KimiDeltaAttention", _Transform),
                    patch.object(
                        alice_ai,
                        "AliceAIFullAttention",
                        lambda config, *args, **kw: _Transform(config),
                    ),
                    patch.object(
                        alice_ai,
                        "AliceAISparseMoeBlock",
                        lambda config, *args, **kw: _Transform(config),
                    ),
                    patch.object(
                        alice_ai,
                        "VocabParallelEmbedding",
                        lambda n, h, **kw: nn.Embedding(n, h),
                    ),
                ):
                    model = alice_ai.AliceAIModel(config, None, "model")
                embeddings = torch.randn(3, config.hidden_size)
                batch = SimpleNamespace(
                    forward_mode=SimpleNamespace(is_idle=lambda: False)
                )
                recorder = SimpleNamespace(with_current_layer=lambda _: nullcontext())
                with (
                    patch.object(
                        alice_ai, "aggregate_stream", alice_ai.aggregate_stream_torch
                    ),
                    patch.object(
                        alice_ai,
                        "get_global_expert_distribution_recorder",
                        return_value=recorder,
                    ),
                ):
                    actual = model(None, torch.arange(3), batch, embeddings)

                def mix(sources, projection, norm):
                    stacked = torch.stack(sources, dim=1)
                    scores = F.linear(norm(stacked), projection.weight).squeeze(-1)
                    return (scores.softmax(-1).unsqueeze(-1) * stacked).sum(1)

                sources, partial = [embeddings], None
                for i, layer in enumerate(model.layers):
                    if i and i % block_size == 0:
                        sources.append(partial)
                        partial = None
                    hidden = (
                        embeddings
                        if i == 0
                        else mix(
                            sources + ([partial] if partial is not None else []),
                            layer.attn_res_proj,
                            layer.attn_res_norm,
                        )
                    )
                    attention = (
                        layer.linear_attn
                        if layer.layer_type == "linear_attention"
                        else layer.self_attn
                    )
                    attention_output = F.linear(
                        layer.input_layernorm(hidden), attention.weight
                    )
                    partial = (
                        attention_output
                        if partial is None
                        else partial + attention_output
                    )
                    hidden = mix(
                        sources + [partial], layer.mlp_res_proj, layer.mlp_res_norm
                    )
                    partial = partial + F.linear(
                        layer.post_attention_layernorm(hidden), layer.mlp.weight
                    )
                final = model.attnres_final
                expected = model.norm(
                    mix(sources + [partial], final.res_proj, final.res_norm)
                )
                torch.testing.assert_close(actual, expected)

    def test_full_attention_query_gate_is_interleaved_by_head(self):
        attention = alice_ai.AliceAIFullAttention.__new__(alice_ai.AliceAIFullAttention)
        nn.Module.__init__(attention)
        attention.head_dim, attention.num_heads = 2, 2
        attention.q_size, attention.kv_size = 4, 2
        packed = torch.tensor(
            [[1.0, 2.0, -1.0, 1.0, 3.0, 4.0, 0.5, -0.5, 5.0, 6.0, 7.0, 8.0]]
        )
        attention.qkv_proj = lambda _: (packed, None)
        attention.q_norm = attention.k_norm = nn.Identity()
        attention.rotary_emb = lambda positions, q, k: (q, k)
        attention.attn = lambda q, k, v, batch: q
        attention.o_proj = lambda value: (value, None)
        output = attention(torch.zeros(1, 4), torch.tensor([0]), None)
        expected = (
            torch.tensor([[1.0, 2.0, 3.0, 4.0]])
            * torch.tensor([[-1.0, 1.0, 0.5, -0.5]]).sigmoid()
        )
        torch.testing.assert_close(output, expected)

    def test_checkpoint_fusion_and_mtp_filtering(self):
        model = alice_ai.AliceAIForCausalLM.__new__(alice_ai.AliceAIForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=1, tie_word_embeddings=False)
        model.model = nn.Module()
        layer = nn.Module()
        layer.linear_attn = nn.Module()
        fused_a = layer.linear_attn.fused_qkvbfg_a_proj = nn.Module()
        fused_a.weight = nn.Parameter(torch.empty(6, 1))
        fused_b = layer.linear_attn.fused_fg_b_proj = nn.Module()
        fused_b.weight = nn.Parameter(torch.empty(2, 1))

        def load_projection(param, weight, shard_id):
            param.data[shard_id].copy_(weight)

        fused_a.weight.weight_loader = fused_b.weight.weight_loader = load_projection
        conv = layer.linear_attn.qkv_conv1d = nn.Module()
        conv.weight = nn.Parameter(torch.empty(6, 1, 3))

        def load_conv(param, weight, shard_id):
            param.data[2 * shard_id : 2 * (shard_id + 1)].copy_(weight)

        conv.weight.weight_loader = load_conv
        layer.linear_attn.A_log = nn.Parameter(torch.empty(1, 1, 2, 1))
        layer.mlp = nn.Module()
        layer.mlp.experts = nn.Module()
        experts = layer.mlp.experts
        experts.w13_weight = nn.Parameter(torch.empty(1, 4, 2))
        experts.w2_weight = nn.Parameter(torch.empty(1, 2, 2))

        def load_expert(param, weight, name, expert_id, shard_id):
            if shard_id == "w2":
                param.data[expert_id].copy_(weight)
            else:
                offset = 0 if shard_id == "w1" else 2
                param.data[expert_id, offset : offset + 2].copy_(weight)

        experts.w13_weight.weight_loader = experts.w2_weight.weight_loader = load_expert
        model.model.layers = nn.ModuleList([layer])
        weights = [
            (
                f"model.layers.0.linear_attn.{kind}_conv1d.weight",
                torch.full((2, 1, 3), float(i)),
            )
            for i, kind in enumerate(("q", "k", "v"), 1)
        ]
        weights += [
            (f"model.layers.0.linear_attn.{kind}_proj.weight", torch.tensor([float(i)]))
            for i, kind in enumerate(("q", "k", "v", "b", "f_a", "g_a"), 1)
        ]
        weights += [
            ("model.layers.0.linear_attn.f_b_proj.weight", torch.tensor([8.0])),
            ("model.layers.0.linear_attn.g_b_proj.weight", torch.tensor([9.0])),
            ("model.layers.0.linear_attn.a_log_bias", torch.tensor([3.0, 4.0])),
            ("model.layers.0.mlp.experts.0.gate_proj.weight", torch.full((2, 2), 5.0)),
            ("model.layers.0.mlp.experts.0.down_proj.weight", torch.full((2, 2), 6.0)),
            ("model.layers.0.mlp.experts.0.up_proj.weight", torch.full((2, 2), 7.0)),
            ("mtp.unused.weight", torch.ones(1)),
        ]
        loaded = model.load_weights(weights)
        self.assertEqual(loaded, set(dict(model.named_parameters())))
        torch.testing.assert_close(fused_a.weight[:, 0], torch.arange(1.0, 7.0))
        torch.testing.assert_close(fused_b.weight[:, 0], torch.tensor([8.0, 9.0]))
        torch.testing.assert_close(
            conv.weight[:, 0, 0], torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        )
        torch.testing.assert_close(
            layer.linear_attn.A_log.flatten(), torch.tensor([3.0, 4.0])
        )
        torch.testing.assert_close(
            experts.w13_weight[0, :, 0], torch.tensor([5.0, 5.0, 7.0, 7.0])
        )
        torch.testing.assert_close(experts.w2_weight, torch.full((1, 2, 2), 6.0))
        with self.assertRaisesRegex(ValueError, "missing weights"):
            model.load_weights(weights[:3])

    def test_packed_checkpoint_experts_preserve_gate_up_order_and_expert_ids(self):
        # The published HF checkpoint stores all experts in two bare parameters,
        # rather than one gate/up/down weight per expert.
        model = alice_ai.AliceAIForCausalLM.__new__(alice_ai.AliceAIForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=3, tie_word_embeddings=False)
        model.model = nn.Module()
        layer = nn.Module()
        layer.mlp = nn.Module()
        experts = layer.mlp.experts = nn.Module()
        experts.w13_weight = nn.Parameter(torch.full((3, 4, 2), float("nan")))
        experts.w2_weight = nn.Parameter(torch.full((3, 2, 2), float("nan")))

        def load_expert(param, weight, name, expert_id, shard_id):
            if shard_id == "w2":
                param.data[expert_id].copy_(weight)
            else:
                offset = 0 if shard_id == "w1" else 2
                param.data[expert_id, offset : offset + 2].copy_(weight)

        experts.w13_weight.weight_loader = experts.w2_weight.weight_loader = load_expert
        model.model.layers = nn.ModuleList([layer])
        gate_up = torch.arange(24, dtype=torch.float32).reshape(3, 4, 2) / 20
        down = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 10
        weights = [
            ("model.layers.0.mlp.experts.gate_up_proj", gate_up),
            ("model.layers.0.mlp.experts.down_proj", down),
        ]
        loaded = model.load_weights(weights)
        self.assertEqual(loaded, set(dict(model.named_parameters())))
        torch.testing.assert_close(experts.w13_weight, gate_up)
        torch.testing.assert_close(experts.w2_weight, down)


if __name__ == "__main__":
    unittest.main()
