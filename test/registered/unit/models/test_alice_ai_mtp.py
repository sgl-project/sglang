"""AliceAI MTP math, published checkpoint layout, and draft isolation."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.kernels.spec import KernelBackend
from sglang.srt.arg_groups.overrides import model_config_of, resolved_view
from sglang.srt.configs.alice_ai import AliceAIConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.model_executor.model_runner_components.spec_aux_hidden_state import (
    resolve_spec_aux_hidden_state_config,
)
from sglang.srt.models import alice_ai_mtp
from sglang.srt.runtime_context import get_context, override_platform
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Attention(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))

    def forward(self, hidden_states, positions, forward_batch):
        return F.linear(hidden_states, self.weight)


class _Moe(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))

    def forward(self, hidden_states):
        return F.linear(hidden_states, self.weight).tanh()


class TestAliceAIMTP(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(patch.dict(os.environ))
        previous = get_fused_op_backend()
        set_fused_op_backend(KernelBackend.TORCH)
        self.addCleanup(set_fused_op_backend, previous)

    def test_post_target_norm_states_and_embeddings_follow_checkpoint_math(self):
        """The MTP head mixes embedding first, then normalized target states,
        and uses ordinary attention/MLP residuals rather than target AttnRes.
        """
        torch.manual_seed(17)
        config = AliceAIConfig(hidden_size=4, vocab_size=8, mtp_num_hidden_layers=1)
        with (
            patch.object(
                alice_ai_mtp,
                "VocabParallelEmbedding",
                lambda n, h, **kwargs: nn.Embedding(n, h),
            ),
            patch.object(alice_ai_mtp, "AliceAIFullAttention", _Attention),
            patch.object(alice_ai_mtp, "AliceAISparseMoeBlock", _Moe),
            patch.object(alice_ai_mtp, "is_cuda", return_value=False),
        ):
            model = alice_ai_mtp.AliceAIMultiTokenPredictor(config, "model")
        layer = model.layers[0]
        norms = (
            model.pre_fc_norm_embedding,
            model.pre_fc_norm_hidden,
            layer.input_layernorm,
            layer.post_attention_layernorm,
            model.norm,
        )
        for index, norm in enumerate(norms):
            norm.weight.weight_loader(
                norm.weight, torch.linspace(-0.4, 0.6, 4) + 0.1 * index
            )

        def norm(value, weight):
            value = value.float()
            return (
                value
                * torch.rsqrt(
                    value.square().mean(-1, keepdim=True) + config.rms_norm_eps
                )
                * (1 + weight)
            )

        # EAGLE supplies the final normalized target stream, not the raw
        # target residual before its final norm.
        raw_target = torch.tensor(
            [[0.3, -0.7, 0.2, 1.1], [2.0, 0.1, -0.2, 0.9], [-0.6, 0.2, 1.4, -0.3]]
        )
        target_states = norm(raw_target, torch.tensor([0.1, -0.2, 0.3, 0.4]))
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            spec_info=SimpleNamespace(hidden_states=target_states),
        )
        ids, positions = torch.tensor([1, 3, 5]), torch.arange(3)
        for input_embeds in (None, torch.randn(3, 4)):
            with self.subTest(provided_embeddings=input_embeds is not None):
                embeddings = (
                    model.embed_tokens(ids) if input_embeds is None else input_embeds
                )
                expected = F.linear(
                    torch.cat(
                        (
                            norm(embeddings, model.pre_fc_norm_embedding.weight),
                            norm(target_states, model.pre_fc_norm_hidden.weight),
                        ),
                        dim=-1,
                    ),
                    model.fc.weight,
                )
                expected = expected + F.linear(
                    norm(expected, layer.input_layernorm.weight), layer.self_attn.weight
                )
                expected = (
                    expected
                    + F.linear(
                        norm(expected, layer.post_attention_layernorm.weight),
                        layer.mlp.weight,
                    ).tanh()
                )
                expected = norm(expected, model.norm.weight)
                actual = model(ids, positions, batch, input_embeds)
                torch.testing.assert_close(actual, expected)

    def test_published_mtp_namespace_loads_only_draft_weights(self):
        """Bare packed-expert tensors under mtp.* must be loaded into the
        draft while target tensors, including the embedding/head that EAGLE
        shares before inference, are ignored. Missing MTP weights still fail.
        """
        model = alice_ai_mtp.AliceAIForCausalLMMTP.__new__(
            alice_ai_mtp.AliceAIForCausalLMMTP
        )
        nn.Module.__init__(model)
        model.config = SimpleNamespace(num_experts=2, tie_word_embeddings=False)
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(3, 2)
        model.lm_head = nn.Linear(2, 3, bias=False)
        initial_embedding = model.model.embed_tokens.weight.detach().clone()
        initial_head = model.lm_head.weight.detach().clone()
        model.model.fc = nn.Linear(4, 2, bias=False)
        for name in ("pre_fc_norm_embedding", "pre_fc_norm_hidden", "norm"):
            module = nn.Module()
            module.weight = nn.Parameter(torch.empty(2))
            setattr(model.model, name, module)
        layer = nn.Module()
        model.model.layers = nn.ModuleList([layer])
        for name in ("input_layernorm", "post_attention_layernorm"):
            module = nn.Module()
            module.weight = nn.Parameter(torch.empty(2))
            setattr(layer, name, module)
        attention = layer.self_attn = nn.Module()
        attention.qkv_proj = nn.Linear(2, 8, bias=False)

        def load_qkv(param, weight, shard_id):
            offset, length = {"q": (0, 4), "k": (4, 2), "v": (6, 2)}[shard_id]
            param.data[offset : offset + length].copy_(weight)

        attention.qkv_proj.weight.weight_loader = load_qkv
        attention.o_proj = nn.Linear(2, 2, bias=False)
        attention.q_norm, attention.k_norm = nn.Module(), nn.Module()
        attention.q_norm.weight = nn.Parameter(torch.empty(2))
        attention.k_norm.weight = nn.Parameter(torch.empty(2))
        mlp = layer.mlp = nn.Module()
        mlp.gate = nn.Linear(2, 2, bias=False)
        mlp.gate.e_score_correction_bias = nn.Parameter(torch.empty(2))
        mlp.shared_expert_gate = nn.Linear(2, 1, bias=False)
        mlp.shared_expert = nn.Module()
        mlp.shared_expert.gate_up_proj = nn.Linear(2, 4, bias=False)
        mlp.shared_expert.down_proj = nn.Linear(2, 2, bias=False)

        def load_shared(param, weight, shard_id):
            param.data[2 * shard_id : 2 * (shard_id + 1)].copy_(weight)

        mlp.shared_expert.gate_up_proj.weight.weight_loader = load_shared
        experts = mlp.experts = nn.Module()
        experts.w13_weight = nn.Parameter(torch.empty(2, 4, 2))
        experts.w2_weight = nn.Parameter(torch.empty(2, 2, 2))

        def load_expert(param, weight, name, expert_id, shard_id):
            if shard_id == "w2":
                param.data[expert_id].copy_(weight)
            else:
                offset = 0 if shard_id == "w1" else 2
                param.data[expert_id, offset : offset + 2].copy_(weight)

        experts.w13_weight.weight_loader = experts.w2_weight.weight_loader = load_expert
        shapes = {
            "mtp.layers.0.input_layernorm.weight": (2,),
            "mtp.layers.0.post_attention_layernorm.weight": (2,),
            "mtp.layers.0.self_attn.q_proj.weight": (4, 2),
            "mtp.layers.0.self_attn.k_proj.weight": (2, 2),
            "mtp.layers.0.self_attn.v_proj.weight": (2, 2),
            "mtp.layers.0.self_attn.o_proj.weight": (2, 2),
            "mtp.layers.0.self_attn.k_norm.weight": (2,),
            "mtp.layers.0.self_attn.q_norm.weight": (2,),
            "mtp.layers.0.mlp.gate.e_score_correction_bias": (2,),
            "mtp.layers.0.mlp.gate.weight": (2, 2),
            "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 2),
            "mtp.layers.0.mlp.shared_expert.up_proj.weight": (2, 2),
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (2, 2),
            "mtp.layers.0.mlp.shared_expert.down_proj.weight": (2, 2),
            "mtp.layers.0.mlp.experts.gate_up_proj": (2, 4, 2),
            "mtp.layers.0.mlp.experts.down_proj": (2, 2, 2),
            "mtp.fc.weight": (2, 4),
            "mtp.pre_fc_norm_embedding.weight": (2,),
            "mtp.pre_fc_norm_hidden.weight": (2,),
            "mtp.norm.weight": (2,),
            "model.embed_tokens.weight": (3, 2),
            "lm_head.weight": (3, 2),
        }
        weights = {
            name: torch.arange(
                torch.tensor(shape).prod().item(), dtype=torch.float32
            ).reshape(shape)
            + index * 100
            for index, (name, shape) in enumerate(shapes.items())
        }
        entries = list(weights.items()) + [
            ("model.layers.0.linear_attn.unused.weight", torch.ones(1)),
            ("model.norm.weight", torch.full((2,), -999.0)),
        ]
        loaded = model.load_weights(iter(entries))
        self.assertEqual(
            loaded,
            set(dict(model.named_parameters()))
            - {"model.embed_tokens.weight", "lm_head.weight"},
        )
        torch.testing.assert_close(model.model.norm.weight, weights["mtp.norm.weight"])
        torch.testing.assert_close(model.model.fc.weight, weights["mtp.fc.weight"])
        torch.testing.assert_close(model.model.embed_tokens.weight, initial_embedding)
        torch.testing.assert_close(model.lm_head.weight, initial_head)
        torch.testing.assert_close(
            attention.qkv_proj.weight,
            torch.cat(
                [
                    weights[f"mtp.layers.0.self_attn.{kind}_proj.weight"]
                    for kind in ("q", "k", "v")
                ]
            ),
        )
        torch.testing.assert_close(
            experts.w13_weight, weights["mtp.layers.0.mlp.experts.gate_up_proj"]
        )
        torch.testing.assert_close(
            experts.w2_weight, weights["mtp.layers.0.mlp.experts.down_proj"]
        )
        with self.assertRaisesRegex(ValueError, "missing weights.*model.fc.weight"):
            model.load_weights(
                (name, value) for name, value in entries if name != "mtp.fc.weight"
            )
        target_embedding = nn.Parameter(torch.randn(3, 2))
        target_head = nn.Parameter(torch.randn(3, 2))
        model.set_embed_and_head(target_embedding, target_head)
        shared_embedding, shared_head = model.get_embed_and_head()
        self.assertIs(shared_embedding, target_embedding)
        self.assertIs(shared_head, target_head)
        self.assertIsNot(shared_embedding, shared_head)

    def test_nextn_resolves_bundled_draft_without_mutating_target_geometry(self):
        """Draft conversion must not mutate the cached target HF config or
        lose the MTP KV budget when the draft path defaults to the target.
        """
        config = AliceAIConfig(
            architectures=["AliceAIForCausalLM"],
            mtp_num_hidden_layers=1,
            hidden_size=4,
            vocab_size=8,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            with override_platform(
                is_cuda=False,
                is_hip=False,
                device_sm=0,
                device_capability=(0, 0),
                is_sm90=False,
                is_sm100=False,
                is_sm100_or_sm110=False,
                is_sm120=False,
                is_blackwell=False,
            ):
                for draft_revision in (None, "draft-pin"):
                    with self.subTest(draft_revision=draft_revision):
                        args = ServerArgs(
                            model_path=directory,
                            revision="target-pin",
                            device="cpu",
                            dtype="float32",
                            attention_backend="torch_native",
                            linear_attn_backend="triton",
                            mamba_radix_cache_strategy="no_buffer",
                            disable_radix_cache=True,
                            disable_cuda_graph=True,
                            speculative_algorithm="NEXTN",
                            speculative_draft_model_revision=draft_revision,
                            speculative_num_steps=3,
                            speculative_eagle_topk=1,
                            speculative_num_draft_tokens=4,
                        )
                        args.resolve_once()
                        resolved = resolved_view(args)
                        self.assertEqual(resolved.speculative_algorithm, "EAGLE")
                        self.assertIsNone(resolved.speculative_draft_model_path)
                        self.assertEqual(args.speculative_algorithm, "NEXTN")
                        self.assertIsNone(args.speculative_draft_model_path)
                        target = model_config_of(args)
                        with get_context().override_server_args(
                            speculative_algorithm=resolved.speculative_algorithm,
                            speculative_draft_model_path=None,
                            speculative_draft_model_revision=draft_revision,
                        ):
                            draft = ModelConfig.from_server_args(
                                args,
                                model_path=resolved.speculative_draft_model_path,
                                model_revision=resolved.speculative_draft_model_revision,
                                is_draft_model=True,
                            )
                            spec_aux = resolve_spec_aux_hidden_state_config(
                                server_args=args,
                                model_config=target,
                                spec_algorithm=SpeculativeAlgorithm.EAGLE,
                                is_draft_worker=False,
                            )
                        self.assertEqual(draft.model_path, directory)
                        self.assertEqual(draft.revision, draft_revision or "target-pin")
                        self.assertEqual(spec_aux.eagle_draft_num_layers, 1)
                        self.assertEqual(
                            draft.hf_config.architectures, ["AliceAIForCausalLMMTP"]
                        )
                        self.assertEqual(draft.hf_config.num_hidden_layers, 1)
                        self.assertEqual(
                            draft.hf_config.layer_types, ["full_attention"]
                        )
                        self.assertEqual(draft.hf_config.num_nextn_predict_layers, 1)
                        self.assertIsNone(get_linear_attn_config(draft.hf_config))
                        self.assertEqual(
                            target.hf_config.architectures, ["AliceAIForCausalLM"]
                        )
                        self.assertEqual(target.hf_config.num_hidden_layers, 48)
                        self.assertEqual(len(target.hf_config.linear_layer_ids), 36)
                        self.assertIsNotNone(get_linear_attn_config(target.hf_config))


if __name__ == "__main__":
    unittest.main()
