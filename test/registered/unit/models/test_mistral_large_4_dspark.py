import unittest
from unittest import mock

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.rotary_embedding import get_rope, rope_variant
from sglang.srt.models import dspark
from sglang.srt.models.dspark import RNNHead
from sglang.srt.utils.hf_transformers.mistral_utils import (
    mistral_large_4_dspark_draft_config,
)
from sglang.srt.utils.hf_transformers_utils import get_rope_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN, HEADS, NOPE, ROPE, V, RANK, FFN, VOCAB = 64, 4, 8, 8, 12, 16, 32, 100
DEPTH, MARKOV_RANK = 3, 8
ROPE_SCALING = {
    "rope_type": "deepseek_yarn",
    "factor": 4,
    "original_max_position_embeddings": 64,
    "beta_fast": 32,
    "beta_slow": 1,
    "mscale_all_dim": 1,
    "apply_yarn_scaling": False,
}

# Every tensor the checkpoint stores under mtp_dspark.
_DRAFT_TENSORS = [
    *(
        f"mtp_dspark.backbone.layers.{layer}.{name}.weight"
        for layer in range(3)
        for name in (
            "attention.wq",
            "attention.wkv_a_with_mqa",
            "attention.wkv_b",
            "attention.wo",
            "attention.kv_a_norm",
            "attention_norm",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "ffn_norm",
        )
    ),
    "mtp_dspark.backbone.final_norm.weight",
    "mtp_dspark.backbone.mask_embedding",
    "mtp_dspark.backbone.target_norm.weight",
    "mtp_dspark.backbone.target_projection.weight",
    "mtp_dspark.autoregressive_head.embedding.weight",
    "mtp_dspark.autoregressive_head.joint_proj.bias",
    "mtp_dspark.autoregressive_head.joint_proj.weight",
    "mtp_dspark.autoregressive_head.output.weight",
]
_TARGET_LAYERS = [1, 18, 36, 54, 71]


def _on_cpu():
    # YaRN rotaries resolve their device from the accelerator when none is given.
    return mock.patch.object(rope_variant, "get_device", return_value="cpu")


def _draft_config(draft_input: str = "position_specific") -> PretrainedConfig:
    target = PretrainedConfig.from_dict(
        {
            "model_type": "mistral_large_4",
            "hidden_size": HIDDEN,
            "num_hidden_layers": 72,
            "num_attention_heads": HEADS,
            "q_lora_rank": None,
            "kv_lora_rank": RANK,
            "qk_nope_head_dim": NOPE,
            "qk_rope_head_dim": ROPE,
            "v_head_dim": V,
            "intermediate_size": FFN,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "rope_scaling": ROPE_SCALING,
            "max_position_embeddings": 256,
            "vocab_size": VOCAB,
            "softmax_tanh_gating": 30.0,
            "nope_layer_flags": [layer % 4 == 3 for layer in range(72)],
            "dspark_block_size": DEPTH,
            "dspark_markov_rank": MARKOV_RANK,
            "dspark_markov_head_type": "rnn",
            "dspark_target_layer_ids": _TARGET_LAYERS,
            "dspark_noise_token_id": 0,
            "dspark_draft_input": draft_input,
            "dspark_num_layers": 3,
            "dspark_sliding_window": 2048,
        }
    )
    return mistral_large_4_dspark_draft_config(target)


def _build(config=None):
    from sglang.srt.models.dspark_mla import MistralLarge4DSparkModel

    with published_topology(), _on_cpu():
        return MistralLarge4DSparkModel(config or _draft_config())


def _target_rope(config) -> nn.Module:
    # The target's MLA rotary on GPU: get_rope with the target config's YaRN.
    rope_theta, rope_scaling = get_rope_config(config)
    with _on_cpu():
        return get_rope(
            ROPE,
            ROPE,
            config.max_position_embeddings,
            rope_theta,
            False,
            rope_scaling,
        )


class TestMistralLarge4DSparkDraft(CustomTestCase):
    def test_checkpoint_tensors_load_into_the_draft(self):
        torch.manual_seed(0)
        model = _build()
        params = dict(model.named_parameters())

        def shape(name: str) -> tuple:
            if ".feed_forward.w2." in name:
                return (HIDDEN, FFN)
            if ".feed_forward.w" in name:
                return (FFN, HIDDEN)
            from sglang.srt.models.dspark_mla import (
                mistral_large_4_dspark_weight_name,
            )

            target = mistral_large_4_dspark_weight_name(name)
            module = dict(model.named_modules())[target.rpartition(".")[0]]
            if hasattr(module, "output_size") and target.endswith(".weight"):
                return (module.output_size, module.input_size)
            return tuple(params[target].shape)

        checkpoint = {name: torch.randn(shape(name)) for name in _DRAFT_TENSORS}
        target_tensors = [
            ("tok_embeddings.weight", torch.randn(VOCAB, HIDDEN)),
            ("layers.0.attention.wq.weight", torch.randn(1)),
            ("lerope.alpha", torch.randn(1)),
            ("vision_encoder.ln_pre.weight", torch.randn(1)),
        ]
        model.load_weights([*target_tensors, *checkpoint.items()])

        prefix = "mtp_dspark.backbone."
        expected = {
            "fc.weight": checkpoint[prefix + "target_projection.weight"],
            "hidden_norm.weight": checkpoint[prefix + "target_norm.weight"],
            "norm.weight": checkpoint[prefix + "final_norm.weight"],
            "mask_embedding": checkpoint[prefix + "mask_embedding"],
            "markov_head.markov_w1.weight": checkpoint[
                "mtp_dspark.autoregressive_head.embedding.weight"
            ],
            "markov_head.markov_w2.weight": checkpoint[
                "mtp_dspark.autoregressive_head.output.weight"
            ],
            "markov_head.joint_proj.weight": checkpoint[
                "mtp_dspark.autoregressive_head.joint_proj.weight"
            ],
            "markov_head.joint_proj.bias": checkpoint[
                "mtp_dspark.autoregressive_head.joint_proj.bias"
            ],
        }
        for layer in range(3):
            source = f"{prefix}layers.{layer}."
            target = f"layers.{layer}."
            expected |= {
                target + "self_attn.q_proj.weight": checkpoint[
                    source + "attention.wq.weight"
                ],
                target + "self_attn.kv_a_proj_with_mqa.weight": checkpoint[
                    source + "attention.wkv_a_with_mqa.weight"
                ],
                target + "self_attn.kv_a_layernorm.weight": checkpoint[
                    source + "attention.kv_a_norm.weight"
                ],
                target + "self_attn.kv_b_proj.weight": checkpoint[
                    source + "attention.wkv_b.weight"
                ],
                target + "self_attn.o_proj.weight": checkpoint[
                    source + "attention.wo.weight"
                ],
                target + "input_layernorm.weight": checkpoint[
                    source + "attention_norm.weight"
                ],
                target + "post_attention_layernorm.weight": checkpoint[
                    source + "ffn_norm.weight"
                ],
                # w1 is the gate, w3 the up projection.
                target + "mlp.gate_up_proj.weight": torch.cat(
                    [
                        checkpoint[source + "feed_forward.w1.weight"],
                        checkpoint[source + "feed_forward.w3.weight"],
                    ]
                ),
                target + "mlp.down_proj.weight": checkpoint[
                    source + "feed_forward.w2.weight"
                ],
            }
        self.assertEqual(set(expected), set(params))
        for name, value in expected.items():
            with self.subTest(param=name):
                self.assertTrue(torch.equal(params[name], value))

        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            _build().load_weights(
                [("mtp_dspark.backbone.layers.0.attention.sinks", torch.zeros(1))]
            )

    def test_draft_rope_is_plain_yarn_after_target_lerope(self):
        config = _draft_config()
        target_rope = _target_rope(config)
        plain_yarn = target_rope.cos_sin_cache.clone()
        target_rope.set_lerope_alpha(torch.full((ROPE // 2,), 0.5))

        model = _build(config)

        for layer in model.layers:
            rope = layer.self_attn.rotary_emb
            self.assertIsNot(rope, target_rope)
            self.assertTrue(torch.equal(rope.cos_sin_cache, plain_yarn))

    def test_blocks_are_anchor_then_position_specific_masks(self):
        model = _build()
        embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        model.embed_tokens = embed_tokens
        with torch.no_grad():
            model.mask_embedding.normal_()
        anchors = torch.tensor([5, 17])
        # The worker fills the mask slots with an arbitrary in-vocab sentinel.
        input_ids = torch.tensor([[5, 0, 0], [17, 0, 0]]).flatten()

        embeds = model.embed_blocks(input_ids, DEPTH).view(2, DEPTH, HIDDEN)

        self.assertTrue(torch.equal(embeds[:, 0], embed_tokens(anchors)))
        for slot in range(1, DEPTH):
            self.assertTrue(
                torch.equal(
                    embeds[:, slot], model.mask_embedding[slot - 1].expand(2, -1)
                )
            )
        with self.assertRaisesRegex(ValueError, "3-token blocks"):
            model.embed_blocks(torch.zeros(8, dtype=torch.long), 4)

    def test_soft_cap_applies_to_base_logits(self):
        model = _build()
        lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
        lm_head.quant_method = None
        lm_head.org_vocab_size = VOCAB
        model.attach_shared_modules(
            embed_tokens=nn.Embedding(VOCAB, HIDDEN), lm_head=lm_head
        )
        hidden = torch.randn(4, HIDDEN) * 50

        with mock.patch.object(
            dspark, "tensor_model_parallel_all_gather", lambda x, dim: x
        ):
            base_logits, _ = model.compute_base_logits(hidden)

        raw = hidden @ lm_head.weight.T
        self.assertTrue(torch.allclose(base_logits, 30.0 * torch.tanh(raw / 30.0)))


def _trainer_rnn_bias(prev_ids, hidden, embedding, joint_proj, output):
    """Mistral's trainer RNNHead (mtp.dspark.autoregressive_head), unrolled."""
    state = None
    biases = []
    for d in range(prev_ids.shape[1]):
        emb = embedding(prev_ids[:, d])
        if state is None:
            state = torch.zeros_like(emb)
        joint = joint_proj(torch.cat([state, emb, hidden[:, d]], dim=-1))
        gate_raw, candidate_raw, output_raw = joint.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(candidate_raw)
        biases.append(output(torch.tanh(output_raw)))
    return torch.stack(biases, dim=1)


class TestRNNHeadMatchesTrainer(CustomTestCase):
    def test_step_bias_threads_the_state_like_the_trainer(self):
        torch.manual_seed(0)
        head = RNNHead(vocab_size=VOCAB, markov_rank=MARKOV_RANK, hidden_size=HIDDEN)
        with torch.no_grad():
            for param in head.parameters():
                param.normal_()
        prev_ids = torch.randint(VOCAB, (3, DEPTH))
        hidden = torch.randn(3, DEPTH, HIDDEN)

        state = head.init_state(hidden[:, 0])
        biases = []
        for d in range(DEPTH):
            bias, state = head.step_bias(
                prev_ids[:, d], hidden_states=hidden[:, d], state=state
            )
            biases.append(bias)

        expected = _trainer_rnn_bias(
            prev_ids, hidden, head.markov_w1, head.joint_proj, head.markov_w2
        )
        torch.testing.assert_close(torch.stack(biases, dim=1), expected)


if __name__ == "__main__":
    unittest.main()
