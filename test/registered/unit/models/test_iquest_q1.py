import unittest

import torch

from sglang.srt.configs.iquest_q1 import IQuestQ1Config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestIQuestQ1WeightLoading(CustomTestCase):
    def test_target_ignores_native_mtp_weights(self):
        from sglang.srt.models.iquest_q1 import IQuestQ1ForCausalLM

        native_weights = [
            (f"mtp_layers.{layer}.{suffix}", torch.empty(1, device="meta"))
            for layer in range(2)
            for suffix in (
                "eh_proj.weight",
                "enorm.weight",
                "final_layernorm.weight",
                "hnorm.weight",
                "mtp_model_layer.mlp.experts.fc",
                "mtp_model_layer.mlp.experts.proj",
                "mtp_model_layer.mlp.router.weight",
                "mtp_model_layer.self_attn.q_proj.weight",
                "mtp_model_layer.self_attn.sink_k",
            )
        ]
        model = IQuestQ1ForCausalLM.__new__(IQuestQ1ForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = IQuestQ1Config(num_mtp_layers=2)
        model.quant_config = None
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 2)
        model.lm_head = torch.nn.Linear(2, 4, bias=False)
        weights = [
            ("model.embed_tokens.weight", torch.full((4, 2), 2.0)),
            ("lm_head.weight", torch.full((4, 2), 3.0)),
        ]
        model.load_weights(iter(native_weights + weights))
        self.assertEqual(set(model.state_dict()), {name for name, _ in weights})
        for name, expected in weights:
            torch.testing.assert_close(model.state_dict()[name], expected)
        with self.assertRaisesRegex(ValueError, "lm_head.weight"):
            model.load_weights(iter(weights[:1] + native_weights))
        for name in ("mtp_layers_extra.weight", "mtp.eh_proj.weight"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "Unexpected IQuest Q1 weight"),
            ):
                model.load_weights(iter(weights + [(name, torch.ones(1))]))

    def test_unrecognized_quantized_parameters_warn_but_partial_weights_fail(self):
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.models.iquest_q1 import check_all_params_loaded

        for name, config in (
            (
                "model.layers.0.mlp.experts.w13_weight_scale",
                Fp8Config(is_checkpoint_fp8_serialized=True),
            ),
            ("model.learned_scale", Fp8Config(is_checkpoint_fp8_serialized=False)),
        ):
            with self.subTest(name=name):
                with self.assertLogs(
                    "sglang.srt.models.iquest_q1", level="WARNING"
                ) as logs:
                    check_all_params_loaded(
                        {name: torch.nn.Parameter(torch.zeros(1))},
                        set(),
                        "IQuestQ1ForCausalLM",
                        quant_config=config,
                    )
                self.assertIn(name, logs.output[0])
                self.assertIn("not found", logs.output[0])
                with self.assertRaisesRegex(ValueError, "not found"):
                    check_all_params_loaded(
                        {name: torch.nn.Parameter(torch.zeros(1))},
                        set(),
                        "IQuestQ1ForCausalLM",
                    )
        name = "model.layers.0.self_attn.qkv_proj.weight"
        with self.assertRaisesRegex(ValueError, "partially loaded"):
            check_all_params_loaded(
                {name: torch.nn.Parameter(torch.zeros(1))},
                {name},
                "IQuestQ1ForCausalLM",
                loaded_shards={name: {"q"}},
                quant_config=Fp8Config(),
            )
        generated = torch.nn.Parameter(torch.zeros(1))
        generated._skip_weight_check = True
        check_all_params_loaded(
            {"generated_parameter": generated}, set(), "IQuestQ1ForCausalLM"
        )

    def test_target_fused_loaders_require_every_checkpoint_shard(self):
        from sglang.srt.models.iquest_q1 import IQuestQ1ForCausalLM

        for projection, components in (
            ("qkv_proj", ("q_proj", "k_proj", "v_proj")),
            ("gate_up_proj", ("gate_proj", "up_proj")),
        ):
            with self.subTest(projection=projection):
                model = IQuestQ1ForCausalLM.__new__(IQuestQ1ForCausalLM)
                torch.nn.Module.__init__(model)
                model.quant_config = None
                layer = torch.nn.Linear(2, 2 * len(components), bias=False)
                layer.weight.requires_grad_(False)

                def loader(param, weight, shard):
                    index = {"q": 0, "k": 1, "v": 2}.get(shard, shard)
                    param.data[index * 2 : (index + 1) * 2].copy_(weight)

                layer.weight.weight_loader = loader
                model.model = torch.nn.Module()
                setattr(model.model, projection, layer)
                weights = [
                    (f"model.{name}.weight", torch.full((2, 2), float(i + 1)))
                    for i, name in enumerate(components)
                ]
                with self.assertRaisesRegex(ValueError, "partially loaded"):
                    model.load_weights(weights[:1])
                model.load_weights(weights)
                torch.testing.assert_close(
                    layer.weight, torch.cat([w for _, w in weights])
                )


class TestIQuestQ1FP8ExpertLoading(CustomTestCase):
    def _make_model(self, is_mtp, first_expert):
        from sglang.srt.layers.quantization.fp8 import Fp8Config
        from sglang.srt.models.iquest_q1 import IQuestQ1ForCausalLM
        from sglang.srt.models.iquest_q1_mtp import IQuestQ1MTP

        cls = IQuestQ1MTP if is_mtp else IQuestQ1ForCausalLM
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)
        model.config = IQuestQ1Config(num_experts=2, intermediate_size=2)
        model.quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True, weight_block_size=[128, 128]
        )
        if is_mtp:
            prefix = "model.mtp_layer.mtp_model_layer.mlp.experts"
            checkpoint_prefix = "mtp.mtp_model_layer.mlp.experts"
        else:
            prefix = "model.layers.0.mlp.experts"
            checkpoint_prefix = prefix

        experts = model
        for component in prefix.split("."):
            child = torch.nn.Module()
            experts.add_module(component, child)
            experts = child

        def load_expert(param, weight, name, shard_id, expert_id):
            # Simulate an EP rank that owns only a subset of the experts.
            if expert_id < first_expert:
                return
            destination = param.data[expert_id - first_expert]
            if shard_id == "w1":
                destination[:2].copy_(weight)
            elif shard_id == "w3":
                destination[2:].copy_(weight)
            else:
                self.assertEqual(shard_id, "w2")
                destination.copy_(weight)

        for name, width in (
            ("w13_weight", 4),
            ("w2_weight", 2),
            ("w13_weight_scale_inv", 4),
            ("w2_weight_scale_inv", 2),
        ):
            param = torch.nn.Parameter(
                torch.full((2 - first_expert, width, 2), float("nan")),
                requires_grad=False,
            )
            param.weight_loader = load_expert
            experts.register_parameter(name, param)
        return model, experts, checkpoint_prefix

    def test_fp8_expert_scales(self):
        for is_mtp in (False, True):
            for first_expert in (0, 1):
                for packed_weights in (False, True):
                    with self.subTest(
                        is_mtp=is_mtp,
                        first_expert=first_expert,
                        packed_weights=packed_weights,
                    ):
                        model, experts, prefix = self._make_model(is_mtp, first_expert)
                        fc = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2)
                        proj = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
                        weights = []
                        if packed_weights:
                            weights.extend(
                                [(f"{prefix}.fc", fc), (f"{prefix}.proj", proj)]
                            )
                        for expert_id in range(2):
                            for projection, weight, scale in (
                                ("gate_proj", fc[expert_id, :2], 10 + expert_id),
                                ("up_proj", fc[expert_id, 2:], 20 + expert_id),
                                ("down_proj", proj[expert_id], 30 + expert_id),
                            ):
                                name = f"{prefix}.{expert_id}.{projection}"
                                if not packed_weights:
                                    weights.append((f"{name}.weight", weight))
                                weights.append(
                                    (
                                        f"{name}.weight_scale_inv",
                                        torch.full((2, 2), float(scale)),
                                    )
                                )
                        model.load_weights(iter(weights))
                        torch.testing.assert_close(
                            experts.w13_weight, fc[first_expert:]
                        )
                        torch.testing.assert_close(
                            experts.w2_weight, proj[first_expert:]
                        )
                        for expert_id in range(first_expert, 2):
                            local_id = expert_id - first_expert
                            scales = experts.w13_weight_scale_inv[local_id]
                            torch.testing.assert_close(
                                scales[:2], torch.full((2, 2), float(10 + expert_id))
                            )
                            torch.testing.assert_close(
                                scales[2:], torch.full((2, 2), float(20 + expert_id))
                            )
                            torch.testing.assert_close(
                                experts.w2_weight_scale_inv[local_id],
                                torch.full((2, 2), float(30 + expert_id)),
                            )

    def test_unknown_expert_parameters_rejected(self):
        for is_mtp in (False, True):
            model, _, prefix = self._make_model(is_mtp, first_expert=0)
            for suffix in (
                "0.gate_proj.misspelled_scale",
                "0.unknown_proj.weight_scale_inv",
                "2.gate_proj.weight_scale_inv",
            ):
                with (
                    self.subTest(is_mtp=is_mtp, suffix=suffix),
                    self.assertRaisesRegex(ValueError, "Unexpected IQuest Q1 weight"),
                ):
                    model.load_weights([(f"{prefix}.{suffix}", torch.ones(2, 2))])


class TestIQuestQ1MTPDraft(CustomTestCase):
    def test_mtp_loads_own_weights(self):
        from sglang.srt.models.iquest_q1_mtp import IQuestQ1MTP

        model = IQuestQ1MTP.__new__(IQuestQ1MTP)
        torch.nn.Module.__init__(model)
        model.quant_config = None
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 2)
        model.lm_head = torch.nn.Linear(2, 4, bias=False)
        model.model.mtp_layer = torch.nn.Module()
        model.model.mtp_layer.self_attn = torch.nn.Module()
        qkv = torch.nn.Linear(2, 6, bias=False)
        model.model.mtp_layer.self_attn.qkv_proj = qkv

        def loader(param, weight, shard):
            offset = {"q": 0, "k": 2, "v": 4}[shard]
            param.data[offset : offset + 2].copy_(weight)

        qkv.weight.weight_loader = loader
        weights = [
            ("embed_tokens.weight", torch.full((4, 2), 2.0)),
            ("lm_head.weight", torch.full((4, 2), 3.0)),
            ("target_final_norm.weight", torch.ones(2)),
        ] + [
            (f"mtp.self_attn.{name}_proj.weight", torch.full((2, 2), float(i)))
            for i, name in enumerate(("q", "k", "v"), 1)
        ]
        model.load_weights(weights)
        torch.testing.assert_close(qkv.weight, torch.cat([w for _, w in weights[-3:]]))
        torch.testing.assert_close(model.model.embed_tokens.weight, weights[0][1])
        torch.testing.assert_close(model.lm_head.weight, weights[1][1])
        with self.assertRaisesRegex(ValueError, "Unexpected MTP weight"):
            model.load_weights(weights + [("unknown.weight", torch.ones(1))])
        with self.assertRaisesRegex(ValueError, "Unexpected MTP weight"):
            model.load_weights(
                weights + [("mtp_layers.0.eh_proj.weight", torch.ones(1))]
            )
        with self.assertRaisesRegex(ValueError, "lm_head.weight"):
            model.load_weights(
                [item for item in weights if item[0] != "lm_head.weight"]
            )

    def test_mtp_keeps_its_own_embedding_and_head(self):
        from sglang.srt.models.iquest_q1_mtp import IQuestQ1MTP

        model = IQuestQ1MTP.__new__(IQuestQ1MTP)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 2)
        model.lm_head = torch.nn.Linear(2, 4, bias=False)
        embed, head = model.get_embed_and_head()
        model.set_embed_and_head(torch.zeros(4, 2), torch.zeros(4, 2))
        current_embed, current_head = model.get_embed_and_head()
        self.assertIs(current_embed, embed)
        self.assertIs(current_head, head)

    def test_mtp_residual_is_fp32_but_moe_input_is_bf16(self):
        from sglang.srt.models.iquest_q1_mtp import (
            IQuestQ1MTPInnerLayer,
        )

        class Attention(torch.nn.Module):
            def forward(self, positions, hidden_states, forward_batch):
                return torch.full_like(hidden_states, 0.25)

        class Experts(torch.nn.Module):
            def forward(self, hidden_states):
                self.input_dtype = hidden_states.dtype
                return torch.full_like(hidden_states, 0.25)

        layer = IQuestQ1MTPInnerLayer.__new__(IQuestQ1MTPInnerLayer)
        torch.nn.Module.__init__(layer)
        layer.attention_norm = torch.nn.Identity()
        layer.attn_out_norm = torch.nn.Identity()
        layer.feed_forward_norm = torch.nn.Identity()
        layer.ffn_out_norm = torch.nn.Identity()
        layer.self_attn = Attention()
        layer.mlp = Experts()
        layer.attn_out_scale = 1.0
        layer.ffn_out_scale = 1.0
        hidden = torch.full((2, 2), 256.0, dtype=torch.bfloat16)
        for fp32 in (False, True):
            layer.fp32_residual_connection = fp32
            result = layer(torch.tensor([0, 1]), hidden, None)
            self.assertEqual(layer.mlp.input_dtype, torch.bfloat16)
            expected = hidden.float() + 0.5 if fp32 else hidden
            torch.testing.assert_close(result, expected)


if __name__ == "__main__":
    unittest.main()
