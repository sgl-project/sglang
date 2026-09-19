import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import (
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestQwen3_5PipelineParallel(CustomTestCase):
    @staticmethod
    def _make_mtp_weight_loader_stub():
        model = Qwen3_5ForCausalLMMTP.__new__(Qwen3_5ForCausalLMMTP)
        torch.nn.Module.__init__(model)
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(4, 3)
        model.config = SimpleNamespace(num_experts=None)
        model.quant_config = None
        with torch.no_grad():
            model.model.embed_tokens.weight.fill_(torch.nan)
        return model

    @staticmethod
    def _get_num_fused_shared_experts(layers, start_layer, end_layer):
        model = SimpleNamespace(
            model=SimpleNamespace(
                layers=layers,
                start_layer=start_layer,
                end_layer=end_layer,
            )
        )
        return Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)

    def test_get_num_fused_shared_experts_returns_zero_without_layers(self):
        model = SimpleNamespace(model=SimpleNamespace())

        num_fused_shared_experts = (
            Qwen3_5MoeForConditionalGeneration._get_num_fused_shared_experts(model)
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_get_num_fused_shared_experts_uses_local_pp_layers(self):
        layers = [
            PPMissingLayer(),
            PPMissingLayer(),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
            SimpleNamespace(
                mlp=SimpleNamespace(num_fused_shared_experts=1),
            ),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=2,
            end_layer=4,
        )

        self.assertEqual(num_fused_shared_experts, 1)

    def test_get_num_fused_shared_experts_returns_zero_without_local_fusion(self):
        layers = [
            PPMissingLayer(),
            SimpleNamespace(mlp=SimpleNamespace()),
        ]

        num_fused_shared_experts = self._get_num_fused_shared_experts(
            layers,
            start_layer=1,
            end_layer=2,
        )

        self.assertEqual(num_fused_shared_experts, 0)

    def test_mtp_loads_vl_target_embedding_for_last_pp_stage(self):
        model = self._make_mtp_weight_loader_stub()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        loaded = model.load_weights(
            [("model.language_model.embed_tokens.weight", expected)]
        )

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)

    def test_mtp_loads_text_target_embedding_for_last_pp_stage(self):
        model = self._make_mtp_weight_loader_stub()
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)

        loaded = model.load_weights([("model.embed_tokens.weight", expected)])

        self.assertEqual(loaded, {"model.embed_tokens.weight"})
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)


class TestQwen3_5SharedExpertWeightLoading(CustomTestCase):
    @staticmethod
    def _make_model(model_class, suffix):
        model = model_class.__new__(model_class)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            num_experts=2, tie_word_embeddings=False, encoder_only=False
        )
        model.quant_config = None
        model.num_fused_shared_experts = 1
        model.enable_shared_expert_fusion = True
        model.model = torch.nn.Module()
        model.model.start_layer = 0
        model.model.end_layer = 1
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.num_fused_shared_experts = 1
        layer.mlp.experts = torch.nn.Module()
        if model_class is Qwen3_5MoeForCausalLM:
            model.layers = torch.nn.ModuleList([layer])
            model._start_layer = 0
            model._end_layer = 1
        else:
            model.model.layers = torch.nn.ModuleList([layer])

        def load_expert(param, weight, name, shard_id, expert_id):
            target = param.data[expert_id]
            if shard_id != "w2":
                target = target.chunk(2, dim=0)[0 if shard_id == "w1" else 1]
            target.copy_(weight)

        for projection, shape in (("w13", (3, 4, 3)), ("w2", (3, 3, 2))):
            param = torch.nn.Parameter(
                torch.full(shape, torch.nan), requires_grad=False
            )
            param.weight_loader = load_expert
            layer.mlp.experts.register_parameter(f"{projection}_{suffix}", param)
        return model, layer.mlp.experts

    def test_shared_weights_and_fp8_scales_reach_the_extra_slot(self):
        """CUDA used to leave the shared slot uninitialized in the MTP loader.

        Packed routed/shared tensors can also arrive in either order; the
        previous tensor must not change the mapping used for this one.
        """
        for model_class, prefix in (
            (Qwen3_5MoeForConditionalGeneration, "model.language_model"),
            (Qwen3_5ForCausalLMMTP, "mtp"),
            (Qwen3_5MoeForCausalLM, ""),
        ):
            for packed_routed, packed_shared, suffix in (
                (False, False, "weight"),
                (False, False, "weight_scale_inv"),
                (True, False, "weight"),
                (True, True, "weight"),
            ):
                for reverse in (False, True):
                    with self.subTest(
                        model=model_class.__name__,
                        packed_routed=packed_routed,
                        packed_shared=packed_shared,
                        suffix=suffix,
                        reverse=reverse,
                    ):
                        model, experts = self._make_model(model_class, suffix)
                        gate = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)
                        up = gate + 30
                        down = (
                            torch.arange(18, dtype=torch.float32).reshape(3, 3, 2) + 60
                        )
                        base = f"{prefix}.layers.0.mlp".lstrip(".")
                        weights = []
                        if packed_routed:
                            weights.extend(
                                [
                                    (
                                        f"{base}.experts.gate_up_proj",
                                        torch.cat((gate[:2], up[:2]), dim=1),
                                    ),
                                    (f"{base}.experts.down_proj", down[:2]),
                                ]
                            )
                        else:
                            for expert_id in range(2):
                                for projection, tensor in (
                                    ("gate", gate),
                                    ("up", up),
                                    ("down", down),
                                ):
                                    weights.append(
                                        (
                                            f"{base}.experts.{expert_id}.{projection}_proj.{suffix}",
                                            tensor[expert_id],
                                        )
                                    )
                        if packed_shared:
                            weights.append(
                                (
                                    f"{base}.shared_expert.gate_up_proj.{suffix}",
                                    torch.cat((gate[2], up[2])),
                                )
                            )
                        else:
                            weights.extend(
                                [
                                    (
                                        f"{base}.shared_expert.gate_proj.{suffix}",
                                        gate[2],
                                    ),
                                    (f"{base}.shared_expert.up_proj.{suffix}", up[2]),
                                ]
                            )
                        weights.append(
                            (f"{base}.shared_expert.down_proj.{suffix}", down[2])
                        )
                        if reverse:
                            weights.reverse()
                        model.load_weights(weights)
                        torch.testing.assert_close(
                            getattr(experts, f"w13_{suffix}"),
                            torch.cat((gate, up), dim=1),
                        )
                        torch.testing.assert_close(
                            getattr(experts, f"w2_{suffix}"), down
                        )


if __name__ == "__main__":
    unittest.main()
