import logging
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.distributed import get_pp_indices
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
    _refreshes_rope_cos_sin,
)
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP


class _RecordingParam(torch.nn.Parameter):
    def __new__(cls, data=None):
        if data is None:
            data = torch.zeros(1)
        instance = super().__new__(cls, data, requires_grad=False)
        instance.loaded_weights = []
        return instance

    def weight_loader(self, param, loaded_weight, shard_id=None):
        self.loaded_weights.append((loaded_weight, shard_id))


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

    def test_rope_refresh_at_first_attention_layer_of_each_pp_stage(self):
        """Each stage must refresh the shared rope cos/sin exactly once per forward.
        Refreshing only at the model's first attention layer left later stages
        without one."""
        # (layers, interval, pp_size, SGLANG_PP_LAYER_PARTITION)
        cases = [
            (48, 4, 1, None),
            (48, 4, 2, None),
            (48, 4, 4, None),
            (40, 4, 3, None),  # uneven split: 13/13/14
            (48, 4, 3, "7,16,25"),  # stages start on attention layers
        ]
        for num_layers, interval, pp_size, partition in cases:
            config = Qwen3_5TextConfig(
                num_hidden_layers=num_layers, full_attention_interval=interval
            )
            env = {} if partition is None else {"SGLANG_PP_LAYER_PARTITION": partition}
            for pp_rank in range(pp_size):
                with (
                    self.subTest(
                        num_layers=num_layers,
                        pp_size=pp_size,
                        pp_rank=pp_rank,
                        partition=partition,
                    ),
                    patch.dict(os.environ, env),
                    patch(
                        "sglang.srt.models.qwen3_5.get_parallel",
                        return_value=SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size),
                    ),
                ):
                    start, end = get_pp_indices(num_layers, pp_rank, pp_size)
                    attn_layers = [
                        i
                        for i in range(start, end)
                        if config.layers_block_type[i] == "attention"
                    ]
                    refreshing = [
                        i
                        for i in attn_layers
                        if _refreshes_rope_cos_sin(config, i, is_nextn=False)
                    ]
                    self.assertEqual(refreshing, attn_layers[:1])

    def test_rope_refresh_for_draft_ignores_target_pp_split(self):
        # The MTP draft's single layer is numbered 0 whatever stage it runs on.
        config = Qwen3_5TextConfig(num_hidden_layers=1, full_attention_interval=1)
        with (
            patch.dict(os.environ, {"SGLANG_PP_LAYER_PARTITION": "24,24"}),
            patch(
                "sglang.srt.models.qwen3_5.get_parallel",
                return_value=SimpleNamespace(pp_rank=1, pp_size=2),
            ),
        ):
            self.assertTrue(_refreshes_rope_cos_sin(config, 0, is_nextn=True))

    @staticmethod
    def _make_conditional_generation_stub(quant_config=None, params=None):
        model = Qwen3_5ForConditionalGeneration.__new__(Qwen3_5ForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            tie_word_embeddings=False,
            num_hidden_layers=1,
            num_experts=None,
        )
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.quant_config = quant_config
        model.pooler = None
        params = params or []
        model.named_parameters = lambda remove_duplicate=False: params
        return model

    def test_unknown_bias_warning_on_vision_weights(self):
        fc1_w = _RecordingParam()
        fc1_b = _RecordingParam()
        model = self._make_conditional_generation_stub(
            quant_config=None,
            params=[
                ("visual.merger.linear_fc1.weight", fc1_w),
                ("visual.merger.linear_fc1.bias", fc1_b),
            ],
        )

        warnings = []
        logger = logging.getLogger("sglang.srt.models.qwen3_5")

        class Handler(logging.Handler):
            def emit(self, record):
                warnings.append(record.getMessage())

        handler = Handler()
        logger.addHandler(handler)
        try:
            weights = [
                ("visual.merger.linear_fc1.weight", torch.tensor([1.0])),
                ("visual.merger.linear_fc1.bias", torch.tensor([2.0])),
                ("visual.merger.linear_fc2_renamed.weight", torch.tensor([3.0])),
                ("visual.merger.linear_fc2_renamed.bias", torch.tensor([4.0])),
            ]
            loaded = model.load_weights(weights)
            self.assertIn("visual.merger.linear_fc1.weight", loaded)
            self.assertIn("visual.merger.linear_fc1.bias", loaded)
            self.assertNotIn("visual.merger.linear_fc2_renamed.weight", loaded)
            self.assertNotIn("visual.merger.linear_fc2_renamed.bias", loaded)
            self.assertEqual(len(fc1_w.loaded_weights), 1)
            self.assertEqual(len(fc1_b.loaded_weights), 1)
            self.assertTrue(
                any("linear_fc2_renamed.bias" in w for w in warnings),
                "Missing warning for unknown visual.merger.*.bias parameter",
            )
            self.assertTrue(
                any("linear_fc2_renamed.weight" in w for w in warnings),
                "Missing warning for unknown visual.merger.*.weight parameter",
            )
        finally:
            logger.removeHandler(handler)

    def test_gptq_skips_non_visual_bias_silently(self):
        model = self._make_conditional_generation_stub(
            quant_config=SimpleNamespace(quant_method="gptq"),
            params=[],
        )

        warnings = []
        logger = logging.getLogger("sglang.srt.models.qwen3_5")

        class Handler(logging.Handler):
            def emit(self, record):
                warnings.append(record.getMessage())

        handler = Handler()
        logger.addHandler(handler)
        try:
            weights = [
                ("visual.merger.unknown.bias", torch.tensor([1.0])),
                ("model.layers.0.extra.bias", torch.tensor([2.0])),
            ]
            model.load_weights(weights)
            # visual bias should warn even on GPTQ
            self.assertTrue(
                any("visual.merger.unknown.bias" in w for w in warnings),
                "Visual bias must warn even when quant_method == gptq",
            )
            # non-visual bias is skipped silently for GPTQ
            self.assertFalse(
                any("model.layers.0.extra.bias" in w for w in warnings),
                "Non-visual bias on GPTQ must be skipped without warning",
            )
        finally:
            logger.removeHandler(handler)

    def test_moe_unknown_bias_warning_on_vision_weights(self):
        model = Qwen3_5MoeForConditionalGeneration.__new__(
            Qwen3_5MoeForConditionalGeneration
        )
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(
            tie_word_embeddings=False,
            num_hidden_layers=1,
            num_experts=8,
            encoder_only=False,
        )
        model.enable_shared_expert_fusion = False
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.quant_config = None
        model.pooler = None
        model.model = SimpleNamespace(start_layer=0, end_layer=0, layers=[])
        model.named_parameters = lambda remove_duplicate=False: []

        warnings = []
        logger = logging.getLogger("sglang.srt.models.qwen3_5")

        class Handler(logging.Handler):
            def emit(self, record):
                warnings.append(record.getMessage())

        handler = Handler()
        logger.addHandler(handler)
        try:
            weights = [
                ("visual.merger.unknown_fc.bias", torch.tensor([1.0])),
            ]
            model.load_weights(weights)
            self.assertTrue(
                any("visual.merger.unknown_fc.bias" in w for w in warnings),
                "Missing warning for unknown visual bias in MoE model",
            )
        finally:
            logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
