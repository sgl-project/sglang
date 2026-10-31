import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.distributed import get_pp_indices
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models.qwen3_5 import (
    Qwen3_5MoeForConditionalGeneration,
    _refreshes_rope_cos_sin,
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


if __name__ == "__main__":
    unittest.main()
