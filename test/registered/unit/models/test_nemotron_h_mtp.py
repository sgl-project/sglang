"""Unit tests for Nemotron-H MTP model behavior."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4A16LinearMethod,
)
from sglang.srt.models.nemotron_h_mtp import (
    NemotronHForCausalLMMTP,
    NemotronHMultiTokenPredictor,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs_embeds = None

    def forward(self, *, inputs_embeds, hidden_states, residual, forward_batch):
        self.inputs_embeds = inputs_embeds
        return hidden_states, residual


class TestNemotronHMultiTokenPredictor(CustomTestCase):
    def test_text_only_forward_uses_model_embeddings(self):
        model = object.__new__(NemotronHMultiTokenPredictor)
        nn.Module.__init__(model)
        model.embed_tokens = nn.Embedding(8, 2)
        model.embed_tokens.weight.data.copy_(torch.arange(16).reshape(8, 2))
        model.pattern_len = 1
        layer = _RecordingLayer()
        model.layers = nn.ModuleDict({"0": layer})
        input_ids = torch.tensor([1, 2, 3])
        forward_batch = SimpleNamespace(
            mm_input_embeds=None,
            forward_mode=SimpleNamespace(is_extend=lambda: False),
            contains_mm_inputs=lambda: False,
            spec_info=SimpleNamespace(hidden_states=torch.zeros(3, 2)),
        )

        model(
            input_ids=input_ids,
            positions=torch.arange(3),
            forward_batch=forward_batch,
        )

        torch.testing.assert_close(
            layer.inputs_embeds,
            model.embed_tokens(input_ids),
        )

    def test_multimodal_prefill_reuses_target_embeddings(self):
        model = object.__new__(NemotronHMultiTokenPredictor)
        nn.Module.__init__(model)
        model.embed_tokens = nn.Embedding(8, 2)
        model.embed_tokens.weight.data.copy_(torch.arange(16).reshape(8, 2))
        model.pattern_len = 1
        layer = _RecordingLayer()
        model.layers = nn.ModuleDict({"0": layer})

        target_embeddings = torch.tensor(
            [[101.0, 102.0], [103.0, 104.0], [105.0, 106.0]]
        )
        forward_batch = SimpleNamespace(
            mm_input_embeds=target_embeddings.clone(),
            forward_mode=SimpleNamespace(
                is_extend=lambda: True,
                is_draft_extend_v2=lambda: False,
            ),
            contains_mm_inputs=lambda: True,
            extend_start_loc=torch.tensor([0]),
            extend_seq_lens=torch.tensor([3]),
            spec_info=SimpleNamespace(hidden_states=torch.zeros(3, 2)),
        )

        model(
            input_ids=torch.tensor([100, 101, 2]),
            positions=torch.arange(3),
            forward_batch=forward_batch,
        )

        expected = target_embeddings.clone()
        expected[-1] = model.embed_tokens(torch.tensor(2))
        torch.testing.assert_close(layer.inputs_embeds, expected)


class TestNemotronHForCausalLMMTP(CustomTestCase):
    def _make_head_model(self):
        model = object.__new__(NemotronHForCausalLMMTP)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            max_n_routed_experts=0, tie_word_embeddings=False
        )
        model.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
        model.model = nn.Module()
        model.model.embed_tokens = nn.Embedding(4, 2)
        model.model.layers = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        model.lm_head = nn.Linear(2, 4, bias=False)
        model.lm_head.quant_method = None
        model.lm_head.register_parameter(
            "weight_scale", nn.Parameter(torch.zeros(1), requires_grad=False)
        )
        return model

    def test_standalone_mtp_head_survives_both_target_sharing_calls(self):
        # Replacing either the head weight or its module silently discards the
        # external checkpoint's output projection (including quantization scales).
        for prefix in ("", "language_model."):
            with self.subTest(prefix=prefix):
                model = self._make_head_model()
                model.load_weights(
                    iter(
                        [
                            (prefix + "mtp.layers.0.weight", torch.ones(2, 2)),
                            (
                                prefix + "lm_head.weight",
                                torch.arange(8.0).reshape(4, 2),
                            ),
                            (prefix + "lm_head.weight_scale", torch.tensor([0.5])),
                        ]
                    )
                )
                draft_head = model.lm_head
                draft_weight = draft_head.weight
                target_embed = nn.Parameter(torch.ones(4, 2))
                target_head = nn.Linear(2, 4, bias=False)
                with patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
                    model.set_embed_and_head(target_embed, target_head.weight)
                self.assertIs(model.lm_head.weight, draft_weight)
                model.set_lm_head_from_target(target_head)
                self.assertIs(model.lm_head, draft_head)
                self.assertIs(model.model.embed_tokens.weight, target_embed)
                torch.testing.assert_close(
                    model.lm_head(torch.ones(1, 2)),
                    torch.tensor([[1.0, 5.0, 9.0, 13.0]]),
                )
                torch.testing.assert_close(
                    model.lm_head.weight_scale, torch.tensor([0.5])
                )

    def test_embedded_and_headless_mtp_share_complete_target_head(self):
        for embedded in (False, True):
            with self.subTest(embedded=embedded):
                model = self._make_head_model()
                weights = [("mtp.layers.0.weight", torch.ones(2, 2))]
                if embedded:
                    # Full checkpoints also contain lm_head tensors; their
                    # presence alone must not opt out of embedded head sharing.
                    weights += [
                        ("lm_head.weight", torch.ones(4, 2)),
                        ("lm_head.weight_scale", torch.ones(1)),
                        ("backbone.layers.0.weight", torch.ones(2, 2)),
                    ]
                model.load_weights(iter(weights))
                target_head = nn.Linear(2, 4, bias=False)
                target_embed = nn.Parameter(torch.ones(4, 2))
                with patch("torch.cuda.synchronize"), patch("torch.cuda.empty_cache"):
                    model.set_embed_and_head(target_embed, target_head.weight)
                model.set_lm_head_from_target(target_head)
                self.assertIs(model.lm_head, target_head)
                self.assertIs(model.model.embed_tokens.weight, target_embed)

    def test_incomplete_standalone_head_is_rejected(self):
        for missing in ("weight", "weight_scale"):
            with self.subTest(missing=missing):
                model = self._make_head_model()
                weights = {
                    "mtp.layers.0.weight": torch.ones(2, 2),
                    "lm_head.weight": torch.ones(4, 2),
                    "lm_head.weight_scale": torch.ones(1),
                }
                del weights["lm_head." + missing]
                with self.assertRaisesRegex(
                    ValueError, "Incomplete standalone MTP lm_head"
                ):
                    model.load_weights(iter(weights.items()))

    def test_w4a16_head_does_not_require_unused_input_scale(self):
        model = self._make_head_model()
        model.lm_head.quant_method = ModelOptNvFp4A16LinearMethod(quant_config=None)
        model.lm_head.register_parameter(
            "input_scale", nn.Parameter(torch.zeros(1), requires_grad=False)
        )
        # NVFP4A16 registers this loader placeholder but discards it before
        # inference. Requiring it would reject valid standalone W4A16 heads.
        model.load_weights(
            iter(
                [
                    ("mtp.layers.0.weight", torch.ones(2, 2)),
                    ("lm_head.weight", torch.ones(4, 2)),
                    ("lm_head.weight_scale", torch.ones(1)),
                ]
            )
        )

    def test_maps_quantized_mtp_metadata(self):
        quant_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "language_model.mtp.layers.0.mixer.q_proj": {"quant_algo": "FP8"}
                },
            }
        )
        quant_config.apply_weight_name_mapper(
            NemotronHForCausalLMMTP.hf_to_sglang_mapper
        )

        self.assertEqual(
            quant_config._resolve_quant_algo("mtp.layers.0.mixer.q_proj"),
            "FP8",
        )


if __name__ == "__main__":
    unittest.main()
