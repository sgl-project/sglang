"""Unit tests for Nemotron-H MTP model behavior."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptMixedPrecisionConfig,
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
