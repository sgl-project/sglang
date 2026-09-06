"""Streaming and completeness checks for DeepSeek-V4 vision weights."""

import unittest
from itertools import chain

import torch
from torch import nn

from sglang.srt.models.deepseek_v4_vl import DeepseekV4ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


class RecordingLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.names = []

    def load_weights(self, weights):
        for name, _ in weights:
            self.names.append(name)


class TestDeepseekV4VisionWeights(CustomTestCase):
    def make_model(self, multimodal=True):
        model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
        nn.Module.__init__(model)
        model.is_multimodal = multimodal
        model._vision_weights_loaded = False
        model.language_model = RecordingLanguageModel()
        if multimodal:
            model.vision = nn.Linear(2, 2, bias=False)
            model.aligner = nn.Linear(2, 2, bias=False)
            for name in model._SENTINEL_NAMES:
                model.register_parameter(name, nn.Parameter(torch.empty(2)))
        return model

    def vision_weights(self, model):
        return [(name, torch.ones_like(p)) for name, p in model.named_parameters()]

    def test_streams_language_weights_across_checkpoint_shards(self):
        model = self.make_model()
        vision = self.vision_weights(model)

        def shards():
            yield iter([("model.first", torch.zeros(1)), *vision[:2]])
            self.assertEqual(model.language_model.names, ["model.first"])
            yield iter([("model.second", torch.zeros(1)), *vision[2:]])

        model.load_weights(chain.from_iterable(shards()))
        self.assertEqual(model.language_model.names, ["model.first", "model.second"])
        for _, param in model.named_parameters():
            torch.testing.assert_close(param, torch.ones_like(param))

    def test_missing_required_weights_fail(self):
        for missing in ("vision.weight", "aligner.weight", "image_start"):
            with self.subTest(missing=missing):
                model = self.make_model()
                weights = [
                    (n, w) for n, w in self.vision_weights(model) if n != missing
                ]
                with self.assertRaisesRegex(ValueError, missing):
                    model.load_weights(iter(weights))
                self.assertFalse(model._vision_weights_loaded)

    def test_text_only_checkpoint(self):
        model = self.make_model(multimodal=False)
        model.load_weights(iter([("model.weight", torch.zeros(1))]))
        self.assertEqual(model.language_model.names, ["model.weight"])

    def test_language_only_update_after_initial_load(self):
        model = self.make_model()
        model.load_weights(iter(self.vision_weights(model)))
        model.load_weights(iter([("model.weight", torch.zeros(1))]))
        self.assertEqual(model.language_model.names, ["model.weight"])
        self.assertTrue(model._vision_weights_loaded)


if __name__ == "__main__":
    unittest.main()
