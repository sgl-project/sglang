"""Unit tests for RADIO checkpoint weight loading."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.models.radio import RadioModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingWeight:
    def __init__(self):
        self.loads = []

    def weight_loader(self, param, weight, shard_id=None):
        self.loads.append((param, weight, shard_id))


class TestRadioWeightLoading(CustomTestCase):
    def _make_model(self, named_parameters=()):
        model = object.__new__(RadioModel)
        nn.Module.__init__(model)
        model.named_parameters = lambda: iter(named_parameters)
        model.model = SimpleNamespace(
            patch_generator=SimpleNamespace(_video_embedder_loaded=False)
        )
        return model

    def test_hf_export_maps_embeddings_and_split_qkv(self):
        position_embedding = _RecordingWeight()
        qkv_weight = _RecordingWeight()
        model = self._make_model(
            [
                ("model.patch_generator.pos_embed", position_embedding),
                ("model.encoder.layers.0.attn.attn.qkv_proj.weight", qkv_weight),
            ]
        )

        position = torch.ones(1)
        query, key, value = (torch.full((1,), value) for value in (2, 3, 4))
        loaded = model.load_weights(
            [
                ("radio_model.hf_model.embeddings.position_embedding", position),
                (
                    "radio_model.hf_model.encoder.layer.0.attention.attention."
                    "query.weight",
                    query,
                ),
                (
                    "radio_model.hf_model.encoder.layer.0.attention.attention."
                    "key.weight",
                    key,
                ),
                (
                    "radio_model.hf_model.encoder.layer.0.attention.attention."
                    "value.weight",
                    value,
                ),
                ("radio_model.hf_model.summary_idxs", torch.tensor([0, 1])),
            ]
        )

        self.assertEqual(
            loaded,
            {
                "model.patch_generator.pos_embed",
                "model.encoder.layers.0.attn.attn.qkv_proj.weight",
            },
        )
        self.assertEqual(
            position_embedding.loads, [(position_embedding, position, None)]
        )
        self.assertEqual(
            qkv_weight.loads,
            [
                (qkv_weight, query, "q"),
                (qkv_weight, key, "k"),
                (qkv_weight, value, "v"),
            ],
        )

    def test_hf_export_loads_encoder_parameters(self):
        cases = {
            "embeddings.video_patch_projection.weight": (
                "model.patch_generator.video_embedder.weight"
            ),
            "encoder.layer.1.attention.output.dense.weight": (
                "model.encoder.layers.1.attn.attn.proj.weight"
            ),
            "encoder.layer.2.layer_scale1.lambda1": "model.encoder.layers.2.ls1",
            "encoder.layer.3.layer_scale2.lambda1": "model.encoder.layers.3.ls2",
            "encoder.layer.4.mlp.fc1.bias": "model.encoder.layers.4.mlp.fc1.bias",
            "encoder.layer.5.norm2.weight": "model.encoder.layers.5.norm2.weight",
        }
        for source, target in cases.items():
            with self.subTest(source=source):
                parameter = _RecordingWeight()
                model = self._make_model([(target, parameter)])
                weight = torch.ones(1)

                self.assertEqual(
                    model.load_weights([(f"radio_model.hf_model.{source}", weight)]),
                    {target},
                )
                self.assertEqual(parameter.loads, [(parameter, weight, None)])

    def test_unmapped_hf_export_weight_raises(self):
        model = self._make_model()

        with self.assertRaisesRegex(ValueError, "Unexpected HF RADIO weight"):
            model.load_weights(
                [("radio_model.hf_model.encoder.layer.0.unknown.weight", torch.ones(1))]
            )

    def test_legacy_unknown_weight_remains_ignored(self):
        model = self._make_model()

        self.assertEqual(
            model.load_weights([("radio_model.unknown.weight", torch.ones(1))]),
            set(),
        )


if __name__ == "__main__":
    unittest.main()
