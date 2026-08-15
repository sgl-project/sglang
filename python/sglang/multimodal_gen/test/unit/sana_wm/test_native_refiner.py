# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.runtime.models.adapter.ltx_2_connector import (
    LTX2ConnectorFeedForward,
    LTX2ConnectorTransformer1d,
    LTX2TextConnectors,
)
from sglang.multimodal_gen.runtime.models.dits.sana_wm_refiner_transformer import (
    SanaWMLTX2VideoRefiner,
)
from sglang.multimodal_gen.runtime.models.encoders.gemma_3 import (
    Gemma3TextEncoder,
    gemma3_text_weights,
)


class _FakeRope:
    def prepare_video_coords(self, **kwargs):
        return torch.zeros(1, 3, 1, 2, device=kwargs["device"])

    def __call__(self, coords, **kwargs):
        return coords, coords


class _FakeRefiner:
    patch_size = 1
    patch_size_t = 1
    rope = _FakeRope()

    def forward_tokens(self, **kwargs):
        return kwargs["hidden_states"] + 1


class _Float32Rope(torch.nn.Module):
    def forward(self, batch_size, seq_len, device):
        frequencies = torch.zeros(
            batch_size, 1, seq_len, 1, device=device, dtype=torch.float32
        )
        return frequencies, frequencies


class TestSanaWMNativeRefiner(unittest.TestCase):
    def test_forward_preserves_packed_input_contract(self) -> None:
        packed = torch.zeros(1, 6, 2)
        output = SanaWMLTX2VideoRefiner.forward(
            _FakeRefiner(),
            hidden_states=packed,
            encoder_hidden_states=torch.empty(1, 0, 1),
            timestep=torch.zeros(1, 6),
            num_frames=2,
            height=1,
            width=3,
        )

        self.assertEqual(output.shape, packed.shape)
        self.assertTrue(torch.equal(output, packed + 1))

    def test_forward_preserves_5d_input_contract(self) -> None:
        latents = torch.zeros(1, 2, 2, 1, 3)
        output = SanaWMLTX2VideoRefiner.forward(
            _FakeRefiner(),
            hidden_states=latents,
            encoder_hidden_states=torch.empty(1, 0, 1),
            timestep=torch.zeros(1, 6),
        )

        self.assertEqual(output.shape, latents.shape)
        self.assertTrue(torch.equal(output, latents + 1))

    def test_refiner_components_expose_layerwise_blocks(self) -> None:
        self.assertEqual(SanaWMLTX2VideoRefiner.layer_names, ["transformer_blocks"])

    def test_refiner_only_ignores_released_audio_checkpoint_branches(self) -> None:
        expected_unused = (
            "audio_proj_in.weight",
            "av_cross_attn_video_scale_shift.linear.weight",
            "transformer_blocks.0.audio_attn1.to_q.weight",
            "transformer_blocks.0.audio_to_video_attn.to_q.weight",
            "transformer_blocks.0.video_to_audio_attn.to_q.weight",
            "transformer_blocks.0.video_a2v_cross_attn_scale_shift_table",
        )
        for name in expected_unused:
            self.assertTrue(
                SanaWMLTX2VideoRefiner.is_expected_unloaded_checkpoint_key(name)
            )
        self.assertFalse(
            SanaWMLTX2VideoRefiner.is_expected_unloaded_checkpoint_key(
                "transformer_blocks.0.attn1.to_q.weight"
            )
        )
        self.assertFalse(
            SanaWMLTX2VideoRefiner.is_expected_unloaded_checkpoint_key(
                "unexpected.weight"
            )
        )

    def test_streaming_prefixes_require_one_entry_per_block(self) -> None:
        refiner = SimpleNamespace(
            transformer_blocks=[
                SimpleNamespace(attn1=SimpleNamespace(kv_prefix=None)),
                SimpleNamespace(attn1=SimpleNamespace(kv_prefix=None)),
            ]
        )
        prefixes = [{"mode": "rf_shifted_sink"}, None]

        SanaWMLTX2VideoRefiner.set_streaming_kv_prefixes(refiner, prefixes)
        self.assertIs(refiner.transformer_blocks[0].attn1.kv_prefix, prefixes[0])
        self.assertIsNone(refiner.transformer_blocks[1].attn1.kv_prefix)

        SanaWMLTX2VideoRefiner.clear_streaming_kv_prefixes(refiner)
        self.assertTrue(
            all(block.attn1.kv_prefix is None for block in refiner.transformer_blocks)
        )
        with self.assertRaisesRegex(ValueError, "one KV prefix per refiner block"):
            SanaWMLTX2VideoRefiner.set_streaming_kv_prefixes(refiner, prefixes[:1])
        self.assertEqual(Gemma3TextEncoder.layer_names, ["layers"])
        self.assertEqual(
            LTX2TextConnectors.layer_names,
            [
                "video_connector.transformer_blocks",
                "audio_connector.transformer_blocks",
            ],
        )

    def test_connector_feed_forward_keeps_checkpoint_key_layout(self) -> None:
        state_keys = set(LTX2ConnectorFeedForward(4, "gelu-approximate").state_dict())
        self.assertEqual(
            state_keys,
            {
                "net.0.proj.weight",
                "net.0.proj.bias",
                "net.2.weight",
                "net.2.bias",
            },
        )

    def test_connector_keeps_rope_frequencies_in_float32(self) -> None:
        connector = object.__new__(LTX2ConnectorTransformer1d)
        torch.nn.Module.__init__(connector)
        connector.learnable_registers = None
        connector.rope = _Float32Rope()
        connector.transformer_blocks = torch.nn.ModuleList()
        connector.norm_out = torch.nn.Identity()
        connector.gradient_checkpointing = False

        hidden_states = torch.zeros(1, 4, 2, dtype=torch.bfloat16)
        output, _ = connector(hidden_states)

        self.assertIs(output, hidden_states)

    def test_text_encoder_selects_only_language_backbone_weights(self) -> None:
        weights = [
            ("model.language_model.model.layers.0.weight", torch.ones(1)),
            ("model.vision_tower.weight", torch.ones(1)),
            ("language_model.model.norm.weight", torch.ones(1)),
        ]

        self.assertEqual(
            [name for name, _ in gemma3_text_weights(weights)],
            ["layers.0.weight", "norm.weight"],
        )


if __name__ == "__main__":
    unittest.main()
