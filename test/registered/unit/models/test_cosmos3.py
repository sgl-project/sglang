"""Unit tests for the Cosmos3 reasoner (understanding tower).

Covers the three pieces of load-time logic that make the diffusers-layout
Cosmos3 checkpoint loadable through the Qwen3-VL inference stack:

1. ``Cosmos3ForConditionalGeneration.hf_to_sglang_mapper`` - renames the
   understanding-tower keys into the nested Qwen3-VL checkpoint form and drops
   the generation-tower weights.
2. ``Cosmos3Config`` - reuses the Qwen3-VL schema under ``model_type
   "cosmos3_omni"`` and is registered with ``AutoConfig``.
3. ``DefaultModelLoader`` ``allow_patterns_overrides`` - globs weights from the
   ``transformer/`` and ``vision_encoder/`` subfolders rather than the repo root.

All of this is pure CPU logic (no server / engine launch).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import safetensors.torch
import torch

from sglang.srt.configs import Cosmos3Config
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.cosmos3 import Cosmos3ForConditionalGeneration
from sglang.test.test_utils import CustomTestCase


class TestCosmos3WeightsMapper(CustomTestCase):
    """Validate the understanding-tower rename + generation-tower drop rules."""

    def setUp(self):
        self.mapper = Cosmos3ForConditionalGeneration.hf_to_sglang_mapper

    def test_understanding_tower_is_renamed(self):
        # Flat Cosmos3 keys -> nested Qwen3-VL checkpoint keys. The parent
        # Qwen3VLForConditionalGeneration.load_weights then strips the
        # `model.language_model.`/`model.visual.` prefixes and fuses q/k/v.
        inputs = [
            "layers.0.self_attn.to_q.weight",
            "layers.0.self_attn.to_k.weight",
            "layers.0.self_attn.to_v.weight",
            "layers.0.self_attn.to_out.weight",
            "layers.0.self_attn.norm_q.weight",
            "layers.0.self_attn.norm_k.weight",
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
            "layers.0.mlp.down_proj.weight",
            "layers.0.input_layernorm.weight",
            "embed_tokens.weight",
            "norm.weight",
            "lm_head.weight",
        ]
        expected = [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.0.self_attn.k_proj.weight",
            "model.language_model.layers.0.self_attn.v_proj.weight",
            "model.language_model.layers.0.self_attn.o_proj.weight",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            "model.language_model.layers.0.self_attn.k_norm.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.embed_tokens.weight",
            "model.language_model.norm.weight",
            "lm_head.weight",
        ]
        self.assertEqual(self.mapper.apply_list(inputs), expected)

    def test_vision_encoder_is_prefixed(self):
        inputs = [
            "blocks.0.attn.qkv.weight",
            "merger.norm.weight",
            "patch_embed.proj.weight",
            "pos_embed.weight",
            "deepstack_merger_list.0.norm.weight",
        ]
        expected = [
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.merger.norm.weight",
            "model.visual.patch_embed.proj.weight",
            "model.visual.pos_embed.weight",
            "model.visual.deepstack_merger_list.0.norm.weight",
        ]
        self.assertEqual(self.mapper.apply_list(inputs), expected)

    def test_generation_tower_is_dropped(self):
        dropped = [
            "layers.0.self_attn.add_q_proj.weight",
            "layers.0.self_attn.add_k_proj.weight",
            "layers.0.self_attn.add_v_proj.weight",
            "layers.0.self_attn.to_add_out.weight",
            "layers.0.self_attn.norm_added_q.weight",
            "layers.0.self_attn.norm_added_k.weight",
            "layers.0.self_attn.q_proj_moe_gen.weight",
            "layers.0.mlp_moe_gen.gate_up_proj.weight",
            "norm_moe_gen.weight",
            "proj_in.weight",
            "proj_out.weight",
            "time_embedder.linear_1.weight",
            "audio_proj_in.weight",
            "audio_proj_out.weight",
            "action_proj_in.weight",
            "action_proj_out.weight",
            "audio_modality_embed",
            "action_modality_embed",
        ]
        self.assertEqual(self.mapper.apply_list(dropped), [])

    def test_moe_gen_substring_wins_over_norm_prefix(self):
        # `norm_moe_gen.weight` must be dropped (generation), not routed to the
        # final `norm.` -> language-model norm.
        self.assertEqual(self.mapper.apply_list(["norm_moe_gen.weight"]), [])
        self.assertEqual(
            self.mapper.apply_list(["norm.weight"]),
            ["model.language_model.norm.weight"],
        )


class TestCosmos3Config(CustomTestCase):
    def test_model_type(self):
        self.assertEqual(Cosmos3Config.model_type, "cosmos3_omni")

    def test_subconfigs_are_objects(self):
        # The Qwen3-VL inference stack reads sub-configs as objects, e.g.
        # `config.vision_config.hidden_size` and
        # `config.vision_config.deepstack_visual_indexes`. Some transformers
        # versions leave sub-configs as raw dicts after construction, which would
        # raise `'dict' object has no attribute 'hidden_size'` at model init.
        # Cosmos3Config coerces them into config objects, so assert that here.
        cfg = Cosmos3Config(
            text_config={
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
            },
            vision_config={"depth": 3, "hidden_size": 64},
        )
        self.assertNotIsInstance(cfg.text_config, dict)
        self.assertNotIsInstance(cfg.vision_config, dict)
        self.assertEqual(cfg.text_config.hidden_size, 128)
        self.assertEqual(cfg.text_config.num_hidden_layers, 2)
        self.assertEqual(cfg.text_config.num_attention_heads, 4)
        self.assertEqual(cfg.vision_config.hidden_size, 64)
        self.assertEqual(cfg.vision_config.depth, 3)

    def test_registered_with_autoconfig(self):
        # Importing common runs the AutoConfig registration side effects.
        from transformers import AutoConfig

        import sglang.srt.utils.hf_transformers.common  # noqa: F401

        cfg = AutoConfig.for_model(
            "cosmos3_omni",
            text_config={"hidden_size": 128, "num_hidden_layers": 2},
        )
        self.assertIsInstance(cfg, Cosmos3Config)


class TestCosmos3MropeIndex(CustomTestCase):
    """Cosmos3 reuses the Qwen3-VL mrope path.

    The multimodal processor calls ``MRotaryEmbedding.get_rope_index`` with the
    config's ``model_type``. Because Cosmos3 declares its own ``cosmos3_omni``
    type (for AutoConfig resolution), the mrope dispatch must recognize it as a
    Qwen3-VL-family model or it raises ``RuntimeError: Unimplemented model type:
    cosmos3_omni``. This guards that regression.
    """

    def _get_rope_index(self, model_type):
        from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

        # A single image: grid [t=1, h=2, w=2] with spatial_merge_size 2 expands
        # to exactly one image placeholder token after the vision-start token.
        input_ids = torch.tensor([[1, 99, 100, 2]], dtype=torch.long)
        image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
        return MRotaryEmbedding.get_rope_index(
            spatial_merge_size=2,
            image_token_id=100,
            video_token_id=101,
            vision_start_token_id=99,
            model_type=model_type,
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
        )

    def test_cosmos3_omni_is_supported(self):
        # Must not raise "Unimplemented model type".
        positions, delta = self._get_rope_index("cosmos3_omni")
        self.assertEqual(positions.shape[0], 3)

    def test_cosmos3_matches_qwen3_vl(self):
        # Cosmos3 must produce identical rope indices to the qwen3_vl path it
        # reuses, so behavior can't silently diverge.
        pos_cosmos, delta_cosmos = self._get_rope_index("cosmos3_omni")
        pos_qwen, delta_qwen = self._get_rope_index("qwen3_vl")
        self.assertTrue(torch.equal(pos_cosmos, pos_qwen))
        self.assertTrue(torch.equal(delta_cosmos, delta_qwen))


class TestAllowPatternsOverrides(CustomTestCase):
    """Validate DefaultModelLoader subfolder globbing for diffusers layouts."""

    def _make_checkpoint(self, root):
        os.makedirs(os.path.join(root, "transformer"))
        os.makedirs(os.path.join(root, "vision_encoder"))
        # A root-level file that must be ignored when an override is given.
        safetensors.torch.save_file(
            {"root": torch.zeros(2)}, os.path.join(root, "model.safetensors")
        )
        safetensors.torch.save_file(
            {"llm": torch.zeros(2)},
            os.path.join(root, "transformer", "diffusion_pytorch_model.safetensors"),
        )
        safetensors.torch.save_file(
            {"vit": torch.zeros(2)},
            os.path.join(root, "vision_encoder", "model.safetensors"),
        )

    @patch("sglang.srt.model_loader.loader.get_global_server_args")
    def test_override_selects_transformer_subfolder(self, mock_gsa):
        mock_gsa.return_value = MagicMock(model_checksum=None)
        loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.AUTO))
        with tempfile.TemporaryDirectory() as root:
            self._make_checkpoint(root)
            _, files, use_safetensors = loader._prepare_weights(
                root, None, True, ["transformer/*.safetensors"]
            )
            self.assertTrue(use_safetensors)
            self.assertEqual(len(files), 1)
            self.assertTrue(
                files[0].endswith("transformer/diffusion_pytorch_model.safetensors")
            )

    @patch("sglang.srt.model_loader.loader.get_global_server_args")
    def test_override_selects_vision_subfolder(self, mock_gsa):
        mock_gsa.return_value = MagicMock(model_checksum=None)
        loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.AUTO))
        with tempfile.TemporaryDirectory() as root:
            self._make_checkpoint(root)
            _, files, _ = loader._prepare_weights(
                root, None, True, ["vision_encoder/*.safetensors"]
            )
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].endswith("vision_encoder/model.safetensors"))

    @patch("sglang.srt.model_loader.loader.get_global_server_args")
    def test_no_override_globs_repo_root(self, mock_gsa):
        mock_gsa.return_value = MagicMock(model_checksum=None)
        loader = DefaultModelLoader(LoadConfig(load_format=LoadFormat.AUTO))
        with tempfile.TemporaryDirectory() as root:
            self._make_checkpoint(root)
            _, files, _ = loader._prepare_weights(root, None, True)
            # Without an override only the root-level file is discovered.
            self.assertEqual(
                [os.path.basename(f) for f in files], ["model.safetensors"]
            )


if __name__ == "__main__":
    unittest.main()
