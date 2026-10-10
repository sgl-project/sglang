# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
import tempfile
import unittest
from types import SimpleNamespace

from sglang.multimodal_gen.configs.models.dits.llada_image import (
    LLaDAImageArchConfig,
    editing_rope_rows,
)
from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.llada_image import (
    LLaDAImageSamplingParams,
    LLaDAImageTurboSamplingParams,
)
from sglang.multimodal_gen.registry import _get_config_info
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    RESIDENT,
)
from sglang.multimodal_gen.runtime.pipelines.llada_image import _check_fp8_transformer
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.llada_image.conditioning import (
    llada_image_text_encoder_quantization,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.utils import is_layer_skipped


class TestLLaDAImageConfig(unittest.TestCase):
    def test_base_and_turbo_checkpoints_get_their_own_defaults(self):
        """A default Base request must not run the 4-step Turbo recipe."""
        _get_config_info.cache_clear()
        cases = {
            "inclusionAI/LLaDA-Image": (LLaDAImageSamplingParams, 50, 5.0),
            "inclusionAI/LLaDA-Image-Turbo": (LLaDAImageTurboSamplingParams, 4, 1.0),
            "inclusionAI/LLaDA-Image-Turbo-FP8": (
                LLaDAImageTurboSamplingParams,
                4,
                1.0,
            ),
        }
        for model_id, (sampling_cls, steps, guidance) in cases.items():
            with self.subTest(model_id=model_id):
                info = _get_config_info(model_id)
                self.assertIs(info.sampling_param_cls, sampling_cls)
                params = info.sampling_param_cls()
                self.assertEqual(params.num_inference_steps, steps)
                self.assertEqual(params.guidance_scale, guidance)

    def test_editing_size_limit_follows_sequence_rope_table(self):
        """Editing sizes past the sequence RoPE table must be caught on the host."""
        # A short prompt of 20 tokens plus the 256 conditioning query tokens.
        caption_tokens = 20 + 256
        limit = LLaDAImageArchConfig().axes_lens[0]

        def rows(side: int) -> int:
            image_tokens = (side // 16) ** 2
            sigvq_tokens = math.ceil(side / 32) ** 2
            return editing_rope_rows(
                caption_tokens, [image_tokens, image_tokens], sigvq_tokens
            )

        self.assertLessEqual(rows(1904), limit)
        self.assertGreater(rows(1920), limit)

    def test_unservable_editing_sizes_are_rejected_at_validation(self):
        """Edits off the SigVQ grid or past the RoPE table fail before any GPU work."""
        config = LLaDAImagePipelineConfig()
        config.validate_output_size(1008, 1008, editing=False)
        config.validate_output_size(1024, 1024, editing=True)
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            config.validate_output_size(1008, 1008, editing=True)
        with self.assertRaisesRegex(ValueError, "sequence positions"):
            config.validate_output_size(2048, 2048, editing=True)

    def test_fp8_transformer_is_served_only_for_turbo(self):
        """The FP8 Base DiT costs text rendering, so only Turbo serves FP8."""
        pinned = []
        with tempfile.TemporaryDirectory() as path:
            with open(os.path.join(path, "config.json"), "w") as config_file:
                json.dump({"quantization_config": {"quant_method": "fp8"}}, config_file)
            for model_path, served in (
                ("inclusionAI/LLaDA-Image-Turbo-FP8", True),
                ("inclusionAI/LLaDA-Image-FP8", False),
            ):
                server_args = SimpleNamespace(
                    model_path=model_path,
                    backend=None,
                    model_id=None,
                    explicit_residency_mode=lambda name: None,
                    residency_mode=lambda name: RESIDENT,
                    require_component_resident=lambda name, feature_name: pinned.append(
                        name
                    ),
                )
                with self.subTest(model_path=model_path):
                    if served:
                        _check_fp8_transformer(server_args, path)
                    else:
                        with self.assertRaisesRegex(
                            ComponentCheckpointUnsupportedError, "Turbo"
                        ):
                            _check_fp8_transformer(server_args, path)
        self.assertEqual(pinned, ["transformer"])

    def test_fp8_text_encoder_quantizes_only_routed_experts(self):
        """BF16 encoder weights must never load into FP8 layers, which corrupts them."""
        config = {
            "first_k_dense_replace": 1,
            "llada_fp8_experts": {
                "enabled": True,
                "format": "e4m3fn",
                "weight_method": "symmetric_absmax_rtn",
                "weight_granularity": "per_expert_2d_block",
                "weight_block_size": [128, 128],
                "activation_method": "dynamic_absmax_rtn",
                "activation_granularity": "per_token",
            },
        }
        fp8 = Fp8Config.from_config(llada_image_text_encoder_quantization(config))
        self.assertTrue(fp8.is_checkpoint_fp8_serialized)
        self.assertEqual(fp8.weight_block_size, [128, 128])
        expected = {
            "model.layers.1.mlp.experts": False,
            "model.layers.10.mlp.experts": False,
            "model.layers.1.attention.query_key_value": True,
            "model.layers.1.attention.dense": True,
            "model.layers.1.mlp.shared_experts.gate_up_proj": True,
            "model.layers.1.mlp.shared_experts.down_proj": True,
            "model.layers.0.mlp.gate_up_proj": True,
            "model.layers.0.mlp.down_proj": True,
        }
        for prefix, skipped in expected.items():
            with self.subTest(prefix=prefix):
                self.assertIs(is_layer_skipped(prefix, fp8.ignored_layers), skipped)
        del config["llada_fp8_experts"]
        self.assertIsNone(llada_image_text_encoder_quantization(config))


if __name__ == "__main__":
    unittest.main()
