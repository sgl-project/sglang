# SPDX-License-Identifier: Apache-2.0

import math
import unittest

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


if __name__ == "__main__":
    unittest.main()
