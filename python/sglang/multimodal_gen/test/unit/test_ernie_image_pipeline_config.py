import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.configs.pipeline_configs.ernie_image import (
    ErnieImagePipelineConfig,
    ernie_image_postprocess_text,
)


class TestErnieImagePostprocessText(unittest.TestCase):
    def test_single_request_returns_full_length_conditioning(self) -> None:
        hidden_states = torch.randn(1, 5, 4)
        outputs = BaseEncoderOutput(hidden_states=(hidden_states, hidden_states))
        text_inputs = SimpleNamespace(attention_mask=torch.ones(1, 5, dtype=torch.long))

        result = ernie_image_postprocess_text(outputs, text_inputs)

        self.assertIsInstance(result, TextConditioningOutput)
        self.assertEqual(result.prompt_seq_lens, [5])
        self.assertEqual(result.prompt_embeds.shape, (1, 5, 4))
        self.assertTrue(bool(result.prompt_embeds_mask.all()))

    def test_ragged_batch_preserves_real_lengths(self) -> None:
        hidden_states = torch.randn(2, 6, 4)
        outputs = BaseEncoderOutput(hidden_states=(hidden_states, hidden_states))
        # Row 0 has 4 real tokens + 2 pad, row 1 has 6 real tokens (no pad).
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        text_inputs = SimpleNamespace(attention_mask=mask)

        result = ernie_image_postprocess_text(outputs, text_inputs)

        self.assertIsInstance(result, TextConditioningOutput)
        self.assertEqual(result.prompt_seq_lens, [4, 6])
        self.assertEqual(result.prompt_embeds.shape, (2, 6, 4))
        self.assertEqual(
            result.prompt_embeds_mask.tolist(),
            [[True, True, True, True, False, False], [True] * 6],
        )


class TestErnieImagePrepareCondKwargs(unittest.TestCase):
    def _make_batch(self, prompt_embeds, prompt_seq_lens):
        return SimpleNamespace(
            height=256,
            width=256,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=prompt_embeds,
            prompt_seq_lens=prompt_seq_lens,
            negative_prompt_seq_lens=prompt_seq_lens,
        )

    def test_uniform_lengths_need_no_mask(self) -> None:
        config = ErnieImagePipelineConfig()
        prompt_embeds = [torch.randn(2, 30, 4)]
        batch = self._make_batch(prompt_embeds, [[30, 30]])

        cond_kwargs = config.prepare_pos_cond_kwargs(
            batch, device=torch.device("cpu"), rotary_emb=None, dtype=torch.float32
        )

        self.assertIsNone(cond_kwargs["encoder_hidden_states_mask"])
        self.assertEqual(cond_kwargs["txt_seq_lens"], [30, 30])

    def test_ragged_lengths_build_boundary_mask(self) -> None:
        config = ErnieImagePipelineConfig()
        prompt_embeds = [torch.randn(2, 30, 4)]
        batch = self._make_batch(prompt_embeds, [[18, 30]])

        cond_kwargs = config.prepare_pos_cond_kwargs(
            batch, device=torch.device("cpu"), rotary_emb=None, dtype=torch.float32
        )

        mask = cond_kwargs["encoder_hidden_states_mask"]
        self.assertEqual(mask.shape, (2, 30))
        self.assertTrue(bool(mask[0, 17]))
        self.assertFalse(bool(mask[0, 18]))
        self.assertTrue(bool(mask[1].all()))


if __name__ == "__main__":
    unittest.main()
