import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.models.encoders.base import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    WanT2V480PConfig,
    t5_postprocess_text,
)


class TestWanT5PostprocessText(unittest.TestCase):
    def test_returns_real_per_request_lengths(self) -> None:
        hidden_state = torch.randn(2, 512, 8)
        mask = torch.zeros(2, 512, dtype=torch.long)
        mask[0, :45] = 1
        mask[1, :30] = 1
        outputs = BaseEncoderOutput(last_hidden_state=hidden_state, attention_mask=mask)

        result = t5_postprocess_text(outputs, None)

        self.assertIsInstance(result, TextConditioningOutput)
        self.assertEqual(result.prompt_seq_lens, [45, 30])
        self.assertEqual(result.prompt_embeds.shape, (2, 512, 8))
        self.assertTrue(bool(result.prompt_embeds_mask[0, 44]))
        self.assertFalse(bool(result.prompt_embeds_mask[0, 45]))
        self.assertTrue(bool(result.prompt_embeds_mask[1, 29]))
        self.assertFalse(bool(result.prompt_embeds_mask[1, 30]))


class TestWanPrepareCondKwargs(unittest.TestCase):
    def _make_batch(self, prompt_embeds, prompt_seq_lens):
        return SimpleNamespace(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=prompt_embeds,
            prompt_seq_lens=prompt_seq_lens,
            negative_prompt_seq_lens=prompt_seq_lens,
        )

    def test_uniform_lengths_need_no_mask(self) -> None:
        config = WanT2V480PConfig()
        prompt_embeds = [torch.randn(2, 512, 8)]
        batch = self._make_batch(prompt_embeds, [[512, 512]])

        cond_kwargs = config.prepare_pos_cond_kwargs(
            batch, device=torch.device("cpu"), rotary_emb=None, dtype=torch.float32
        )

        self.assertIsNone(cond_kwargs["encoder_hidden_states_mask"])

    def test_ragged_lengths_build_boundary_mask(self) -> None:
        config = WanT2V480PConfig()
        prompt_embeds = [torch.randn(2, 512, 8)]
        batch = self._make_batch(prompt_embeds, [[45, 30]])

        cond_kwargs = config.prepare_pos_cond_kwargs(
            batch, device=torch.device("cpu"), rotary_emb=None, dtype=torch.float32
        )

        mask = cond_kwargs["encoder_hidden_states_mask"]
        self.assertEqual(mask.shape, (2, 512))
        self.assertTrue(bool(mask[0, 44]))
        self.assertFalse(bool(mask[0, 45]))
        self.assertTrue(bool(mask[1, 29]))
        self.assertFalse(bool(mask[1, 30]))

    def test_negative_branch_reads_negative_seq_lens(self) -> None:
        config = WanT2V480PConfig()
        prompt_embeds = [torch.randn(2, 512, 8)]
        batch = SimpleNamespace(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=prompt_embeds,
            prompt_seq_lens=[[512, 512]],
            negative_prompt_seq_lens=[[10, 512]],
        )

        cond_kwargs = config.prepare_neg_cond_kwargs(
            batch, device=torch.device("cpu"), rotary_emb=None, dtype=torch.float32
        )

        mask = cond_kwargs["encoder_hidden_states_mask"]
        self.assertFalse(bool(mask[0, 10]))
        self.assertTrue(bool(mask[1].all()))


if __name__ == "__main__":
    unittest.main()
