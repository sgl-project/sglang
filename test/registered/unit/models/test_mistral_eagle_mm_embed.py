"""Mistral EAGLE draft bodies must not call embed_tokens on MM sentinel ids.

Same contract as DeepseekModelNextN (sgl-project/sglang#37548): multimodal
positions in input_ids hold MM_PAD_SHIFT_VALUE+hash sentinels far above
vocab_size, so the draft must read forward_batch.mm_input_embeds for them and
only embed the appended next-token, instead of a CUDA index-out-of-bounds
gather. Covers MistralEagleModel (Pixtral / GQA targets) and
MistralLarge3EagleModel.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.req_types import MM_PAD_SHIFT_VALUE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

VOCAB_SIZE = 131072
HIDDEN_SIZE = 64  # tiny for CPU test


def _make_forward_batch(
    input_ids: torch.Tensor,
    mm_input_embeds: torch.Tensor = None,
    extend_seq_lens: torch.Tensor = None,
    extend_start_loc: torch.Tensor = None,
    has_mm: bool = True,
):
    """Build a minimal mock ForwardBatch for the draft forward."""
    fb = MagicMock()
    fb.mm_input_embeds = mm_input_embeds
    fb.contains_mm_inputs.return_value = has_mm
    fb.forward_mode.is_extend.return_value = True
    fb.forward_mode.is_draft_extend_v2.return_value = False
    fb.forward_mode.is_idle.return_value = False
    fb.extend_seq_lens = extend_seq_lens
    fb.extend_start_loc = extend_start_loc
    fb.spec_info.hidden_states = torch.randn(input_ids.shape[0], HIDDEN_SIZE)
    return fb


def _make_mistral_eagle_model(vocab_size: int, hidden_size: int):
    """Build a mock MistralEagleModel with a real embed_tokens layer."""
    from sglang.srt.models.mistral_eagle import MistralEagleModel

    model = MistralEagleModel.__new__(MistralEagleModel)
    torch.nn.Module.__init__(model)
    model.vocab_size = vocab_size
    model.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
    model.fc = MagicMock(side_effect=lambda x: (x[..., :hidden_size], None))
    model.layers = torch.nn.ModuleList()
    return model


def _make_mistral_large_3_eagle_model(vocab_size: int, hidden_size: int):
    """Build a mock MistralLarge3EagleModel with a real embed_tokens layer."""
    from sglang.srt.models.mistral_large_3_eagle import MistralLarge3EagleModel

    model = MistralLarge3EagleModel.__new__(MistralLarge3EagleModel)
    torch.nn.Module.__init__(model)
    model.vocab_size = vocab_size
    model.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
    model.fc = MagicMock(side_effect=lambda x: (x[..., :hidden_size], None))
    return model


def _run_forward_capturing_embed_calls(model, input_ids: torch.Tensor, fb):
    """Run forward; return embed_tokens call args (later stages may raise)."""
    mock_embed = MagicMock(side_effect=model.embed_tokens)
    object.__setattr__(model, "embed_tokens", mock_embed)
    positions = torch.arange(input_ids.shape[0], dtype=torch.long)
    try:
        model.forward(input_ids, positions, fb)
    except Exception:
        pass  # We only care about embed_tokens call args
    return mock_embed.call_args_list


def _sentinel_input(num_tokens: int, mm_start: int, mm_end: int) -> torch.Tensor:
    input_ids = torch.arange(num_tokens, dtype=torch.long)
    for i in range(mm_start, mm_end):
        input_ids[i] = MM_PAD_SHIFT_VALUE + i
    return input_ids


class _MmEmbedContract:
    """Shared cases; subclasses provide _make_model."""

    def _make_model(self, vocab_size: int, hidden_size: int):
        raise NotImplementedError

    def test_mm_sentinel_ids_do_not_cause_oob(self):
        """input_ids containing MM_PAD_SHIFT_VALUE+hash must not reach embed_tokens."""
        num_tokens = 10
        input_ids = _sentinel_input(num_tokens, mm_start=3, mm_end=7)
        mm_embeds = torch.randn(num_tokens, HIDDEN_SIZE)
        fb = _make_forward_batch(
            input_ids,
            mm_input_embeds=mm_embeds.clone(),
            extend_seq_lens=torch.tensor([num_tokens]),
            extend_start_loc=torch.tensor([0]),
        )

        model = self._make_model(VOCAB_SIZE, HIDDEN_SIZE)
        calls = _run_forward_capturing_embed_calls(model, input_ids, fb)

        for call in calls:
            call_ids = call[0][0]
            max_id = call_ids.max().item()
            self.assertLess(
                max_id,
                VOCAB_SIZE,
                f"embed_tokens was called with id {max_id} >= vocab_size "
                f"{VOCAB_SIZE}. MM sentinel values (MM_PAD_SHIFT_VALUE+hash) "
                f"must not reach embed_tokens.",
            )

    def test_no_mm_falls_back_to_embed_tokens(self):
        """Without mm_input_embeds, embed_tokens is called normally."""
        num_tokens = 5
        input_ids = torch.arange(num_tokens, dtype=torch.long)
        fb = _make_forward_batch(
            input_ids,
            mm_input_embeds=None,
            extend_seq_lens=torch.tensor([num_tokens]),
            extend_start_loc=torch.tensor([0]),
            has_mm=False,
        )

        model = self._make_model(VOCAB_SIZE, HIDDEN_SIZE)
        calls = _run_forward_capturing_embed_calls(model, input_ids, fb)

        self.assertTrue(len(calls) > 0, "embed_tokens should be called")
        full_ids_call = calls[0][0][0]
        self.assertEqual(full_ids_call.numel(), num_tokens)


class TestMistralEagleMmEmbed(_MmEmbedContract, CustomTestCase):
    def _make_model(self, vocab_size: int, hidden_size: int):
        return _make_mistral_eagle_model(vocab_size, hidden_size)


class TestMistralLarge3EagleMmEmbed(_MmEmbedContract, CustomTestCase):
    def _make_model(self, vocab_size: int, hidden_size: int):
        return _make_mistral_large_3_eagle_model(vocab_size, hidden_size)


if __name__ == "__main__":
    unittest.main()
