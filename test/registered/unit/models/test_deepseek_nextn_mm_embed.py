"""Regression test for sgl-project/sglang#37548.

DeepseekModelNextN.forward must use forward_batch.mm_input_embeds for multimodal
positions (where input_ids hold MM_PAD_SHIFT_VALUE+hash sentinels far above
vocab_size) instead of calling embed_tokens on those sentinel values, which
causes a CUDA index-out-of-bounds gather.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

VOCAB_SIZE = 154880
HIDDEN_SIZE = 64  # tiny for CPU test


def _make_forward_batch(
    input_ids: torch.Tensor,
    mm_input_embeds: torch.Tensor = None,
    extend_seq_lens: torch.Tensor = None,
    extend_start_loc: torch.Tensor = None,
    has_mm: bool = True,
):
    """Build a minimal mock ForwardBatch for DeepseekModelNextN.forward."""
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


def _make_model_nextn(vocab_size: int, hidden_size: int):
    """Build a mock DeepseekModelNextN with a real embed_tokens layer."""
    from sglang.srt.models.deepseek_nextn import DeepseekModelNextN

    model = DeepseekModelNextN.__new__(DeepseekModelNextN)
    torch.nn.Module.__init__(model)
    # Minimal attributes needed by forward
    model.vocab_size = vocab_size
    model.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
    model.enorm = torch.nn.RMSNorm(hidden_size)
    model.hnorm = torch.nn.RMSNorm(hidden_size)
    model.eh_proj = torch.nn.Linear(2 * hidden_size, hidden_size, bias=False)
    model.rot_weight = None
    model.alt_stream = None
    model.quant_config = None
    model.cp_rank = None
    model.cp_size = None
    model.dsa_enable_prefill_cp = False
    model.mla_enable_prefill_cp = False
    model.mtp_block = MagicMock(side_effect=lambda **kw: (kw["hidden_states"], None))
    return model


class TestDeepseekNextNMmEmbed(CustomTestCase):
    """DeepseekModelNextN must not call embed_tokens on MM sentinel token ids."""

    def test_mm_sentinel_ids_do_not_cause_oob(self):
        """input_ids containing MM_PAD_SHIFT_VALUE+hash must not reach embed_tokens."""
        num_tokens = 10
        mm_start, mm_end = 3, 7  # MM sentinel positions

        input_ids = torch.arange(num_tokens, dtype=torch.long)
        # Insert MM sentinel values
        for i in range(mm_start, mm_end):
            input_ids[i] = MM_PAD_SHIFT_VALUE + i

        # Build mm_input_embeds matching the target-produced embeddings
        mm_embeds = torch.randn(num_tokens, HIDDEN_SIZE)
        extend_seq_lens = torch.tensor([num_tokens])
        extend_start_loc = torch.tensor([0])

        fb = _make_forward_batch(
            input_ids,
            mm_input_embeds=mm_embeds.clone(),
            extend_seq_lens=extend_seq_lens,
            extend_start_loc=extend_start_loc,
        )

        model = _make_model_nextn(VOCAB_SIZE, HIDDEN_SIZE)

        # Use MagicMock to track embed_tokens calls
        mock_embed = MagicMock(side_effect=model.embed_tokens)
        object.__setattr__(model, "embed_tokens", mock_embed)

        with (
            patch(
                "sglang.srt.models.deepseek_nextn.is_cp_v2_active", return_value=False
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.mla_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.fused_eh_norm",
                side_effect=lambda h, p, ew, hw, eps: torch.cat(
                    [model.enorm(h), model.hnorm(p)], dim=-1
                ),
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.get_global_expert_distribution_recorder"
            ),
            patch("sglang.srt.models.deepseek_nextn.is_cuda", False),
            patch("sglang.srt.models.deepseek_nextn.is_npu", False),
            patch("sglang.srt.models.deepseek_nextn.envs") as mock_envs,
            patch("sglang.srt.models.deepseek_nextn.get_model") as mock_get_model,
            patch("sglang.srt.models.deepseek_nextn.get_parallel") as mock_get_parallel,
            patch("sglang.srt.models.deepseek_nextn.get_spec") as mock_get_spec,
        ):
            mock_envs.SGLANG_NPU_USE_MULTI_STREAM.get.return_value = False
            mock_get_model.return_value.quantization = None

            positions = torch.arange(num_tokens, dtype=torch.long)
            try:
                model.forward(input_ids, positions, fb)
            except Exception:
                pass  # We only care about embed_tokens call args

        # embed_tokens should only be called for last_indices (the appended
        # next-token), not with the full input_ids containing MM sentinels.
        for call in mock_embed.call_args_list:
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

        model = _make_model_nextn(VOCAB_SIZE, HIDDEN_SIZE)
        mock_embed = MagicMock(side_effect=model.embed_tokens)
        object.__setattr__(model, "embed_tokens", mock_embed)
        embed_calls = mock_embed.call_args_list

        with (
            patch(
                "sglang.srt.models.deepseek_nextn.is_cp_v2_active", return_value=False
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.mla_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.fused_eh_norm",
                side_effect=lambda h, p, ew, hw, eps: torch.cat(
                    [model.enorm(h), model.hnorm(p)], dim=-1
                ),
            ),
            patch(
                "sglang.srt.models.deepseek_nextn.get_global_expert_distribution_recorder"
            ),
            patch("sglang.srt.models.deepseek_nextn.is_cuda", False),
            patch("sglang.srt.models.deepseek_nextn.is_npu", False),
            patch("sglang.srt.models.deepseek_nextn.envs") as mock_envs,
            patch("sglang.srt.models.deepseek_nextn.get_model") as mock_get_model,
            patch("sglang.srt.models.deepseek_nextn.get_parallel") as mock_get_parallel,
            patch("sglang.srt.models.deepseek_nextn.get_spec") as mock_get_spec,
        ):
            mock_envs.SGLANG_NPU_USE_MULTI_STREAM.get.return_value = False
            mock_get_model.return_value.quantization = None

            positions = torch.arange(num_tokens, dtype=torch.long)
            try:
                model.forward(input_ids, positions, fb)
            except Exception:
                pass

        # embed_tokens should be called with the full input_ids
        self.assertTrue(mock_embed.call_count > 0, "embed_tokens should be called")
        full_ids_call = mock_embed.call_args_list[0][0][0]
        self.assertEqual(full_ids_call.numel(), num_tokens)


if __name__ == "__main__":
    unittest.main()
