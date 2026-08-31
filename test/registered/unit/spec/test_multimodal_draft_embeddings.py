"""Regression tests for multimodal EAGLE draft-prefill embeddings."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.speculative.eagle_worker_v2 import (
    _shift_mm_input_embeds_for_draft_prefill,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _MutatingLanguageModel:
    pp_group = SimpleNamespace(is_first_rank=True)

    def get_input_embeddings(self):
        return lambda input_ids: input_ids

    def __call__(self, *, input_ids, forward_batch, input_embeds, **kwargs):
        del input_ids, forward_batch, kwargs
        input_embeds.add_(100)
        return input_embeds


class TestMultimodalDraftEmbeddings(CustomTestCase):
    def test_target_forward_cannot_mutate_draft_embedding_snapshot(self):
        input_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        expected = input_embeds.clone()
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: False,
                is_target_verify=lambda: False,
            ),
            contains_mm_inputs=lambda: True,
            mm_inputs=[SimpleNamespace(mm_items=[])],
            extend_prefix_lens_cpu=[0],
            extend_seq_lens_cpu=[3],
            input_embeds=None,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            mm_input_embeds=None,
        )

        with (
            patch(
                "sglang.srt.managers.mm_utils.embed_mm_inputs",
                return_value=(input_embeds, {}),
            ),
            patch("sglang.srt.managers.mm_utils.get_server_args", return_value=None),
        ):
            general_mm_embed_routine(
                torch.arange(3),
                forward_batch,
                _MutatingLanguageModel(),
            )

        self.assertTrue(torch.equal(forward_batch.mm_input_embeds, expected))
        self.assertFalse(
            forward_batch.mm_input_embeds.data_ptr() == input_embeds.data_ptr()
        )

    def test_shift_preserves_request_boundaries(self):
        embeds = torch.arange(24, dtype=torch.float32).reshape(6, 4)

        shifted = _shift_mm_input_embeds_for_draft_prefill(embeds, [2, 1, 3])

        expected = torch.stack(
            (embeds[1], embeds[1], embeds[2], embeds[4], embeds[5], embeds[5])
        )
        self.assertTrue(torch.equal(shifted, expected))


if __name__ == "__main__":
    unittest.main()
