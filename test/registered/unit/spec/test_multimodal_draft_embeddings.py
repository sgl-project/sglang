import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.eagle_worker_v2 import (
    _shift_mm_input_embeds_for_draft_prefill,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _MutatingLanguageModel:
    def get_input_embeddings(self):
        return lambda input_ids: torch.zeros((input_ids.shape[0], 2))

    def __call__(self, *, input_ids, forward_batch, input_embeds):
        input_embeds.add_(100)
        return input_embeds


def test_eagle_mm_embeddings_survive_in_place_target_updates():
    source_embeds = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    expected_embeds = source_embeds.clone()
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        contains_mm_inputs=lambda: True,
        mm_inputs=[SimpleNamespace()],
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[4],
        input_embeds=None,
        spec_algorithm=SpeculativeAlgorithm.EAGLE3,
    )

    with (
        get_context().override_server_args(),
        patch(
            "sglang.srt.managers.mm_utils.embed_mm_inputs",
            return_value=(source_embeds, {}),
        ),
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=torch.arange(4),
            forward_batch=forward_batch,
            language_model=_MutatingLanguageModel(),
        )

    torch.testing.assert_close(hidden_states, expected_embeds + 100)
    torch.testing.assert_close(forward_batch.mm_input_embeds, expected_embeds)
    assert forward_batch.mm_input_embeds.data_ptr() != hidden_states.data_ptr()


def test_shift_mm_embeddings_stays_within_request_boundaries():
    mm_input_embeds = torch.tensor([[0], [1], [2], [10], [11], [20]])

    shifted_embeds = _shift_mm_input_embeds_for_draft_prefill(
        mm_input_embeds, extend_lens=[3, 2, 1]
    )

    torch.testing.assert_close(
        shifted_embeds, torch.tensor([[1], [2], [2], [11], [11], [20]])
    )
    assert shifted_embeds.data_ptr() != mm_input_embeds.data_ptr()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
