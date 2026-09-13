import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _MutatingLanguageModel:
    def get_input_embeddings(self):
        return lambda input_ids: torch.zeros((input_ids.shape[0], 2))

    def __call__(self, *, input_ids, forward_batch, input_embeds):
        input_embeds.add_(100)
        return input_embeds


def _run_mutating_prefill(input_embeds):
    source_embeds = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    expected_embeds = source_embeds.clone()
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        contains_mm_inputs=lambda: True,
        mm_inputs=[SimpleNamespace()],
        extend_prefix_lens_cpu=[0],
        extend_seq_lens_cpu=[4],
        input_embeds=input_embeds,
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

    return source_embeds, expected_embeds, forward_batch, hidden_states


def test_eagle_mm_embeddings_survive_in_place_target_updates():
    source_embeds, expected_embeds, forward_batch, hidden_states = (
        _run_mutating_prefill(input_embeds=None)
    )

    torch.testing.assert_close(hidden_states, expected_embeds + 100)
    torch.testing.assert_close(forward_batch.mm_input_embeds, expected_embeds)
    assert forward_batch.mm_input_embeds.data_ptr() != hidden_states.data_ptr()


def test_eagle_mm_embeddings_reuse_source_when_static_buffer_exists():
    static_embeds = torch.zeros(4, 2)
    source_embeds, expected_embeds, forward_batch, hidden_states = (
        _run_mutating_prefill(input_embeds=static_embeds)
    )

    torch.testing.assert_close(hidden_states, expected_embeds + 100)
    assert forward_batch.mm_input_embeds is source_embeds
    torch.testing.assert_close(forward_batch.mm_input_embeds, expected_embeds)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
