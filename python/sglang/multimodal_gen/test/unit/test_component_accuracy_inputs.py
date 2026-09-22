from types import SimpleNamespace

import torch.nn as nn

from sglang.multimodal_gen.test.single_test_file.component_accuracy.hooks import (
    _build_transformer_hook_inputs,
)


class _TransformerWithOptionalMask(nn.Module):
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_hidden_states_mask=None,
    ):
        raise NotImplementedError


def _case(*, ring_degree: int) -> SimpleNamespace:
    return SimpleNamespace(
        server_args=SimpleNamespace(
            model_path="test/model",
            ring_degree=ring_degree,
        )
    )


def test_omits_noop_attention_mask_for_ring_parallel_case():
    inputs = _build_transformer_hook_inputs(
        _case(ring_degree=2), _TransformerWithOptionalMask(), "cpu"
    )

    assert "encoder_hidden_states_mask" not in inputs
    assert "encoder_attention_mask" not in inputs


def test_keeps_attention_mask_for_non_ring_case():
    inputs = _build_transformer_hook_inputs(
        _case(ring_degree=1), _TransformerWithOptionalMask(), "cpu"
    )

    assert inputs["encoder_hidden_states_mask"].all()
    assert inputs["encoder_attention_mask"].all()
