"""Deepstack visual embeddings are added to a decoder layer's output. When the
layer left its FFN all-reduce to the next layer, that output is one rank's
partial sum, and an embedding added to it is counted once per rank."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.layers.communicator import LayerCommunicator, UnreducedOutput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TP_SIZE = 2
HIDDEN = 4
TOKENS = 3
NUM_LAYERS = 4


def all_reduce(hidden_states):
    # Every rank holds the same partial sum in this single-process stand-in.
    return hidden_states * TP_SIZE


# The group a deferred FFN output owes its sum over.
GROUP = SimpleNamespace(all_reduce=all_reduce)


class DeferringLayer(nn.Module):
    """One TP rank's view of a decoder layer whose FFN output sums to one. It
    completes a reduction left by the previous layer, and leaves its own to the
    next layer unless it is the last."""

    def __init__(self, is_last_layer):
        super().__init__()
        self.is_last_layer = is_last_layer
        # The model ends its layers at this communicator's finish_layer_stack.
        self.layer_communicator = LayerCommunicator.__new__(LayerCommunicator)

    def forward(
        self, positions=None, hidden_states=None, forward_batch=None, residual=None, **_
    ):
        if isinstance(hidden_states, UnreducedOutput):
            hidden_states = all_reduce(hidden_states.partial)
        residual = hidden_states if residual is None else hidden_states + residual
        if self.is_last_layer:
            return torch.ones_like(residual), residual
        return (
            UnreducedOutput(torch.full_like(residual, 1 / TP_SIZE), group=GROUP),
            residual,
        )


class SumNorm(nn.Module):
    def forward(self, hidden_states, residual=None, post_residual_addition=None):
        if residual is None:
            return hidden_states
        return hidden_states + residual, residual


def stub_model(cls, **attrs):
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    layers = [DeferringLayer(i == NUM_LAYERS - 1) for i in range(NUM_LAYERS)]
    common = dict(
        pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
        layers=layers,
        layers_to_capture=[],
        hidden_size=HIDDEN,
        norm=SumNorm(),
    )
    for name, value in {**common, **attrs}.items():
        object.__setattr__(model, name, value)
    return model


def qwen3_vl_moe():
    from sglang.srt.models.qwen3_vl_moe import Qwen3MoeLLMModel

    return stub_model(
        Qwen3MoeLLMModel,
        start_layer=0,
        end_layer=NUM_LAYERS,
        use_hf_deepstack_order=False,
        deepstack_embed_to_decoder_layer=range(3),
    )


def qwen3_5():
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    return stub_model(
        Qwen3_5ForCausalLM,
        _start_layer=0,
        _end_layer=NUM_LAYERS,
        flashinfer_mnnvl_cutedsl_fusion=None,
    )


MODELS = (qwen3_vl_moe, qwen3_5)


class TestDeepstackOnDeferredReduction(CustomTestCase):
    def setUp(self):
        self.embeds = torch.zeros(TOKENS, HIDDEN)

    def run_model(self, build, deepstack):
        model = build()
        return model.forward(
            input_ids=None,
            positions=None,
            forward_batch=None,
            input_embeds=self.embeds.clone(),
            input_deepstack_embeds=deepstack,
        )

    def test_each_deepstack_embedding_is_added_once(self):
        values = (0.125, 0.25, 0.5)
        deepstack = torch.cat([torch.full((TOKENS, HIDDEN), v) for v in values], 1)
        for build in MODELS:
            with self.subTest(model=build.__name__):
                hidden_states = self.run_model(build, deepstack)
                expected = torch.full((TOKENS, HIDDEN), NUM_LAYERS + sum(values))
                torch.testing.assert_close(hidden_states, expected)

    def test_without_deepstack(self):
        for build in MODELS:
            with self.subTest(model=build.__name__):
                hidden_states = self.run_model(build, None)
                torch.testing.assert_close(
                    hidden_states, torch.full((TOKENS, HIDDEN), float(NUM_LAYERS))
                )


if __name__ == "__main__":
    unittest.main()
