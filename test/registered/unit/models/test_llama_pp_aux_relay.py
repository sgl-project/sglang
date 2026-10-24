"""EAGLE3 aux hidden states captured on earlier pipeline stages reach the last stage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models import llama
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HIDDEN = 4
_NUM_TOKENS = 3
_NUM_LAYERS = 8


class _Layer:
    """Captures its input stream, like a decoder layer reading the residual."""

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.attn_boundary = SimpleNamespace(
            from_pp=lambda proxy, forward_batch: proxy["hidden_states"]
        )

    def __call__(self, positions, hidden_states, forward_batch, capture_output=None):
        if capture_output is not None:
            capture_output(hidden_states)
        return hidden_states * 2 + self.layer_id


_RESIDUAL_BATCH = SimpleNamespace(
    start=lambda forward_batch: None,
    to_pp=lambda hidden_states, forward_batch: PPProxyTensors(
        {"hidden_states": hidden_states}
    ),
    final_norm=lambda hidden_states, forward_batch, norm: hidden_states,
)


def _stage(*, start, end, layers_to_capture, pp_size, pp_rank):
    return SimpleNamespace(
        pp_group=SimpleNamespace(
            is_first_rank=pp_rank == 0, is_last_rank=pp_rank == pp_size - 1
        ),
        embed_tokens=lambda input_ids: (
            input_ids[:, None].float().expand(-1, _HIDDEN).contiguous()
        ),
        layers={i: _Layer(i) for i in range(start, end)},
        start_layer=start,
        end_layer=end,
        layers_to_capture=layers_to_capture,
        norm=None,
    )


def _run_pipeline(*, boundaries, layers_to_capture):
    """Run LlamaModel.forward stage by stage; return the last stage's packed aux
    and, per receiving stage, (received width, get_pp_proxy_aux_hidden_size)."""
    input_ids = torch.arange(1, _NUM_TOKENS + 1)
    proxy = None
    widths = []
    pp_size = len(boundaries) - 1
    with patch.object(llama, "residual_batch", _RESIDUAL_BATCH):
        for rank in range(pp_size):
            model = _stage(
                start=boundaries[rank],
                end=boundaries[rank + 1],
                layers_to_capture=layers_to_capture,
                pp_size=pp_size,
                pp_rank=rank,
            )
            if rank > 0:
                causal_lm = SimpleNamespace(
                    config=SimpleNamespace(hidden_size=_HIDDEN), model=model
                )
                received = proxy.tensors.get("aux_hidden_states")
                widths.append(
                    (
                        0 if received is None else received.shape[-1],
                        llama.LlamaForCausalLM.get_pp_proxy_aux_hidden_size(causal_lm),
                    )
                )
            out = llama.LlamaModel.forward(
                model,
                input_ids,
                positions=None,
                forward_batch=None,
                pp_proxy_tensors=proxy,
            )
            proxy = out
    _, aux_hidden_states = out
    return pack_aux_hidden_states(aux_hidden_states), widths


class TestLlamaPipelineAuxRelay(unittest.TestCase):
    def test_multi_stage_aux_matches_single_stage(self):
        # Stage [3, 5) owns no capture layer but must still forward the
        # earlier captures; layer 5 is captured at the input of a stage.
        layers_to_capture = [1, 2, 5, 7]
        expected, _ = _run_pipeline(
            boundaries=[0, _NUM_LAYERS], layers_to_capture=layers_to_capture
        )
        self.assertEqual(
            expected.shape, (_NUM_TOKENS, _HIDDEN * len(layers_to_capture))
        )

        actual, widths = _run_pipeline(
            boundaries=[0, 3, 5, _NUM_LAYERS], layers_to_capture=layers_to_capture
        )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The CUDA graph input buffer of each stage is sized by the hook, so it
        # must equal the width the previous stage sends.
        self.assertEqual(
            widths, [(2 * _HIDDEN, 2 * _HIDDEN), (2 * _HIDDEN, 2 * _HIDDEN)]
        )


class TestAuxCaptureRequiresPipelineRelay(unittest.TestCase):
    """A target that only captures on the last stage gives the EAGLE3 draft a
    narrower feature; at width == hidden_size the draft skips its fc silently."""

    class _LastStageOnlyTarget:
        def set_eagle3_layers_to_capture(self, layer_ids=None):
            self.layer_ids = layer_ids

    def _configure(self, model, *, pp_size):
        from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
            configure_aux_hidden_state_capture,
        )

        configure_aux_hidden_state_capture(
            model=model,
            eagle_use_aux_hidden_state=True,
            eagle_aux_hidden_state_layer_ids=[1, 4, 6],
            dflash_use_aux_hidden_state=False,
            dflash_target_layer_ids=None,
            is_dspark=False,
            pp_size=pp_size,
        )

    def test_rejects_target_without_relay_under_pp(self):
        with self.assertRaisesRegex(ValueError, "does not relay aux hidden states"):
            self._configure(self._LastStageOnlyTarget(), pp_size=2)

    def test_allows_target_without_relay_without_pp(self):
        model = self._LastStageOnlyTarget()
        self._configure(model, pp_size=1)
        self.assertEqual(model.layer_ids, [1, 4, 6])


if __name__ == "__main__":
    unittest.main()
