"""Aux hidden states captured on earlier pipeline stages reach the last stage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.model_executor.model_runner_components.attention_backend_setup import (
    configure_aux_hidden_state_capture,
)
from sglang.srt.models import llama, qwen3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_HIDDEN = 4
_NUM_LAYERS = 8
_TARGETS = [
    (llama, llama.LlamaModel, llama.LlamaForCausalLM),
    (qwen3, qwen3.Qwen3Model, qwen3.Qwen3ForCausalLM),
]


class _Layer:
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
    final_norm=lambda hidden_states, forward_batch, norm, **kwargs: hidden_states,
)


def _stage(*, start, end, layers_to_capture, is_first_rank, is_last_rank):
    return SimpleNamespace(
        pp_group=SimpleNamespace(
            is_first_rank=is_first_rank, is_last_rank=is_last_rank
        ),
        embed_tokens=lambda input_ids: input_ids[:, None].float().repeat(1, _HIDDEN),
        layers={i: _Layer(i) for i in range(start, end)},
        start_layer=start,
        end_layer=end,
        layers_to_capture=layers_to_capture,
        norm=None,
    )


def _run_pipeline(target, *, boundaries, layers_to_capture):
    """Returns the last stage's packed aux and (received width, sized width) per stage."""
    module, model_cls, causal_lm_cls = target
    out = None
    widths = []
    num_stages = len(boundaries) - 1
    with patch.object(module, "residual_batch", _RESIDUAL_BATCH):
        for rank in range(num_stages):
            model = _stage(
                start=boundaries[rank],
                end=boundaries[rank + 1],
                layers_to_capture=layers_to_capture,
                is_first_rank=rank == 0,
                is_last_rank=rank == num_stages - 1,
            )
            if rank > 0:
                received = out.tensors.get("aux_hidden_states")
                sized = causal_lm_cls.get_pp_proxy_aux_hidden_size(
                    SimpleNamespace(
                        config=SimpleNamespace(hidden_size=_HIDDEN), model=model
                    )
                )
                widths.append((0 if received is None else received.shape[-1], sized))
            out = model_cls.forward(
                model,
                torch.arange(1, 4),
                positions=None,
                forward_batch=None,
                pp_proxy_tensors=out,
            )
    _, aux_hidden_states = out
    return pack_aux_hidden_states(aux_hidden_states), widths


class TestPipelineAuxHiddenRelay(unittest.TestCase):
    def test_multi_stage_aux_matches_single_stage(self):
        # Stage [3, 5) owns no capture layer; layer 5 is a stage's first layer.
        layers_to_capture = [1, 2, 5, 7]
        for target in _TARGETS:
            with self.subTest(model=target[1].__name__):
                expected, _ = _run_pipeline(
                    target,
                    boundaries=[0, _NUM_LAYERS],
                    layers_to_capture=layers_to_capture,
                )
                actual, widths = _run_pipeline(
                    target,
                    boundaries=[0, 3, 5, _NUM_LAYERS],
                    layers_to_capture=layers_to_capture,
                )

                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for received, sized in widths:
                    self.assertEqual(received, sized)

    def test_target_without_relay_is_rejected_under_pp(self):
        target = SimpleNamespace(set_eagle3_layers_to_capture=lambda layer_ids: None)
        with self.assertRaisesRegex(ValueError, "does not relay aux hidden states"):
            configure_aux_hidden_state_capture(
                model=target,
                eagle_use_aux_hidden_state=True,
                eagle_aux_hidden_state_layer_ids=None,
                dflash_use_aux_hidden_state=False,
                dflash_target_layer_ids=None,
                is_dspark=False,
                pp_size=2,
            )


if __name__ == "__main__":
    unittest.main()
