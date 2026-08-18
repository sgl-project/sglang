import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.models.deepseek_v4 import (
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.calls = []
        self.hc_post_calls = 0

    def __call__(
        self,
        *,
        hidden_states,
        prev_residual,
        prev_post,
        prev_comb,
        **kwargs,
    ):
        self.calls.append((prev_residual, prev_post, prev_comb))
        value = self.layer_id + 1
        return (
            hidden_states + value,
            torch.tensor(value),
            torch.tensor(value + 10),
            torch.tensor(value + 20),
        )

    def hc_post(self, hidden_states, residual, post, comb):
        self.hc_post_calls += 1
        return hidden_states + residual + post + comb


class TestDeepseekV4SplitPrefill(unittest.TestCase):
    def _make_model(self):
        layers = [_FakeLayer(0), _FakeLayer(1)]
        model = SimpleNamespace(
            embed_tokens=lambda input_ids: input_ids.float().unsqueeze(-1),
            hc_mult=2,
            layers=layers,
            end_layer=len(layers),
            use_fused_mhc_post_pre=True,
            hc_head=lambda hidden, *args: hidden.sum(dim=1),
            hc_head_fn=None,
            hc_head_scale=None,
            hc_head_base=None,
            norm=lambda hidden: hidden,
        )
        return model, layers

    def _run_split(self, model, forward_batch, split_interval):
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.dsa_use_prefill_cp",
                return_value=False,
            ),
            patch(
                "sglang.srt.models.deepseek_v4.check_cuda_graph_backend",
                return_value=True,
            ),
        ):
            return DeepseekV4Model.forward_split_prefill(
                model,
                torch.tensor([1, 2]),
                torch.tensor([0, 1]),
                forward_batch,
                split_interval,
            )

    def test_split_execution_preserves_cross_layer_mhc_state(self):
        model, layers = self._make_model()
        forward_batch = SimpleNamespace(
            hidden_states=None,
            model_specific_states=None,
            freqs_cis_c4=object(),
            freqs_cis_c128=object(),
        )

        self.assertIsNone(self._run_split(model, forward_batch, (0, 1)))
        self.assertFalse(hasattr(forward_batch, "freqs_cis_c4"))
        self.assertFalse(hasattr(forward_batch, "freqs_cis_c128"))
        self.assertEqual(layers[0].hc_post_calls, 0)

        result = self._run_split(model, forward_batch, (1, 2))

        self.assertIsNotNone(result)
        for actual, expected in zip(layers[1].calls[0], (1, 11, 21)):
            self.assertEqual(actual.item(), expected)
        self.assertEqual(layers[0].hc_post_calls, 0)
        self.assertEqual(layers[1].hc_post_calls, 1)

    def test_split_execution_matches_one_shot_execution(self):
        split_model, _ = self._make_model()
        split_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)
        self._run_split(split_model, split_batch, (0, 1))
        split_result = self._run_split(split_model, split_batch, (1, 2))

        one_shot_model, _ = self._make_model()
        one_shot_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)
        one_shot_result = self._run_split(one_shot_model, one_shot_batch, (0, 2))

        torch.testing.assert_close(split_result[0], one_shot_result[0])
        torch.testing.assert_close(split_result[1], one_shot_result[1])

    def test_causal_lm_initializes_cp_once_and_processes_final_logits(self):
        prepare_cp = Mock()
        model_forward = Mock(
            side_effect=[None, (torch.tensor([1.0]), torch.tensor([2.0]))]
        )
        logits_processor = Mock(return_value="logits")
        model = SimpleNamespace(
            _prepare_dsa_prefill_cp=prepare_cp,
            model=SimpleNamespace(forward_split_prefill=model_forward),
            logits_processor=logits_processor,
            lm_head=object(),
        )
        attn_context = SimpleNamespace(
            maybe_input_scattered=lambda forward_batch: nullcontext()
        )
        args = (
            torch.tensor([1]),
            torch.tensor([0]),
            SimpleNamespace(),
        )

        with patch(
            "sglang.srt.models.deepseek_v4.get_attn_tp_context",
            return_value=attn_context,
        ):
            self.assertIsNone(
                DeepseekV4ForCausalLM.forward_split_prefill(
                    model, *args, split_interval=(0, 1)
                )
            )
            result = DeepseekV4ForCausalLM.forward_split_prefill(
                model, *args, split_interval=(1, 2)
            )

        self.assertEqual(result, "logits")
        prepare_cp.assert_called_once_with(args[0], args[2])
        logits_processor.assert_called_once()


if __name__ == "__main__":
    unittest.main()
