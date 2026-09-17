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


class _FakePrevLayer:
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.engram = None
        self.calls = []

    def forward_hc_pre_from_prev(self, *, hidden_states, prev_pre, **kwargs):
        self.calls.append(prev_pre)
        value = self.layer_id + 1
        return hidden_states + value, torch.tensor(float(value))


class TestDeepseekV4SplitPrefill(unittest.TestCase):
    def _make_model(self):
        layers = [_FakeLayer(0), _FakeLayer(1)]
        model = SimpleNamespace(
            embed_tokens=lambda input_ids: input_ids.float().unsqueeze(-1),
            hc_mult=2,
            layers=layers,
            start_layer=0,
            end_layer=len(layers),
            hc_pre_from_prev_sublayer=False,
            use_fused_mhc_post_pre=True,
            hc_head=lambda hidden, *args: hidden.sum(dim=1),
            hc_head_fn=None,
            hc_head_scale=None,
            hc_head_base=None,
            norm=lambda hidden: hidden,
            dspark_layers_to_capture=None,
        )
        return model, layers

    def _run_split(self, model, forward_batch, split_interval):
        with (
            patch(
                "sglang.srt.models.deepseek_v4.get_parallel",
                return_value=SimpleNamespace(attn_dp_size=1),
            ),
            patch(
                "sglang.srt.models.deepseek_v4.is_cp_active",
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

    def test_split_execution_accumulates_dspark_captures(self):
        split_model, _ = self._make_model()
        split_model.dspark_layers_to_capture = [0, 1]
        split_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)

        self.assertIsNone(self._run_split(split_model, split_batch, (0, 1)))
        split_result, split_aux = self._run_split(
            split_model, split_batch, (1, 2)
        )

        one_shot_model, _ = self._make_model()
        one_shot_model.dspark_layers_to_capture = [0, 1]
        one_shot_batch = SimpleNamespace(hidden_states=None, model_specific_states=None)
        one_shot_result, one_shot_aux = self._run_split(
            one_shot_model, one_shot_batch, (0, 2)
        )

        torch.testing.assert_close(split_result[0], one_shot_result[0])
        self.assertEqual(len(split_aux), 2)
        for actual, expected in zip(split_aux, one_shot_aux):
            torch.testing.assert_close(actual, expected)

    def test_dsv41_split_preserves_predecessor_mhc_state(self):
        layers = [_FakePrevLayer(0), _FakePrevLayer(1)]
        model = SimpleNamespace(
            embed_tokens=lambda input_ids: input_ids.float().unsqueeze(-1),
            hc_mult=2,
            layers=layers,
            start_layer=0,
            end_layer=2,
            hc_pre_from_prev_sublayer=True,
            engram_hasher=None,
            late_layer_start=None,
            config=SimpleNamespace(model_type="deepseek_v41", vision_n_layers=0),
            norm=lambda hidden: hidden,
            dspark_layers_to_capture=None,
        )
        forward_mode = SimpleNamespace(is_extend=lambda: True)
        batch = SimpleNamespace(
            hidden_states=None,
            model_specific_states=None,
            forward_mode=forward_mode,
        )

        with patch(
            "sglang.kernels.ops.layernorm.mhc.hc_combine",
            side_effect=lambda x, pre, *_: x + pre,
        ):
            self.assertIsNone(self._run_split(model, batch, (0, 1)))
            result = self._run_split(model, batch, (1, 2))

        self.assertIsNone(layers[0].calls[0])
        self.assertEqual(layers[1].calls[0].item(), 1.0)
        self.assertIsNotNone(result)

    def test_causal_lm_processes_only_final_split_logits(self):
        model_forward = Mock(
            side_effect=[None, (torch.tensor([1.0]), torch.tensor([2.0]))]
        )
        logits_processor = Mock(return_value="logits")
        model = SimpleNamespace(
            vision=None,
            model=SimpleNamespace(start_layer=0, forward_split_prefill=model_forward),
            logits_processor=logits_processor,
            lm_head=object(),
            capture_aux_hidden_states=False,
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
        logits_processor.assert_called_once()

    def test_causal_lm_passes_split_dspark_captures_to_logits_processor(self):
        aux = [torch.tensor([[3.0]])]
        model = SimpleNamespace(
            vision=None,
            model=SimpleNamespace(
                start_layer=0,
                forward_split_prefill=Mock(
                    return_value=((torch.tensor([1.0]), torch.tensor([2.0])), aux)
                )
            ),
            logits_processor=Mock(return_value="logits"),
            lm_head=object(),
            capture_aux_hidden_states=True,
        )
        attn_context = SimpleNamespace(
            maybe_input_scattered=lambda forward_batch: nullcontext()
        )
        input_ids = torch.tensor([1])
        forward_batch = SimpleNamespace()

        with patch(
            "sglang.srt.models.deepseek_v4.get_attn_tp_context",
            return_value=attn_context,
        ):
            result = DeepseekV4ForCausalLM.forward_split_prefill(
                model,
                input_ids,
                torch.tensor([0]),
                forward_batch,
                split_interval=(0, 2),
            )

        self.assertEqual(result, "logits")
        call = model.logits_processor.call_args
        self.assertIs(call.args[4], aux)
        self.assertIsNone(call.kwargs["hidden_states_before_norm"])


if __name__ == "__main__":
    unittest.main()
