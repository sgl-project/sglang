import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

from sglang.srt.layers.aux_hidden_states import pack_aux_hidden_states
from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models import deepseek_v4
from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _Layer(nn.Module):
    """Defer a stream-dependent residual update in the fused mHC path."""

    def __init__(self, layer_id, fused):
        super().__init__()
        self.layer_id = layer_id
        self.fused = fused

    def forward(self, *, hidden_states, prev_residual, prev_post, prev_comb, **kwargs):
        if prev_residual is not None:
            hidden_states = self.hc_post(
                hidden_states, prev_residual, prev_post, prev_comb
            )
        residual = hidden_states
        delta = torch.full_like(hidden_states[:, 0], self.layer_id + 1)
        post = torch.arange(1, hidden_states.shape[1] + 1).view(1, -1, 1)
        if self.fused:
            return delta, residual, post, None
        return self.hc_post(delta, residual, post, None), None, None, None

    @staticmethod
    def hc_post(hidden_states, residual, post, comb):
        return residual + hidden_states.unsqueeze(1) * post


class TestDeepseekV4PipelineAuxCapture(CustomTestCase):
    def setUp(self):
        super().setUp()
        for name, kwargs in (
            ("get_parallel", {"return_value": SimpleNamespace(attn_dp_size=1)}),
            ("check_cuda_graph_backend", {"return_value": True}),
            ("_is_npu", {"new": False}),
            (
                "get_attn_tp_context",
                {
                    "return_value": SimpleNamespace(
                        maybe_input_scattered=lambda _: nullcontext()
                    )
                },
            ),
        ):
            patcher = patch.object(deepseek_v4, name, **kwargs)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _make_stage(start, end, *, fused=False):
        model = DeepseekV4Model.__new__(DeepseekV4Model)
        nn.Module.__init__(model)
        model.start_layer, model.end_layer = start, end
        model.hidden_size, model.hc_mult = 3, 4
        model.pp_group = SimpleNamespace(
            is_first_rank=start == 0, is_last_rank=end == 6
        )
        model.layers = nn.ModuleList(
            _Layer(i, fused) if start <= i < end else nn.Identity() for i in range(6)
        )
        model.use_fused_mhc_post_pre = fused
        model._can_run_tbo = lambda _: False
        model.dspark_layers_to_capture = None
        model.hc_head = lambda hidden, *_: hidden.mean(dim=1)
        model.hc_head_fn = model.hc_head_scale = model.hc_head_base = None
        model.norm = nn.Identity()

        wrapper = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
        nn.Module.__init__(wrapper)
        wrapper.model = model
        wrapper.pp_group = model.pp_group
        wrapper.config = SimpleNamespace(num_hidden_layers=6)
        wrapper.capture_aux_hidden_states = False
        wrapper.lm_head = nn.Identity()
        wrapper.logits_processor = Mock(side_effect=lambda *args, **kwargs: args[4])
        return wrapper

    @staticmethod
    def _forward(stage, embeds, proxy=None):
        num_tokens = embeds.shape[0]
        return stage.model.forward(
            input_ids=torch.zeros(num_tokens, dtype=torch.long),
            positions=torch.arange(num_tokens),
            forward_batch=SimpleNamespace(),
            input_embeds=embeds,
            pp_proxy_tensors=proxy,
        )

    def test_cross_stage_capture_matches_single_stage_and_requested_order(self):
        for fused in (False, True):
            for num_tokens in (0, 1, 5):
                # Include boundaries, non-monotonic order, and stages that
                # forward captures but produce none themselves.
                for layer_ids in ([5, 0, 2, 1, 4, 3], [0], [2], [5], [4, 0]):
                    with self.subTest(fused=fused, tokens=num_tokens, layers=layer_ids):
                        embeds = torch.arange(num_tokens * 3, dtype=torch.float32).view(
                            num_tokens, 3
                        )
                        single = self._make_stage(0, 6, fused=fused)
                        single.set_dspark_layers_to_capture(layer_ids)
                        expected_output, expected_aux = self._forward(single, embeds)

                        proxy = None
                        for start, end in ((0, 2), (2, 4), (4, 6)):
                            stage = self._make_stage(start, end, fused=fused)
                            stage.set_dspark_layers_to_capture(layer_ids)
                            output = self._forward(stage, embeds, proxy)
                            if end == 6:
                                actual_output, actual_aux = output
                                break
                            expected_keys = {"hidden_states"} | {
                                f"dspark_aux_hidden_states_{i}"
                                for i in layer_ids
                                if i < end
                            }
                            self.assertEqual(set(output.tensors), expected_keys)
                            self.assertEqual(
                                output["hidden_states"].shape, (num_tokens, 12)
                            )
                            if proxy is not None:
                                for key in proxy.tensors:
                                    if key != "hidden_states":
                                        self.assertIs(output[key], proxy[key])
                            # Receive into independent storage, as the PP transport does.
                            proxy = PPProxyTensors(
                                {
                                    key: value.clone()
                                    for key, value in output.tensors.items()
                                }
                            )

                        torch.testing.assert_close(actual_output, expected_output)
                        torch.testing.assert_close(actual_aux, expected_aux)
                        torch.testing.assert_close(
                            pack_aux_hidden_states(actual_aux),
                            torch.cat(
                                [
                                    embeds + 2.5 * (i + 1) * (i + 2) / 2
                                    for i in layer_ids
                                ],
                                dim=-1,
                            ),
                        )

    def test_configuration_enables_all_stages_and_sizes_only_incoming_captures(self):
        layer_ids = [4, 0, 3]
        for start, end in ((0, 2), (2, 4), (4, 6)):
            stage = self._make_stage(start, end)
            stage.set_dspark_layers_to_capture(layer_ids)
            self.assertTrue(stage.capture_aux_hidden_states)
            self.assertEqual(stage.model.dspark_layers_to_capture, layer_ids)
            self.assertEqual(
                stage.pp_proxy_aux_hidden_state_keys,
                tuple(f"dspark_aux_hidden_states_{i}" for i in layer_ids if i < start),
            )

    def test_invalid_layers_fail_on_every_stage(self):
        for start, end in ((0, 2), (2, 4), (4, 6)):
            for layer_ids in (None, [], [0, 0], [-1], [6]):
                with self.subTest(start=start, layers=layer_ids):
                    stage = self._make_stage(start, end)
                    with self.assertRaises(ValueError):
                        stage.set_dspark_layers_to_capture(layer_ids)
                    self.assertFalse(stage.capture_aux_hidden_states)

    def test_missing_upstream_capture_fails_instead_of_returning_partial_features(self):
        stage = self._make_stage(4, 6)
        stage.set_dspark_layers_to_capture([0, 5])
        with self.assertRaisesRegex(KeyError, "dspark_aux_hidden_states_0"):
            self._forward(
                stage,
                torch.zeros(2, 3),
                PPProxyTensors({"hidden_states": torch.zeros(2, 12)}),
            )

    def test_capture_disabled_preserves_pipeline_outputs(self):
        embeds = torch.arange(6, dtype=torch.float32).view(2, 3)
        expected = self._forward(self._make_stage(0, 6), embeds)
        proxy = None
        for start, end in ((0, 2), (2, 4), (4, 6)):
            output = self._forward(self._make_stage(start, end), embeds, proxy)
            if end < 6:
                self.assertEqual(set(output.tensors), {"hidden_states"})
                proxy = output
        torch.testing.assert_close(output, expected)

    def test_wrapper_only_passes_assembled_captures_to_last_stage_logits(self):
        embeds = torch.zeros(2, 3)
        proxy = None
        for start, end in ((0, 2), (2, 4), (4, 6)):
            stage = self._make_stage(start, end)
            stage.set_dspark_layers_to_capture([2, 0])
            output = stage.forward(
                torch.zeros(2, dtype=torch.long),
                torch.arange(2),
                SimpleNamespace(),
                input_embeds=embeds,
                pp_proxy_tensors=proxy,
            )
            if end < 6:
                self.assertIsInstance(output, PPProxyTensors)
                stage.logits_processor.assert_not_called()
                proxy = output
            else:
                stage.logits_processor.assert_called_once()
                torch.testing.assert_close(output, [embeds + 15, embeds + 2.5])


if __name__ == "__main__":
    unittest.main()
