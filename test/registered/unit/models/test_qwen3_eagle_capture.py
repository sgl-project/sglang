"""Regression tests for Qwen3-VL and Qwen3.5 EAGLE3 layer capture."""

import unittest
from types import SimpleNamespace

from torch import nn

from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _mock_wrapper(wrapper_cls, backbone, *, num_hidden_layers=12):
    wrapper = wrapper_cls.__new__(wrapper_cls)
    nn.Module.__init__(wrapper)
    wrapper.pp_group = SimpleNamespace(is_last_rank=True)
    wrapper.config = SimpleNamespace(num_hidden_layers=num_hidden_layers)
    wrapper.model = backbone
    wrapper.capture_aux_hidden_states = False
    return wrapper


class TestQwen3EagleCapture(CustomTestCase):
    def test_qwen3_vl_keeps_native_layer_list_path(self):
        backbone = SimpleNamespace(
            capture_aux_hidden_states=False,
            layers_to_capture=[],
        )
        wrapper = _mock_wrapper(Qwen3VLForConditionalGeneration, backbone)

        wrapper.set_eagle3_layers_to_capture([1, 4, 8])

        self.assertTrue(wrapper.capture_aux_hidden_states)
        self.assertTrue(backbone.capture_aux_hidden_states)
        self.assertEqual(backbone.layers_to_capture, [2, 5, 9])

    def test_qwen3_5_marks_decoder_layers(self):
        layers = [SimpleNamespace() for _ in range(12)]
        backbone = Qwen3_5ForCausalLM.__new__(Qwen3_5ForCausalLM)
        backbone.layers = layers
        backbone.layers_to_capture = []
        wrapper = _mock_wrapper(Qwen3_5ForConditionalGeneration, backbone)

        wrapper.set_eagle3_layers_to_capture([1, 4, 8])

        self.assertTrue(wrapper.capture_aux_hidden_states)
        self.assertEqual(backbone.layers_to_capture, [2, 5, 9])
        self.assertTrue(layers[2]._is_layer_to_capture)
        self.assertTrue(layers[5]._is_layer_to_capture)
        self.assertTrue(layers[9]._is_layer_to_capture)


if __name__ == "__main__":
    unittest.main()
