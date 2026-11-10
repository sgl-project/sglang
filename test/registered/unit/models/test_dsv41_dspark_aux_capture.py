"""DSpark aux-hidden buffer must exist before CUDA-graph capture."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.deepseek_v4 import DeepseekV4Model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _stub_model() -> DeepseekV4Model:
    model = DeepseekV4Model.__new__(DeepseekV4Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=5120)
    model.hidden_size = 5120
    model.dspark_layers_to_capture = [37, 38, 39]
    model._dspark_aux_buf = None
    model._dspark_aux_n_tok = 0
    model.embed_tokens = nn.Embedding(8, 5120)
    model.embed_tokens.weight.data = model.embed_tokens.weight.data.to(torch.float16)
    return model


class TestDsv41DsparkAuxCapture(CustomTestCase):
    def test_alloc_then_store_during_capture(self):
        model = _stub_model()
        model.alloc_dspark_aux_hidden(6)
        self.assertEqual(tuple(model._dspark_aux_buf.shape), (3, 6, 5120))
        self.assertEqual(model._dspark_aux_buf.dtype, torch.float16)
        hs = torch.ones(6, 5120, dtype=torch.float16)
        with patch(
            "sglang.srt.models.deepseek_v4.get_is_capture_mode", return_value=True
        ):
            model._store_dspark_aux_hidden(37, hs)
        self.assertEqual(model._dspark_aux_n_tok, 6)
        self.assertTrue(torch.equal(model._dspark_aux_buf[0], hs))

    def test_store_during_capture_without_alloc_raises(self):
        model = _stub_model()
        hs = torch.ones(6, 5120, dtype=torch.float16)
        with patch(
            "sglang.srt.models.deepseek_v4.get_is_capture_mode", return_value=True
        ):
            with self.assertRaisesRegex(RuntimeError, r"need \[3, 6, 5120\]"):
                model._store_dspark_aux_hidden(37, hs)


if __name__ == "__main__":
    unittest.main()
