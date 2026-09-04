"""Regression tests for Bailing's qualified lm_head quantization prefix."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import torch

import sglang.srt.layers.vocab_parallel_embedding as embedding
import sglang.srt.models.bailing_moe_linear as bailing
from sglang.srt.layers.quantization.unquant import UnquantizedEmbeddingMethod
from sglang.test.test_utils import CustomTestCase


class _RecordingQuantConfig:
    def __init__(self):
        self.prefixes = []

    def get_quant_method(self, _layer, prefix):
        self.prefixes.append(prefix)
        return UnquantizedEmbeddingMethod()


class TestBailingLmHeadPrefix(CustomTestCase):
    def _construct(self, prefix):
        config = SimpleNamespace(
            tie_word_embeddings=False, vocab_size=128, hidden_size=128
        )
        parallel = SimpleNamespace(enable_dp_lm_head=False, tp_rank=0, tp_size=1)
        quant_config = _RecordingQuantConfig()

        # Keep Bailing's real lm_head construction and quantization dispatch,
        # while avoiding the transformer body and distributed initialization.
        with (
            patch.object(
                bailing, "BailingMoELinearModel", return_value=torch.nn.Identity()
            ),
            patch.object(bailing, "LogitsProcessor", return_value=torch.nn.Identity()),
            patch.object(
                bailing, "get_pp_group", return_value=SimpleNamespace(is_last_rank=True)
            ),
            patch.object(bailing, "get_parallel", return_value=parallel),
            patch.object(embedding, "get_parallel", return_value=parallel),
        ):
            bailing.BailingMoELinearForCausalLM(
                config=config, quant_config=quant_config, prefix=prefix
            )

        return quant_config.prefixes

    def test_lm_head_receives_qualified_prefix(self):
        self.assertEqual(self._construct("language_model"), ["language_model.lm_head"])

    def test_lm_head_prefix_is_unqualified_at_model_root(self):
        self.assertEqual(self._construct(""), ["lm_head"])


if __name__ == "__main__":
    unittest.main()
