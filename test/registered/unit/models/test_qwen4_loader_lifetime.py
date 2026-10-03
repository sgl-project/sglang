import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpNGramEmbedding,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def make_model(ple_dtype=None):
    # Exercise the real loader without constructing the distributed model.
    model = Qwen4ExpForConditionalGeneration.__new__(Qwen4ExpForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=False, split_ngram_parts=2)
    model.language_model_only = False
    model.weight = nn.Parameter(torch.zeros(4), requires_grad=False)
    if ple_dtype is not None:
        ple = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
        nn.Module.__init__(ple)
        ple.ngram_embedding = nn.Module()
        ple.ngram_embedding.weight = nn.Parameter(
            torch.zeros(4, 2, dtype=ple_dtype), requires_grad=False
        )
        ple.ngram_embedding.org_vocab_size = 4
        ple.ngram_embedding.shard_indices = SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=4
        )
        model.ple = ple
    return model


def ple_weights(dtype):
    return [
        (f"ple.ngram_embedding.shard_{i}.weight", torch.full((2, 2), i + 1.0).to(dtype))
        for i in range(2)
    ]


class TestQwen4LoaderLifetime(unittest.TestCase):
    def test_replaced_parameters_are_released_without_cyclic_gc(self):
        for with_ple in (False, True):
            with self.subTest(with_ple=with_ple):
                model = make_model(torch.float32 if with_ple else None)
                source = weakref.ref(model.weight)
                weights = [("weight", torch.arange(4.0))]
                expected_names = {"weight"}
                if with_ple:
                    weights.extend(ple_weights(torch.float32))
                    expected_names.add("ple.ngram_embedding.weight")
                gc_enabled = gc.isenabled()
                gc.disable()
                try:
                    self.assertEqual(model.load_weights(weights), expected_names)
                    torch.testing.assert_close(model.weight, torch.arange(4.0))
                    # Quantization post-processing replaces loader-format Parameters
                    # after load_weights returns. Its snapshot must not pin them.
                    model.weight = nn.Parameter(torch.ones(4), requires_grad=False)
                    self.assertIsNone(source())
                finally:
                    gc.collect()
                    if gc_enabled:
                        gc.enable()

    def test_ple_downcast_warns_once_per_load_and_preserves_values(self):
        model = make_model(torch.float8_e4m3fn)
        with patch("sglang.srt.models.qwen4_exp.logger.warning") as warning:
            for load_index in range(2):
                self.assertEqual(
                    model.load_weights(ple_weights(torch.bfloat16)),
                    {"ple.ngram_embedding.weight"},
                )
                self.assertEqual(warning.call_count, load_index + 1)
                self.assertIn("downcasting is lossy", warning.call_args.args[0])
                self.assertEqual(warning.call_args.args[1], torch.bfloat16)
                torch.testing.assert_close(
                    model.ple.ngram_embedding.weight.float(),
                    torch.tensor([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0]]),
                    rtol=0,
                    atol=0,
                )

    def test_ple_matching_dtype_does_not_warn(self):
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                model = make_model(dtype)
                with patch("sglang.srt.models.qwen4_exp.logger.warning") as warning:
                    model.load_weights(ple_weights(dtype))
                    warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
