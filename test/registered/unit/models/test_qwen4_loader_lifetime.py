"""Loading must not retain replaced parameters until cyclic garbage collection."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models import qwen4_exp

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TinyPLE(qwen4_exp.Qwen4ExpNGramEmbedding):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.ngram_embedding = torch.nn.Embedding(
            4, 2, _weight=torch.ones(4, 2).to(torch.float8_e4m3fn)
        )
        self.ngram_embedding.org_vocab_size = 4
        self.ngram_embedding.shard_indices = SimpleNamespace(
            org_vocab_start_index=0, org_vocab_end_index=4
        )


class _TinyModel(torch.nn.Module):
    _load_qwen4_exp_ple_buffer = (
        qwen4_exp.Qwen4ExpForConditionalGeneration._load_qwen4_exp_ple_buffer
    )

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_experts=None, split_ngram_parts=2)
        self.weight = torch.nn.Parameter(torch.ones(2, 2))


class TestQwen4LoaderLifetime(CustomTestCase):
    def test_replaced_parameter_is_released_without_cyclic_gc(self):
        model = _TinyModel()
        old_weight = weakref.ref(model.weight)
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            loaded = qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights(model, [])
            self.assertEqual(loaded, set())
            # ModelOpt postprocessing also replaces parameters after load_weights.
            model.weight = torch.nn.Parameter(torch.zeros(2, 2))
            self.assertIsNone(old_weight())
        finally:
            gc.collect()
            if gc_was_enabled:
                gc.enable()

    def test_downcast_warning_is_once_per_load(self):
        model = _TinyModel()
        model.ple = _TinyPLE()
        shards = [
            (
                f"ple.ngram_embedding.shard_{i}.weight",
                torch.ones(2, 2, dtype=torch.bfloat16),
            )
            for i in range(2)
        ]
        with patch.object(qwen4_exp.logger, "warning") as warning:
            for expected_warnings in (1, 2):
                loaded = qwen4_exp.Qwen4ExpForConditionalGeneration.load_weights(
                    model, shards
                )
                self.assertEqual(loaded, {"ple.ngram_embedding.weight"})
                self.assertEqual(warning.call_count, expected_warnings)
        torch.testing.assert_close(
            model.ple.ngram_embedding.weight.float(), torch.ones(4, 2)
        )


if __name__ == "__main__":
    unittest.main()
