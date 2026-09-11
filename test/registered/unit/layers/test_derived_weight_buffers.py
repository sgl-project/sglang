"""Unit tests for rebuilding weight-derived buffers after a bulk weight load.

Regression guard: GemmaRMSNorm caches weight + 1 in a non-persistent buffer
(gemma_weight) that forward actually reads, and only _weight_loader filled it
in. Loaders that overwrite parameters in bulk skip those per-parameter hooks --
and R-Fork's remote_instance path additionally transfers only
named_parameters(), which excludes buffers. A Qwen3.5-35B-A3B instance loaded
over R-Fork came up healthy with all 101 Gemma-style norms still at their
ones_like init, degrading every norm to w = 1.0 while weight itself looked
correct; greedy decoding diverged from the seed on 8/8 prompts with no error
anywhere.
"""

import unittest

import torch

from sglang.srt.layers.layernorm import GemmaRMSNorm, refresh_derived_weight_buffers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDerivedWeightBuffers(CustomTestCase):
    def test_bulk_weight_overwrite_needs_refresh(self):
        norm = GemmaRMSNorm(hidden_size=8)
        trained = torch.arange(8, dtype=norm.weight.dtype) * 0.25

        # A bulk loader writes the parameter directly, bypassing _weight_loader.
        norm.weight.data.copy_(trained)
        self.assertFalse(
            torch.equal(norm.gemma_weight, trained + 1.0),
            "precondition: the cache must be stale before the refresh",
        )

        refresh_derived_weight_buffers(norm)

        torch.testing.assert_close(norm.gemma_weight, trained + 1.0)

    def test_walker_reaches_nested_norms(self):
        # The loader hands in the whole model, so the walk must descend; a
        # top-level-only implementation would silently miss every real norm.
        model = torch.nn.Sequential(
            torch.nn.Sequential(GemmaRMSNorm(hidden_size=4)),
            GemmaRMSNorm(hidden_size=4),
        )
        for norm in (model[0][0], model[1]):
            norm.weight.data.fill_(2.0)

        refresh_derived_weight_buffers(model)

        for norm in (model[0][0], model[1]):
            torch.testing.assert_close(
                norm.gemma_weight, torch.full_like(norm.gemma_weight, 3.0)
            )

    def test_weight_loader_path_still_fills_the_cache(self):
        # The normal load path must keep working through the extracted method.
        norm = GemmaRMSNorm(hidden_size=4)
        loaded = torch.tensor([0.5, 1.5, -0.5, 0.0], dtype=norm.weight.dtype)

        norm.weight.weight_loader(norm.weight, loaded)

        torch.testing.assert_close(norm.weight.data, loaded)
        torch.testing.assert_close(norm.gemma_weight, loaded + 1.0)

    def test_refresh_keeps_buffer_storage(self):
        # CUDA graphs and fused paths capture this buffer's address, so the
        # refresh must write in place rather than rebind the attribute.
        norm = GemmaRMSNorm(hidden_size=4)
        before = norm.gemma_weight.data_ptr()

        norm.weight.data.fill_(1.0)
        refresh_derived_weight_buffers(norm)

        self.assertEqual(norm.gemma_weight.data_ptr(), before)


if __name__ == "__main__":
    unittest.main()
