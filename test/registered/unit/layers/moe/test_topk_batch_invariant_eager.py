"""In batch-invariant mode the torch.compile'd MoE routing functions run eagerly.

Inductor tunes the compiled routing kernels by timing on first use, so their
summation order, and the selected experts, can differ between server starts and
with the batch. Deterministic inference needs the eager path.
"""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.moe.topk as topk_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROUTING_FUNCTIONS = (
    "grouped_topk_gpu",
    "kimi_k2_biased_topk_impl",
    "biased_topk_impl",
    "biased_grouped_topk_impl",
    "_biased_grouped_topk_postprocess",
)


def _compiled_must_not_run(*args, **kwargs):
    raise AssertionError("the compiled routing function ran in batch-invariant mode")


class TestTopkBatchInvariantEager(CustomTestCase):
    def test_routing_functions_are_wrapped(self):
        for name in ROUTING_FUNCTIONS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(getattr(topk_module, name), "compiled"))

    def test_batch_invariant_mode_runs_eagerly(self):
        torch.manual_seed(0)
        hidden_states = torch.randn(5, 16)
        gating_output = torch.randn(5, 64)
        fn = topk_module.grouped_topk_gpu
        args = (hidden_states, gating_output, 6, True, 8, 3)

        with (
            patch.object(
                topk_module, "is_batch_invariant_mode_enabled", return_value=True
            ),
            patch.object(fn, "compiled", _compiled_must_not_run),
        ):
            weights, ids = fn(*args)
        eager_weights, eager_ids = fn.__wrapped__(*args)
        self.assertTrue(torch.equal(weights, eager_weights))
        self.assertTrue(torch.equal(ids, eager_ids))

    def test_normal_mode_uses_the_compiled_function(self):
        sentinel = object()
        fn = topk_module.grouped_topk_gpu
        with (
            patch.object(
                topk_module, "is_batch_invariant_mode_enabled", return_value=False
            ),
            patch.object(fn, "compiled", return_value=sentinel),
        ):
            self.assertIs(fn(torch.randn(2, 4), torch.randn(2, 8), 2, True), sentinel)


if __name__ == "__main__":
    unittest.main()
