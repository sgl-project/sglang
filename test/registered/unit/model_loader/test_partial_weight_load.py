# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import patch

import torch

from sglang.srt.model_loader.loader import (
    QuantizedRLModelLoader,
    load_model_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _PartialAwareModel:
    def __init__(self):
        self.calls = []

    def load_weights(self, weights, *, is_full_load=True):
        self.calls.append((list(weights), is_full_load))
        return "loaded"


class _LegacyModel:
    def __init__(self):
        self.calls = []

    def load_weights(self, weights):
        self.calls.append(list(weights))
        return "loaded"


class TestPartialWeightLoad(unittest.TestCase):
    def test_forwards_partial_context_when_supported(self):
        model = _PartialAwareModel()
        weights = [("weight", object())]

        self.assertEqual(
            load_model_weights(model, weights, is_full_load=False), "loaded"
        )
        self.assertEqual(model.calls, [(weights, False)])

    def test_preserves_legacy_loaders(self):
        model = _LegacyModel()
        weights = [("weight", object())]

        self.assertEqual(
            load_model_weights(model, weights, is_full_load=False), "loaded"
        )
        self.assertEqual(model.calls, [weights])

    def test_quantized_rl_proxy_preserves_full_then_partial_context(self):
        class Model:
            def __init__(self):
                self.calls = []

            def load_weights(self, weights, *, is_full_load=True):
                self.calls.append((list(weights), is_full_load))

            def named_parameters(self):
                return iter([])

            def named_modules(self):
                return iter([])

        model = Model()
        loader = object.__new__(QuantizedRLModelLoader)
        loader._initial_load_complete = False

        loader.load_weights_and_postprocess(
            model,
            [("initial", torch.empty(1, dtype=torch.uint8))],
            torch.device("cpu"),
            is_full_load=True,
        )
        self.assertTrue(model.calls[0][1])

        def run_reload(_model, captured_load_weights, weights):
            captured_load_weights(weights)

        with patch.object(
            QuantizedRLModelLoader,
            "rebinding_and_load_weights",
            side_effect=run_reload,
        ) as reload_weights:
            loader.load_weights_and_postprocess(
                model,
                [("partial", torch.empty(1, dtype=torch.uint8))],
                torch.device("cpu"),
                is_full_load=False,
            )

        reload_weights.assert_called_once()
        self.assertEqual(
            [(weights[0][0], is_full_load) for weights, is_full_load in model.calls],
            [("initial", True), ("partial", False)],
        )


if __name__ == "__main__":
    unittest.main()
