"""Unit tests for ``_model_load_weights`` -- CPU only, no GPU required.

Verifies the ``is_full_load`` kwarg is forwarded to loaders that accept it
and omitted for loaders with the legacy signature.
"""

import unittest

from sglang.srt.model_executor.model_runner import _model_load_weights
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
        return "legacy-loaded"


class TestModelLoadWeights(unittest.TestCase):
    def test_passes_full_load_flag_when_supported(self):
        model = _PartialAwareModel()
        weights = [("weight", object())]

        result = _model_load_weights(model, weights, is_full_load=False)

        self.assertEqual(result, "loaded")
        self.assertEqual(model.calls, [(weights, False)])

    def test_omits_full_load_flag_for_legacy_models(self):
        model = _LegacyModel()
        weights = [("weight", object())]

        result = _model_load_weights(model, weights, is_full_load=False)

        self.assertEqual(result, "legacy-loaded")
        self.assertEqual(model.calls, [weights])


if __name__ == "__main__":
    unittest.main()
