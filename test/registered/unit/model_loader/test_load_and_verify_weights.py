# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.srt.model_loader.weight_completeness import (
    load_and_verify_weights,
    unloaded_required_params,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeModel:
    def __init__(
        self,
        *,
        loaded=None,
        parameter_names=("required.weight", "optional.weight"),
        verify_weights_on_load=True,
    ):
        self.loaded = loaded
        self.parameter_names = parameter_names
        self.verify_weights_on_load = verify_weights_on_load
        self.seen_weights = None

    def load_weights(self, weights):
        self.seen_weights = list(weights)
        return self.loaded

    def named_parameters(self):
        return ((name, object()) for name in self.parameter_names)

    @staticmethod
    def is_optional_weight(name):
        return name.startswith("optional.")


class TestUnloadedRequiredParams(unittest.TestCase):
    def test_reports_only_unloaded_required_params(self):
        self.assertEqual(
            unloaded_required_params(
                ["loaded.weight", "required.weight", "optional.weight"],
                {"loaded.weight"},
                lambda name: name.startswith("optional."),
            ),
            {"required.weight"},
        )


class TestLoadAndVerifyWeights(unittest.TestCase):
    def test_full_load_verifies_migrated_model(self):
        model = _FakeModel(loaded={"optional.weight"})

        with self.assertRaisesRegex(RuntimeError, r"required\.weight"):
            load_and_verify_weights(model, [("checkpoint.weight", object())])

        self.assertEqual(model.seen_weights[0][0], "checkpoint.weight")

    def test_partial_load_can_skip_full_checkpoint_verification(self):
        model = _FakeModel(loaded=set())

        self.assertEqual(
            load_and_verify_weights(model, [], is_full_checkpoint=False),
            set(),
        )

    def test_unmigrated_model_preserves_none_return(self):
        model = _FakeModel(loaded=None, verify_weights_on_load=False)

        self.assertIsNone(load_and_verify_weights(model, []))

    def test_migrated_model_must_return_loaded_names(self):
        model = _FakeModel(loaded=None)

        with self.assertRaisesRegex(TypeError, "must return loaded parameter names"):
            load_and_verify_weights(model, [])

    def test_complete_load_returns_loaded_names(self):
        loaded = {"required.weight"}
        model = _FakeModel(loaded=loaded)

        self.assertIs(load_and_verify_weights(model, []), loaded)


if __name__ == "__main__":
    unittest.main()
