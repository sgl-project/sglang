import unittest

from sglang.srt.model_loader.weight_completeness import (
    required_weight_names,
    unloaded_required_params,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _never(name: str) -> bool:
    return False


class _FakeModule:
    def __init__(self, buffers, non_persistent=()):
        self.buffers = buffers
        self._non_persistent_buffers_set = set(non_persistent)

    def named_buffers(self, recurse=False):
        return iter(self.buffers)


class _FakeModel:
    def named_parameters(self):
        return iter([("weight", object())])

    def named_modules(self):
        return iter(
            [
                (
                    "",
                    _FakeModule(
                        [("scale", object()), ("scratch", object())], {"scratch"}
                    ),
                ),
                ("sub", _FakeModule([("running", object())])),
            ]
        )


class TestUnloadedRequiredParams(unittest.TestCase):
    def test_reports_required_params_not_loaded(self):
        missing = unloaded_required_params(
            ["a.weight", "b.weight"], {"a.weight"}, _never
        )
        self.assertEqual(missing, {"b.weight"})

    def test_optional_params_are_excluded(self):
        missing = unloaded_required_params(
            ["a.weight", "b.k_scale"], set(), lambda n: n.endswith(".k_scale")
        )
        self.assertEqual(missing, {"a.weight"})

    def test_empty_when_all_required_loaded(self):
        missing = unloaded_required_params(
            ["a.weight", "b.weight"], {"a.weight", "b.weight"}, _never
        )
        self.assertEqual(missing, set())

    def test_persistent_buffers_are_required(self):
        model = _FakeModel()

        self.assertEqual(
            set(required_weight_names(model)), {"weight", "scale", "sub.running"}
        )
        missing = unloaded_required_params(
            required_weight_names(model), {"weight", "sub.running"}, _never
        )
        self.assertEqual(missing, {"scale"})


if __name__ == "__main__":
    unittest.main()
