import unittest

from sglang.srt.model_loader.weight_completeness import unloaded_required_params
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _never(name: str) -> bool:
    return False


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


if __name__ == "__main__":
    unittest.main()
