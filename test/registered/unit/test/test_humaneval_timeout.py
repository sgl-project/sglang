import builtins
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test import simple_eval_common
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHumanEvalTimeout(CustomTestCase):
    def test_evaluator_forwards_configured_and_default_timeout(self):
        observed = []
        problem = {"prompt": "def identity(x):\n", "task_id": "fixture/identity"}

        def check_correctness(sample, completion, timeout, completion_id):
            observed.append(timeout)
            return {"passed": True, "completion_id": completion_id}

        optional_modules = {
            "human_eval.data": SimpleNamespace(
                read_problems=lambda: {"fixture": problem}
            ),
            "human_eval.evaluation": SimpleNamespace(
                estimate_pass_at_k=lambda *args: [1.0]
            ),
            "human_eval.execution": SimpleNamespace(
                check_correctness=check_correctness
            ),
        }
        real_import = builtins.__import__

        def import_optional(name, *args, **kwargs):
            if name in optional_modules:
                return optional_modules[name]
            return real_import(name, *args, **kwargs)

        # Load a private module instance with only the optional external executor
        # boundary replaced. Do not install fake modules in the global registry.
        path = Path(simple_eval_common.__file__).with_name("simple_eval_humaneval.py")
        spec = importlib.util.spec_from_file_location("humaneval_timeout_fixture", path)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=import_optional):
            spec.loader.exec_module(module)

        class FixedSampler:
            _pack_message = staticmethod(lambda **kwargs: kwargs)

            def __call__(self, messages):
                return "def identity(x):\n    return x\n"

        for timeout_kwargs, expected in (({}, 120), ({"timeout": 17}, 17)):
            with self.subTest(timeout_kwargs=timeout_kwargs):
                observed.clear()
                with patch("builtins.__import__", side_effect=import_optional):
                    evaluator = module.HumanEval(
                        num_examples=1,
                        num_threads=1,
                        num_samples_per_task=1,
                        ks_passes=[1],
                        **timeout_kwargs,
                    )
                    result = evaluator(FixedSampler())
                self.assertEqual(result.score, 1.0)
                self.assertEqual(observed, [expected])


if __name__ == "__main__":
    unittest.main()
