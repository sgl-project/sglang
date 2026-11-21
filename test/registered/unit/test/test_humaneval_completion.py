import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.simple_eval_humaneval import _extract_completion
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHumanEvalCompletion(CustomTestCase):
    def test_valid_python_function_formats_preserve_correctness(self):
        prompt = 'def identity(x):\n    """Return the input."""\n'
        completions = [
            "def identity(x): return x\n",
            "def identity(x):\n\treturn x\n",
            "    return x\n",
            "```python\ndef identity(x): return x\n```",
            "def identity(\n    x,\n):\n    return x\n",
            "def identity(x):\n    return helper(x)\ndef helper(x):\n    return x\n",
        ]
        for completion in completions:
            with self.subTest(completion=completion):
                # Only execute the fixed identity fixture above, never model output.
                namespace = {}
                exec(prompt + _extract_completion(completion), namespace)
                self.assertEqual(namespace["identity"](42), 42)

    def test_nested_blocks_keep_relative_indentation(self):
        completion = "def identity(x):\n    if x:\n        return x\n    return 0\n"
        self.assertEqual(
            _extract_completion(completion),
            "    if x:\n        return x\n    return 0\n",
        )


if __name__ == "__main__":
    unittest.main()
