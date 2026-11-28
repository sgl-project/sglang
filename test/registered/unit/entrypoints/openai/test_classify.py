"""Unit tests for the OpenAIServingClassify class from serving_classify.py.

Regression tests for issue #52618: /v1/classify returned probs=[1.0] for every
input on single-output heads (softmax over one logit is always 1) and turned
regression head scores into a distribution, losing their values and signs.
"""

import importlib.abc
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import Mock


# Stub out sgl_kernel (and all submodules) before any sglang import so
# the test runs on CPU-only runners without the real CUDA library.
class _SglKernelMockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__loader__ = self
        mod.__getattr__ = lambda name: Mock()
        return mod

    def exec_module(self, module):
        pass


class _SglKernelMockFinder(importlib.abc.MetaPathFinder):
    _PREFIX = "sgl_kernel"
    _loader = _SglKernelMockLoader()

    def find_spec(self, fullname, path, target=None):
        if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
            return importlib.machinery.ModuleSpec(
                fullname, self._loader, is_package=True
            )
        return None


if "sgl_kernel" not in sys.modules:
    sys.meta_path.insert(0, _SglKernelMockFinder())

from sglang.srt.entrypoints.openai.protocol import ClassifyRequest
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=8, suite="base-c-test-cpu")


class _MockHfConfig:
    def __init__(self, id2label, problem_type=None):
        self.id2label = id2label
        self.problem_type = problem_type


class _MockTokenizerManager:
    def __init__(self, id2label, problem_type=None, head_outputs=None):
        self.model_config = Mock()
        self.model_config.hf_config = _MockHfConfig(id2label, problem_type)
        self.model_config.embedding_model_spec = Mock()
        self.model_config.embedding_model_spec.task = None
        self.served_model_name = "test-model"
        self.server_args = Mock()
        self.server_args.model_path = "test-model"

        async def mock_generate_request(adapted_request, raw_request):
            yield {
                "embedding": head_outputs,
                "meta_info": {"prompt_tokens": len(str(adapted_request.text))},
            }

        self.generate_request = Mock(return_value=mock_generate_request())


class ServingClassifyTestCase(unittest.TestCase):
    """Regression tests for issue #52618."""

    def _build(self, id2label, problem_type, head_outputs):
        tokenizer_manager = _MockTokenizerManager(
            id2label, problem_type=problem_type, head_outputs=head_outputs
        )
        serving = OpenAIServingClassify(tokenizer_manager, template_manager=None)
        resp = serving._build_classify_response(
            [{"embedding": head_outputs, "meta_info": {"prompt_tokens": 1}}]
        )
        return resp.data[0]

    def test_single_label_uses_sigmoid_not_constant_one(self):
        """A softmax over one logit is always 1.0; sigmoid keeps the score."""
        item = self._build({0: "LABEL_0"}, None, [0.0409])
        self.assertAlmostEqual(item["probs"][0], 0.5102, places=3)
        self.assertNotEqual(item["probs"], [1.0])
        self.assertEqual(item["label"], "LABEL_0")

    def test_single_label_negative_logit_below_half(self):
        item = self._build({0: "LABEL_0"}, None, [-3.5])
        self.assertLess(item["probs"][0], 0.5)

    def test_regression_single_target_returns_raw_score(self):
        """Regression heads must return raw scores, not a softmax distribution."""
        item = self._build({0: "LABEL_0"}, "regression", [0.0409])
        self.assertAlmostEqual(item["probs"][0], 0.0409, places=6)

    def test_regression_multi_target_preserves_values_and_signs(self):
        item = self._build(
            {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, "regression",
            [0.1345, 0.1886, -0.0186],
        )
        self.assertAlmostEqual(item["probs"][0], 0.1345, places=6)
        self.assertAlmostEqual(item["probs"][1], 0.1886, places=6)
        self.assertAlmostEqual(item["probs"][2], -0.0186, places=6)

    def test_multi_class_uses_softmax(self):
        """Multi-class heads keep the softmax behavior."""
        import torch
        import torch.nn.functional as F

        head = [0.1345, 0.1886, -0.0186]
        item = self._build({0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"}, None, head)
        expected = F.softmax(torch.tensor(head, dtype=torch.float32), dim=-1).tolist()
        self.assertEqual(len(item["probs"]), 3)
        for got, exp in zip(item["probs"], expected):
            self.assertAlmostEqual(got, exp, places=6)
        self.assertEqual(item["label"], "LABEL_1")

    def test_multi_label_classification_keeps_sigmoid(self):
        """Multi-label problem_type keeps the sigmoid behavior."""
        item = self._build(
            {0: "neutral", 1: "positive", 2: "negative"}, "multi_label_classification",
            [0.1345, 0.1886, -0.0186],
        )
        self.assertEqual(len(item["probs"]), 3)
        for got, exp in zip(
            item["probs"],
            torch.sigmoid(torch.tensor([0.1345, 0.1886, -0.0186])).tolist(),
        ):
            self.assertAlmostEqual(got, exp, places=6)

    def test_missing_scores_raises(self):
        tokenizer_manager = _MockTokenizerManager({0: "LABEL_0"}, None, None)
        serving = OpenAIServingClassify(tokenizer_manager, template_manager=None)
        with self.assertRaises(RuntimeError):
            serving._build_classify_response([{"meta_info": {"prompt_tokens": 1}}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
