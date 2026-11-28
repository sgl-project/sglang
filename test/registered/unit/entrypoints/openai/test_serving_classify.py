"""
Unit tests for the OpenAIServingClassify class from serving_classify.py.
"""

import asyncio
import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock

import torch
import torch.nn.functional as F


# Stub out sgl_kernel (and all submodules) before any sglang import so
# the test runs on CPU-only runners without the real CUDA library.
class _SglKernelMockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__loader__ = self
        mod.__getattr__ = lambda name: MagicMock()
        return mod

    def exec_module(self, module):
        pass


class _SglKernelMockFinder(importlib.abc.MetaPathFinder):
    """Import hook that intercepts all sgl_kernel.* imports and returns mocks."""

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

from sglang.srt.entrypoints.openai.protocol import ClassifyRequest  # noqa: E402
from sglang.srt.entrypoints.openai.serving_classify import (  # noqa: E402
    OpenAIServingClassify,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="extra-test-cpu-intel")


class _MockTokenizerManager:
    def __init__(self, hf_config, outputs):
        self.model_config = Mock()
        self.model_config.hf_config = hf_config
        self.served_model_name = "test-classifier"
        self.model_path = "test-classifier"
        self.server_args = None
        self.request_logger = Mock(log_requests=False, log_requests_level=0)
        self.outputs = outputs

        async def generate_request(adapted_request, raw_request=None):
            texts = adapted_request.text
            texts = [texts] if isinstance(texts, str) else texts
            yield [
                {
                    "embedding": self.outputs.get(t, [0.0]),
                    "meta_info": {"prompt_tokens": len(t.split())},
                }
                for t in texts
            ]

        self.generate_request = generate_request


class TestOpenAIServingClassify(unittest.TestCase):
    def test_single_logit_classifier_applies_sigmoid(self):
        """Single-logit classifiers (num_labels=1) must apply sigmoid rather than softmax."""
        hf_config = Mock(
            num_labels=1,
            problem_type="single_label_classification",
            id2label={0: "NEGATIVE"},
        )
        outputs = {"hello": [0.0409], "world": [-0.0807]}
        tm = _MockTokenizerManager(hf_config, outputs)
        serving = OpenAIServingClassify(tm, template_manager=None)

        req = ClassifyRequest(model="test-classifier", input=["hello", "world"])
        resp = asyncio.run(serving.handle_request(req, None))

        expected_0 = torch.sigmoid(torch.tensor([0.0409])).tolist()
        expected_1 = torch.sigmoid(torch.tensor([-0.0807])).tolist()

        self.assertAlmostEqual(resp.data[0].probs[0], expected_0[0], places=4)
        self.assertAlmostEqual(resp.data[1].probs[0], expected_1[0], places=4)
        self.assertNotEqual(resp.data[0].probs, [1.0])
        self.assertEqual(resp.data[0].label, "NEGATIVE")

    def test_regression_single_target_returns_raw(self):
        """Regression heads with single target must return raw score without activation."""
        hf_config = Mock(num_labels=1, problem_type="regression", id2label=None)
        outputs = {"score this": [4.25]}
        tm = _MockTokenizerManager(hf_config, outputs)
        serving = OpenAIServingClassify(tm, template_manager=None)

        req = ClassifyRequest(model="test-classifier", input="score this")
        resp = asyncio.run(serving.handle_request(req, None))

        self.assertEqual(resp.data[0].probs, [4.25])
        self.assertEqual(resp.data[0].label, "LABEL_0")

    def test_regression_multi_target_returns_raw(self):
        """Regression heads with multi-target must return raw scores without activation."""
        hf_config = Mock(
            num_labels=3, problem_type="regression", id2label={0: "x", 1: "y", 2: "z"}
        )
        outputs = {"point": [0.1345, 0.1886, -0.0186]}
        tm = _MockTokenizerManager(hf_config, outputs)
        serving = OpenAIServingClassify(tm, template_manager=None)

        req = ClassifyRequest(model="test-classifier", input="point")
        resp = asyncio.run(serving.handle_request(req, None))

        self.assertEqual(len(resp.data[0].probs), 3)
        for p, exp in zip(resp.data[0].probs, [0.1345, 0.1886, -0.0186]):
            self.assertAlmostEqual(p, exp, places=4)
        self.assertEqual(resp.data[0].label, "y")

    def test_multi_class_applies_softmax(self):
        """Multi-class classification applies softmax across logits."""
        hf_config = Mock(
            num_labels=3, problem_type="single_label_classification", id2label=None
        )
        outputs = {"text": [1.0, 2.0, 3.0]}
        tm = _MockTokenizerManager(hf_config, outputs)
        serving = OpenAIServingClassify(tm, template_manager=None)

        req = ClassifyRequest(model="test-classifier", input="text")
        resp = asyncio.run(serving.handle_request(req, None))

        expected = F.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0).tolist()
        self.assertEqual(len(resp.data[0].probs), 3)
        for actual_p, exp_p in zip(resp.data[0].probs, expected):
            self.assertAlmostEqual(actual_p, exp_p, places=4)
        self.assertEqual(resp.data[0].label, "LABEL_2")

    def test_multi_label_applies_sigmoid(self):
        """Multi-label classification applies elementwise sigmoid across logits."""
        hf_config = Mock(
            num_labels=3, problem_type="multi_label_classification", id2label=None
        )
        outputs = {"text": [1.0, -2.0, 0.5]}
        tm = _MockTokenizerManager(hf_config, outputs)
        serving = OpenAIServingClassify(tm, template_manager=None)

        req = ClassifyRequest(model="test-classifier", input="text")
        resp = asyncio.run(serving.handle_request(req, None))

        expected = torch.sigmoid(torch.tensor([1.0, -2.0, 0.5])).tolist()
        self.assertEqual(len(resp.data[0].probs), 3)
        for actual_p, exp_p in zip(resp.data[0].probs, expected):
            self.assertAlmostEqual(actual_p, exp_p, places=4)

    def test_reward_model_single_score_activation(self):
        """Reward models with single logit output apply sigmoid when problem_type is None."""
        hf_config = Mock(num_labels=None, problem_type=None, id2label=None)
        outputs = {"high": [4.25], "low": [-3.5]}
        tm = _MockTokenizerManager(hf_config, outputs)
        serving = OpenAIServingClassify(tm, template_manager=None)

        req = ClassifyRequest(model="test-classifier", input=["high", "low"])
        resp = asyncio.run(serving.handle_request(req, None))

        self.assertAlmostEqual(
            resp.data[0].probs[0], torch.sigmoid(torch.tensor([4.25])).item(), places=4
        )
        self.assertAlmostEqual(
            resp.data[1].probs[0], torch.sigmoid(torch.tensor([-3.5])).item(), places=4
        )
        self.assertNotEqual(resp.data[0].probs, resp.data[1].probs)

    def test_validate_request(self):
        """Verify request validation rejects empty and invalid inputs."""
        hf_config = Mock(num_labels=2, problem_type=None, id2label=None)
        tm = _MockTokenizerManager(hf_config, {})
        serving = OpenAIServingClassify(tm, template_manager=None)

        self.assertIsNotNone(serving._validate_request(ClassifyRequest(input="   ")))
        self.assertIsNotNone(
            serving._validate_request(ClassifyRequest(input=["valid", "  "]))
        )
        self.assertIsNone(
            serving._validate_request(ClassifyRequest(input="valid string"))
        )
        self.assertIsNone(
            serving._validate_request(ClassifyRequest(input=["valid", "list"]))
        )


if __name__ == "__main__":
    unittest.main()
