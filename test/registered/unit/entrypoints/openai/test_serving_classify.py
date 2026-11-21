"""Unit tests for OpenAIServingClassify in serving_classify.py."""

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import unittest
from unittest.mock import MagicMock, Mock

import torch


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


def _make_tokenizer_manager(id2label=None, num_labels=None):
    tm = Mock()
    tm.model_path = "test-classifier"
    tm.served_model_name = "test-classifier"
    tm.server_args = Mock()
    tm.server_args.enable_cache_report = False
    tm.model_config = Mock()
    tm.model_config.is_multimodal = False
    hf_config = Mock()
    hf_config.id2label = id2label
    hf_config.num_labels = num_labels
    tm.model_config.hf_config = hf_config
    return tm


class TestOpenAIServingClassify(unittest.TestCase):
    def test_string_keyed_id2label_mapping(self):
        tm = _make_tokenizer_manager(id2label={"0": "NEGATIVE", "1": "POSITIVE"})
        serving = OpenAIServingClassify(tm, Mock())

        self.assertEqual(serving.id2label, {0: "NEGATIVE", 1: "POSITIVE"})

        resp = serving._build_classify_response(
            [
                {
                    "embedding": [-1.0, 2.0],
                    "meta_info": {"prompt_tokens": 4, "e2e_latency": 0.01},
                }
            ]
        )
        self.assertEqual(len(resp.data), 1)
        self.assertEqual(resp.data[0].label, "POSITIVE")
        self.assertEqual(resp.data[0].num_classes, 2)
        self.assertGreater(resp.data[0].probs[1], resp.data[0].probs[0])

    def test_out_of_range_id2label_and_tensor_embedding_fallback(self):
        tm = _make_tokenizer_manager(id2label={0: "LABEL_0", 1: "LABEL_1"})
        serving = OpenAIServingClassify(tm, Mock())

        # Passing a multi-element torch.Tensor whose argmax (index 2) is absent from id2label
        # must preserve softmax probs and fall back to "LABEL_2" instead of raising KeyError.
        resp = serving._build_classify_response(
            [
                {
                    "embedding": torch.tensor([0.1, 0.5, 3.0]),
                    "meta_info": {"prompt_tokens": 3},
                }
            ]
        )
        self.assertEqual(resp.data[0].label, "LABEL_2")
        self.assertEqual(resp.data[0].num_classes, 3)
        self.assertGreater(resp.data[0].probs[2], resp.data[0].probs[1])

    def test_validate_request_rejects_booleans_and_invalid_items(self):
        tm = _make_tokenizer_manager(id2label={0: "A", 1: "B"})
        serving = OpenAIServingClassify(tm, Mock())

        bool_first = ClassifyRequest.model_construct(
            model="test-classifier", input=[True, 1]
        )
        self.assertEqual(
            serving._validate_request(bool_first),
            "Input list items must be strings or integers",
        )

        bool_second = ClassifyRequest.model_construct(
            model="test-classifier", input=[1, False]
        )
        self.assertEqual(
            serving._validate_request(bool_second),
            "All items in input list must be integers",
        )

        neg_token = ClassifyRequest(model="test-classifier", input=[1, -2])
        self.assertEqual(
            serving._validate_request(neg_token),
            "Token ID at index 1 must be non-negative",
        )


if __name__ == "__main__":
    unittest.main()
