"""Regression tests for streaming Bailing multimodal weight dispatch."""

import unittest

import torch
import torch.nn as nn

from sglang.srt.models.bailing_mm import BailingMMNativeForConditionalGeneration
from sglang.srt.models.bailing_mm_v3 import (
    BailingMoeV3VLForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _OneShotWeights:
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("checkpoint iterator was consumed more than once")
        return iter(self.values)


class _TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.register_parameter("text_weight", nn.Parameter(torch.zeros(2, 2)))

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, value in weights:
            params[name].data.copy_(value)
            loaded.add(name)
        return loaded


class TestBailingVLWeightLoading(CustomTestCase):
    @staticmethod
    def _wrapper(wrapper_class):
        wrapper = wrapper_class.__new__(wrapper_class)
        nn.Module.__init__(wrapper)
        wrapper.model = _TextModel()
        wrapper.linear_proj = nn.Linear(2, 2)
        wrapper._build_mm_encoders = True
        if wrapper_class is BailingMoeV3VLForConditionalGeneration:
            wrapper.visual = nn.Identity()
        else:
            wrapper.vision = nn.Identity()
        return wrapper

    @staticmethod
    def _weights():
        return [
            ("model.model.text_weight", torch.ones(2, 2)),
            ("model.linear_proj.weight", torch.full((2, 2), 2.0)),
            ("model.linear_proj.bias", torch.full((2,), 3.0)),
        ]

    def test_legacy_and_v3_loaders_consume_checkpoint_once(self):
        """A streaming checkpoint iterable must never be duplicated with tee."""
        for wrapper_class in (
            BailingMMNativeForConditionalGeneration,
            BailingMoeV3VLForConditionalGeneration,
        ):
            with self.subTest(wrapper_class=wrapper_class.__name__):
                wrapper = self._wrapper(wrapper_class)
                weights = _OneShotWeights(self._weights())

                wrapper.load_weights(weights)

                self.assertEqual(weights.iterations, 1)
                torch.testing.assert_close(
                    wrapper.linear_proj.weight, torch.full((2, 2), 2.0)
                )
                torch.testing.assert_close(
                    wrapper.linear_proj.bias, torch.full((2,), 3.0)
                )

    def test_required_multimodal_weight_coverage_is_enforced(self):
        """A truncated checkpoint must fail instead of leaving random projection bias."""
        wrapper = self._wrapper(BailingMoeV3VLForConditionalGeneration)
        weights = _OneShotWeights(self._weights()[:-1])

        with self.assertRaisesRegex(
            RuntimeError, "Missing required Bailing VL weights"
        ):
            wrapper.load_weights(weights)


if __name__ == "__main__":
    unittest.main()
