"""Tests for the shared multimodal feature materialization helper."""

import unittest

import torch

from sglang.srt.multimodal.mm_utils import materialize_multimodal_features
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestFeatureMaterialization(CustomTestCase):
    def test_packs_variable_length_features_and_converts_dtype(self):
        features = [
            torch.arange(6, dtype=torch.float32).view(2, 3),
            torch.arange(9, dtype=torch.float32).view(3, 3) + 10,
        ]

        result = materialize_multimodal_features(
            features, device=torch.device("cpu"), dtype=torch.bfloat16
        )

        self.assertEqual(result.shape, (5, 3))
        self.assertEqual(result.dtype, torch.bfloat16)
        torch.testing.assert_close(result.float(), torch.cat(features, dim=0))

    def test_rejects_incompatible_trailing_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching trailing shapes"):
            materialize_multimodal_features(
                [torch.empty(2, 3), torch.empty(1, 4)],
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
