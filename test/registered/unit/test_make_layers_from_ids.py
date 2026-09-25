import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch.nn as nn

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.utils.common import make_layers_from_ids
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMakeLayersFromIds(unittest.TestCase):
    def test_instantiates_only_owned_absolute_layer_ids(self):
        offloader = SimpleNamespace(
            wrap_modules=lambda modules, **kwargs: list(modules)
        )
        with patch(
            "sglang.srt.utils.offloader.get_offloader",
            return_value=offloader,
        ):
            layers = make_layers_from_ids(
                8,
                (0, 1, 4, 5),
                lambda idx, prefix: nn.Linear(1, 1),
                prefix="layers",
            )

        self.assertIsInstance(layers[0], nn.Linear)
        self.assertIsInstance(layers[1], nn.Linear)
        self.assertIsInstance(layers[2], PPMissingLayer)
        self.assertIsInstance(layers[3], PPMissingLayer)
        self.assertIsInstance(layers[4], nn.Linear)
        self.assertIsInstance(layers[5], nn.Linear)
        self.assertIsInstance(layers[6], PPMissingLayer)
        self.assertIsInstance(layers[7], PPMissingLayer)

    def test_rejects_duplicate_or_unsorted_layer_ids(self):
        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            make_layers_from_ids(8, (0, 4, 1), lambda idx, prefix: nn.Identity())

        with self.assertRaisesRegex(ValueError, "sorted and unique"):
            make_layers_from_ids(8, (0, 1, 1), lambda idx, prefix: nn.Identity())

    def test_rejects_out_of_range_layer_id(self):
        with self.assertRaisesRegex(ValueError, "outside the model"):
            make_layers_from_ids(8, (0, 8), lambda idx, prefix: nn.Identity())


if __name__ == "__main__":
    unittest.main()
