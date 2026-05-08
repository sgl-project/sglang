# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.srt.model_loader.weight_utils import _check_sharded_weight_files_complete


class TestWeightUtils(unittest.TestCase):
    def test_incomplete_sharded_weights_are_rejected(self):
        complete, message = _check_sharded_weight_files_complete(
            ["/tmp/model-00008-of-00008.safetensors"]
        )

        self.assertFalse(complete)
        self.assertIn("Missing 7 shard(s)", message)

    def test_complete_sharded_weights_are_accepted(self):
        complete, message = _check_sharded_weight_files_complete(
            [
                "/tmp/model-00001-of-00002.safetensors",
                "/tmp/model-00002-of-00002.safetensors",
            ]
        )

        self.assertTrue(complete)
        self.assertIsNone(message)

    def test_unsharded_weights_are_accepted(self):
        complete, message = _check_sharded_weight_files_complete(
            ["/tmp/model.safetensors"]
        )

        self.assertTrue(complete)
        self.assertIsNone(message)


if __name__ == "__main__":
    unittest.main()
