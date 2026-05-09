# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.omni.configs.sensenova_u1 import SenseNovaU1OmniPlugin
from sglang.omni.protocol import OmniInputSegment, OmniRequest


class TestSenseNovaU1OmniConfig(unittest.TestCase):
    def test_sampling_payload_request_metadata_is_split_before_sampling_build(self):
        plugin = SenseNovaU1OmniPlugin()
        request = OmniRequest(
            messages=(OmniInputSegment(type="text", text="describe"),),
            sampling_params={
                "task": "vlm",
                "max_new_tokens": 4,
                "max_interleave_images": 2,
                "think": "true",
                "num_steps": 3,
            },
        )

        normalized = plugin.normalize_request(request)

        self.assertEqual("vlm", normalized.mode)
        self.assertEqual(4, normalized.metadata["max_new_tokens"])
        self.assertEqual(2, normalized.max_images)
        self.assertTrue(normalized.think)
        self.assertEqual(3, normalized.sampling_params.num_inference_steps)


if __name__ == "__main__":
    unittest.main()
