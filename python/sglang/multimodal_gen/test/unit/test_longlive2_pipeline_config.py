# SPDX-License-Identifier: Apache-2.0
import unittest

from sglang.multimodal_gen.configs.pipeline_configs.longlive2 import LongLive2T2VConfig


class TestLongLive2AdjustNumFrames(unittest.TestCase):
    def setUp(self):
        self.config = LongLive2T2VConfig()

    def test_reuses_wan_temporal_frame_adjustment(self):
        self.assertEqual(self.config.adjust_num_frames(62), 61)

    def test_keeps_frames_when_latents_match_causal_block(self):
        self.assertEqual(self.config.adjust_num_frames(93), 93)

    def test_rounds_to_causal_block_aligned_latents(self):
        self.assertEqual(self.config.adjust_num_frames(65), 61)


if __name__ == "__main__":
    unittest.main()
