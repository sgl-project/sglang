import unittest

import torch

from sglang.srt.models.mimo_audio import group_by_length
from sglang.srt.multimodal.mm_utils import split_audio_length
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAudioChunkLengths(unittest.TestCase):
    def test_preserve_clip_boundaries(self):
        lengths = [0, 99, 100, 101, 200]
        self.assertEqual(
            [split_audio_length(n, 100) for n in lengths],
            [[], [99], [100], [100, 1], [100, 100]],
        )

    def test_packed_groups(self):
        lengths = torch.tensor([1, 3, 0, 5, 2])
        features = torch.arange(11)
        feature_groups, len_groups = group_by_length(features, lengths, max_length=4)
        self.assertEqual([g.tolist() for g in len_groups], [[1, 3, 0], [5], [2]])
        self.assertEqual(
            [g.tolist() for g in feature_groups],
            [[0, 1, 2, 3], [4, 5, 6, 7, 8], [9, 10]],
        )
        with self.assertRaisesRegex(ValueError, "Feature size mismatch"):
            group_by_length(features[:-1], lengths, max_length=4)


if __name__ == "__main__":
    unittest.main()
