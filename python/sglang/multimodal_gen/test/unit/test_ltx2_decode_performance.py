import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding_av import (
    LTX2AVDecodingStage,
)


class TestLTX2DecodePerformance(unittest.TestCase):
    def test_ltx2_video_postprocess_returns_uint8_bthwc_frames(self):
        video = torch.tensor(
            [
                [
                    [[[-1.0, 0.0], [1.0, 2.0]]],
                    [[[0.5, -0.5], [0.0, 1.0]]],
                    [[[1.0, 0.0], [-1.0, -2.0]]],
                ]
            ],
            dtype=torch.float32,
        )

        out = LTX2AVDecodingStage._postprocess_video_to_uint8_np(video)
        expected = (
            ((video / 2 + 0.5).clamp(0, 1) * 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 4, 1)
            .numpy()
        )

        self.assertEqual(out.shape, (1, 1, 2, 2, 3))
        self.assertEqual(out.dtype.name, "uint8")
        self.assertTrue((out == expected).all())


if __name__ == "__main__":
    unittest.main()
