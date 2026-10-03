"""Unit tests for the Nano Nemotron VL processor."""

import tempfile
import unittest
from fractions import Fraction

import av
import numpy as np

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.nano_nemotron_vl import NemotronH_Omni_Reasoning_V3
from sglang.srt.multimodal.processors.nano_nemotron_vl import (
    NanoNemotronVLImageProcessor,
)
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestNanoNemotronVLProcessor(CustomTestCase):
    def test_supports_nemotron_h_omni(self):
        self.assertIn(
            NemotronH_Omni_Reasoning_V3,
            NanoNemotronVLImageProcessor.models,
        )

    def test_video_timestamps_preserve_frame_rate_precision(self):
        fps = Fraction(24000, 1001)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/video.avi"
            with av.open(path, "w") as container:
                stream = container.add_stream("ffv1", rate=fps)
                stream.width = stream.height = 16
                stream.pix_fmt = "bgr0"
                for index in range(120):
                    # Lossless pixels identify each sampled frame independently.
                    pixels = np.full((16, 16, 3), index, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            with VideoDecoderWrapper(path) as video:
                frames, timestamps = NanoNemotronVLImageProcessor.parse_video(video)

        expected_timestamps = frames[:, 0, 0, 0].astype(float) / float(fps)
        np.testing.assert_allclose(timestamps, expected_timestamps, rtol=0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
