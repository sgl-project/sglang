# SPDX-License-Identifier: Apache-2.0
"""Encoding MiniMax-H3 video while it decodes must not change the MP4."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.entrypoints.utils import (
    CudaVideoEncoder,
    _try_save_cuda_video_direct,
)
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3_video_vae import (
    AutoencoderKLLegacy,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")

FPS = 24
SAMPLE_RATE = 24000


def _temporal_vae() -> AutoencoderKLLegacy:
    """The released temporal chunking, with each clip decoded by repetition."""
    vae = AutoencoderKLLegacy.__new__(AutoencoderKLLegacy)
    torch.nn.Module.__init__(vae)
    vae.vae_ratio = 16
    vae.vae_ratio_t = 4
    vae.use_3d_conv = True
    vae.transform = None
    vae.transform_rev = None
    vae.setup_forward(clip_length=17, token_drop=3)
    vae._adaptive_decode = lambda clip_z: clip_z.repeat_interleave(4, dim=2)
    return vae.eval()


class TestDecodeHandsOutFinishedFrames(CustomTestCase):
    def test_every_frame_arrives_once_in_order_and_final(self):
        vae = _temporal_vae()
        latents = torch.randn(1, 3, 31, 2, 2)
        for stream_cat in ("1", "0"):
            with (
                self.subTest(stream_cat=stream_cat),
                mock.patch.dict(
                    os.environ,
                    {"MINIMAX_H3_VAE_DECODER_STREAM_TEMPORAL_CAT": stream_cat},
                ),
            ):
                parts = []
                video = vae.decode_base(
                    latents, on_frames=lambda frames: parts.append(frames.clone())
                )
                self.assertTrue(torch.equal(torch.cat(parts, dim=2), video))
                self.assertEqual(len(parts) > 1, stream_cat == "1")

    def test_trimmed_decodes_refuse_the_callback(self):
        with self.assertRaises(ValueError):
            _temporal_vae().decode_base(
                torch.randn(1, 3, 31, 2, 2), frame_num=100, on_frames=print
            )


@unittest.skipUnless(
    torch.cuda.is_available() and hasattr(os, "memfd_create"),
    "needs CUDA and memfd",
)
class TestStreamedEncode(CustomTestCase):
    @staticmethod
    def _video(height: int, width: int) -> torch.Tensor:
        generator = torch.Generator(device="cuda").manual_seed(0)
        return torch.rand(3, 30, height, width, device="cuda", generator=generator)

    def test_writes_in_pieces_match_the_one_shot_save(self):
        audio = torch.sin(torch.arange(2 * 2 * SAMPLE_RATE) / 7).reshape(2, -1)
        # 70x90 is not a macroblock multiple, so it also covers the scale filter
        for height, width, with_audio in ((64, 96, False), (70, 90, True)):
            video = self._video(height, width)
            sample_audio = audio if with_audio else None
            sample_rate = SAMPLE_RATE if with_audio else None
            with (
                self.subTest(size=(height, width)),
                tempfile.TemporaryDirectory() as tmp,
            ):
                one_shot, streamed = (
                    Path(tmp, "one_shot.mp4"),
                    Path(tmp, "streamed.mp4"),
                )
                self.assertTrue(
                    _try_save_cuda_video_direct(
                        save_file_path=str(one_shot),
                        sample=(video, sample_audio) if with_audio else video,
                        fps=FPS,
                        audio_sample_rate=sample_rate,
                        output_compression=None,
                    )
                )
                encoder = CudaVideoEncoder.open(
                    str(streamed),
                    device=video.device,
                    height=height,
                    width=width,
                    fps=FPS,
                    num_frames=video.shape[1],
                    audio=sample_audio,
                    audio_sample_rate=sample_rate,
                    max_queued_frames=64,
                )
                for start, end in ((0, 1), (1, 6), (6, 25), (25, 30)):
                    encoder.write(video[:, start:end])
                encoder.close()
                self.assertEqual(one_shot.read_bytes(), streamed.read_bytes())

    def test_concurrent_encodes_both_finish(self):
        # only one staging buffer is cached, so the other one is closed on release
        video = self._video(64, 96)
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp, f"clip_{i}.mp4") for i in range(2)]
            encoders = [
                CudaVideoEncoder.open(
                    str(path),
                    device=video.device,
                    height=64,
                    width=96,
                    fps=FPS,
                    num_frames=video.shape[1],
                )
                for path in paths
            ]
            for encoder in encoders:
                encoder.write(video)
            for encoder in encoders:
                encoder.close()
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_a_failed_encode_leaves_no_file(self):
        video = self._video(64, 96)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "clip.mp4")
            encoder = CudaVideoEncoder.open(
                str(path),
                device=video.device,
                height=64,
                width=96,
                fps=FPS,
                num_frames=video.shape[1],
            )
            encoder.write(video[:, :10])
            encoder._process.kill()
            encoder._process.wait()
            try:
                encoder.write(video[:, 10:])
            except RuntimeError:
                pass
            with self.assertRaises((RuntimeError, subprocess.CalledProcessError)):
                encoder.close()
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=3)
