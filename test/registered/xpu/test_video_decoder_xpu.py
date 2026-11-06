"""Video decode on the Intel XPU media engine through torchcodec-xpu.

Checks the frames ``VideoDecoderWrapper`` and ``load_video`` return on XPU,
compares them with the CPU decode, and records the CPU-time saving so a
regression to CPU decoding fails the test rather than only slowing it.
"""

import importlib.util
import os
import tempfile
import time
import unittest

import av
import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.utils.common import is_xpu, load_video
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.srt.utils.video_decoder_xpu import xpu_video_decode_device
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase, is_in_ci, write_github_step_summary

register_xpu_ci(est_time=60, suite="stage-a-test-1-gpu-xpu")

_WIDTH, _HEIGHT, _FPS, _SECONDS = 1280, 720, 30, 10
_SAMPLED = 60  # 2 fps over the clip, the Qwen-Omni sampling pattern


def _synthesize_h264_clip(path: str) -> None:
    """Moving-gradient H.264 clip; libx264 ships with PyAV's FFmpeg."""
    yy, xx = np.mgrid[0:_HEIGHT, 0:_WIDTH]
    with av.open(path, mode="w") as container:
        stream = container.add_stream("libx264", rate=_FPS)
        stream.width, stream.height, stream.pix_fmt = _WIDTH, _HEIGHT, "yuv420p"
        stream.options = {"preset": "ultrafast", "g": "30"}
        for i in range(_FPS * _SECONDS):
            frame = np.stack(
                [(xx + 3 * i) % 256, (yy + 2 * i) % 256, (xx // 4 + yy // 4 + i) % 256],
                axis=-1,
            ).astype(np.uint8)
            packets = stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24"))
            for packet in packets:
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _timed(fn):
    wall, cpu = time.perf_counter(), time.process_time()
    out = fn()
    return out, time.perf_counter() - wall, time.process_time() - cpu


@unittest.skipUnless(is_xpu(), "Intel XPU not available")
@unittest.skipUnless(
    importlib.util.find_spec("torchcodec_xpu") is not None,
    "torchcodec-xpu plugin not installed",
)
class TestVideoDecoderXPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.clip = os.path.join(cls.tmpdir.name, "clip.mp4")
        _synthesize_h264_clip(cls.clip)
        cls.indices = np.linspace(0, _FPS * _SECONDS - 1, _SAMPLED, dtype=int).tolist()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def setUp(self):
        self.enterContext(envs.SGLANG_ENABLE_XPU_VIDEO_HW_DECODE.override(True))
        xpu_video_decode_device.cache_clear()
        self.addCleanup(xpu_video_decode_device.cache_clear)

    def test_wrapper_decodes_on_xpu(self):
        with VideoDecoderWrapper(self.clip, device="xpu") as w:
            self.assertEqual(w._tc_kwargs.get("device"), "xpu")
            self.assertEqual(len(w), _FPS * _SECONDS)
            self.assertAlmostEqual(w.avg_fps, _FPS, places=2)
            self.assertEqual(w.frame_shape, (_HEIGHT, _WIDTH))
            frames = w.get_frames_at(self.indices[:4])
            tensor = w.get_frames_as_tensor(self.indices[:4])
        for value in (frames, tensor):
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(value.device.type, "xpu")
            self.assertEqual(value.dtype, torch.uint8)
            self.assertEqual(tuple(value.shape), (4, _HEIGHT, _WIDTH, 3))

    def test_load_video_uses_xpu_when_enabled(self):
        with load_video(self.clip, use_gpu=True) as w:
            self.assertEqual(w._tc_kwargs.get("device"), "xpu")
            self.assertEqual(w[0].device.type, "xpu")

    def test_env_switch_forces_cpu_decode(self):
        with envs.SGLANG_ENABLE_XPU_VIDEO_HW_DECODE.override(False):
            xpu_video_decode_device.cache_clear()
            with load_video(self.clip, use_gpu=True) as w:
                self.assertNotIn("device", w._tc_kwargs)
                self.assertIsInstance(w.get_frames_at([0, 1]), np.ndarray)

    def test_xpu_frames_match_cpu_decode(self):
        with VideoDecoderWrapper(self.clip, device="xpu") as gpu:
            on_xpu = gpu.get_frames_at(self.indices[:8]).cpu().numpy()
        with VideoDecoderWrapper(self.clip, device="cpu") as cpu:
            on_cpu = cpu.get_frames_at(self.indices[:8])
        diff = np.abs(on_xpu.astype(np.int16) - on_cpu.astype(np.int16))
        # Different YUV->RGB converters (VAAPI/SYCL vs swscale) round differently.
        self.assertLess(diff.mean(), 1.0)
        self.assertLessEqual(diff.max(), 8)

    def test_xpu_decode_saves_cpu_time(self):
        def decode(device):
            with VideoDecoderWrapper(self.clip, device=device) as w:
                out = w.get_frames_as_tensor(self.indices)
                if out.device.type != "cpu":
                    torch.xpu.synchronize()
                return out

        decode("xpu")  # warm the media engine and the plugin
        _, xpu_wall, xpu_cpu = _timed(lambda: decode("xpu"))
        _, cpu_wall, cpu_cpu = _timed(lambda: decode("cpu"))
        summary = (
            f"### XPU video decode ({_SAMPLED} frames of {_WIDTH}x{_HEIGHT} H.264)\n"
            "| path | wall ms | CPU ms |\n|---|---|---|\n"
            f"| CPU torchcodec (sglang parallel decoders) | {cpu_wall * 1e3:.0f} | {cpu_cpu * 1e3:.0f} |\n"
            f"| XPU media engine (torchcodec-xpu) | {xpu_wall * 1e3:.0f} | {xpu_cpu * 1e3:.0f} |\n"
        )
        print(summary)
        if is_in_ci():
            write_github_step_summary(summary)
        # CPU time is the stable signal on shared runners; wall time is not.
        # Measured ~30x less on Arc Pro B60, so 0.5 leaves room for slow hosts.
        self.assertLess(xpu_cpu, 0.5 * cpu_cpu)


if __name__ == "__main__":
    unittest.main()
