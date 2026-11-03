"""Device selection and output contract of ``VideoDecoderWrapper``.

The torchcodec boundary is faked so the assertions check what the wrapper
asks torchcodec for and what it hands back, not whether a mock was called.
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt import platforms
from sglang.srt.environ import envs
from sglang.srt.utils import common
from sglang.srt.utils import video_decoder as vd
from sglang.srt.utils import video_decoder_xpu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeTorchcodecDecoder:
    """Stands in for ``torchcodec.decoders.VideoDecoder``.

    Frames are reported on the requested device; a device listed in
    ``failing_devices`` raises the RuntimeError torchcodec raises when a
    device backend cannot open the source.
    """

    failing_devices: set = set()
    num_frames = 8
    cpu_fallback = (
        False  # torchcodec's CpuFallbackStatus is falsy when decoding on the device
    )

    def __init__(self, source, **kwargs):
        device = kwargs.get("device", "cpu")
        if device in self.failing_devices:
            raise RuntimeError(f"no decoder for {device}")
        self.kwargs = kwargs
        self.metadata = SimpleNamespace(average_fps=30.0, height=4, width=6)
        self._device = torch.device("meta" if device != "cpu" else "cpu")

    def _frames(self, n):
        return torch.zeros((n, 4, 6, 3), dtype=torch.uint8, device=self._device)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        return SimpleNamespace(data=self._frames(1)[0])

    def get_frames_at(self, indices):
        return SimpleNamespace(data=self._frames(len(indices)))


class _FakeDeviceModule:
    """Stands in for torch.xpu / torch.cuda on hosts without the accelerator."""

    @staticmethod
    def current_device():
        return 0


class _FakePlatform:
    """Stands in for current_platform, whose device calls need real hardware."""

    @staticmethod
    def get_device(device_id=0):
        return f"fake:{device_id}"

    @staticmethod
    def set_device(device):
        pass


@contextmanager
def _patched(*, xpu_ok, failing=()):
    """Fake the torchcodec and accelerator-runtime boundaries for the duration."""
    _FakeTorchcodecDecoder.failing_devices = set(failing)
    with (
        patch.object(vd, "VideoDecoder", _FakeTorchcodecDecoder, create=True),
        patch.object(vd, "_BACKEND", "torchcodec"),
        patch.object(vd, "_try_cuda_backend", lambda: True),
        patch.object(vd, "_try_xpu_backend", lambda: xpu_ok),
        patch.object(torch, "get_device_module", lambda device_type: _FakeDeviceModule),
        patch.object(platforms, "current_platform", _FakePlatform, create=True),
    ):
        yield


def _wrapper(device, *, xpu_ok, failing=()):
    with _patched(xpu_ok=xpu_ok, failing=failing):
        return vd.VideoDecoderWrapper(b"not-a-real-video", device=device)


class TestVideoDecoderWrapperDevice(CustomTestCase):
    def test_xpu_decoders_per_video_follow_the_concurrency_limit(self):
        # The default of 1 concurrent hardware decode means one decoder, not 16 CPU-style ones.
        self.assertEqual(_wrapper("xpu", xpu_ok=True)._num_decode_threads, 1)
        with envs.SGLANG_XPU_VIDEO_MAX_CONCURRENT_DECODES.override(2):
            w = _wrapper("xpu", xpu_ok=True)
            self.assertEqual(
                w._num_decode_threads, video_decoder_xpu.XPU_DECODERS_PER_CARD
            )
        with envs.SGLANG_XPU_VIDEO_MAX_CONCURRENT_DECODES.override(8):
            # Never more decoders per video than codec engines per card.
            self.assertEqual(
                video_decoder_xpu.xpu_decoders_per_video(),
                video_decoder_xpu.XPU_DECODERS_PER_CARD,
            )

    def test_xpu_requested_without_plugin_decodes_on_cpu(self):
        w = _wrapper("xpu", xpu_ok=False)
        self.assertNotIn("device", w._tc_kwargs)
        self.assertEqual(w._num_decode_threads, 0)

    def test_xpu_open_failure_falls_back_to_cpu(self):
        with _patched(xpu_ok=True, failing={"xpu:1"}):
            w = vd.VideoDecoderWrapper(b"not-a-real-video", device="xpu:1")
            self.assertNotIn("device", w._tc_kwargs)
            self.assertIsInstance(w.get_frames_at([0, 1]), np.ndarray)

    def test_cpu_frames_are_numpy(self):
        with _patched(xpu_ok=True):
            w = vd.VideoDecoderWrapper(b"not-a-real-video", device="cpu")
            self.assertIsInstance(w[0], np.ndarray)
            self.assertIsInstance(w.get_frames_at([0, 1]), np.ndarray)

    def test_accelerator_frames_stay_tensors(self):
        with (
            _patched(xpu_ok=True),
            envs.SGLANG_XPU_VIDEO_MAX_CONCURRENT_DECODES.override(2),
        ):
            w = vd.VideoDecoderWrapper(b"not-a-real-video", device="xpu")
            frame = w[0]
            batch = w.get_frames_at([0, 1, 2])
            # Three indices over two decoders exercises the split-and-concat path.
            tensor = w.get_frames_as_tensor([0, 1, 2])
        for value in (frame, batch, tensor):
            self.assertIsInstance(value, torch.Tensor)
            self.assertNotEqual(value.device.type, "cpu")
        self.assertEqual(tuple(batch.shape), (3, 4, 6, 3))
        self.assertEqual(tuple(tensor.shape), (3, 4, 6, 3))


class TestDenseVideoMovesToCpu(CustomTestCase):
    def setUp(self):
        video_decoder_xpu._dense_video_seen = False
        video_decoder_xpu.xpu_video_decode_device.cache_clear()
        self.addCleanup(setattr, video_decoder_xpu, "_dense_video_seen", False)
        self.addCleanup(video_decoder_xpu.xpu_video_decode_device.cache_clear)

    def test_dense_video_moves_itself_and_the_process_to_cpu(self):
        with _patched(xpu_ok=True), envs.SGLANG_XPU_VIDEO_HW_MAX_FRAMES.override(100):
            w = vd.VideoDecoderWrapper(b"not-a-real-video", device="xpu")
            self.assertIsInstance(w.get_frames_at(list(range(512))), np.ndarray)
            self.assertNotIn("device", w._tc_kwargs)
            self.assertEqual(w._num_decode_threads, 0)
            # Sticky: a later sparse video in this process decodes on the CPU too.
            later = vd.VideoDecoderWrapper(b"not-a-real-video", device="xpu")
            self.assertIsInstance(later.get_frames_at([0, 1]), np.ndarray)
        with (
            patch.object(common, "is_xpu", lambda: True),
            envs.SGLANG_ENABLE_XPU_VIDEO_HW_DECODE.override(True),
        ):
            self.assertIsNone(video_decoder_xpu.xpu_video_decode_device())

    def test_sparse_video_stays_on_xpu(self):
        with _patched(xpu_ok=True), envs.SGLANG_XPU_VIDEO_HW_MAX_FRAMES.override(100):
            w = vd.VideoDecoderWrapper(b"not-a-real-video", device="xpu")
            self.assertNotEqual(w.get_frames_at(list(range(60))).device.type, "cpu")

    def test_dense_video_off_xpu_keeps_its_device(self):
        with _patched(xpu_ok=False), envs.SGLANG_XPU_VIDEO_HW_MAX_FRAMES.override(100):
            for device in ("cuda", "cpu"):
                w = vd.VideoDecoderWrapper(b"not-a-real-video", device=device)
                before = dict(w._tc_kwargs)
                w.get_frames_at(list(range(512)))
                self.assertEqual(w._tc_kwargs, before)


class TestLoadVideoDeviceSelection(CustomTestCase):
    def setUp(self):
        video_decoder_xpu.xpu_video_decode_device.cache_clear()
        self.addCleanup(video_decoder_xpu.xpu_video_decode_device.cache_clear)

    def _resolver(self, *, is_xpu, env_on, plugin):
        with (
            patch.object(
                video_decoder_xpu.importlib.util,
                "find_spec",
                lambda name: object() if plugin else None,
            ),
            patch.object(common, "is_xpu", lambda: is_xpu),
            envs.SGLANG_ENABLE_XPU_VIDEO_HW_DECODE.override(env_on),
        ):
            return video_decoder_xpu.xpu_video_decode_device()

    def test_non_xpu_platforms_never_resolve_xpu(self):
        self.assertIsNone(self._resolver(is_xpu=False, env_on=True, plugin=True))

    def test_xpu_without_plugin_resolves_cpu(self):
        self.assertIsNone(self._resolver(is_xpu=True, env_on=True, plugin=False))

    def test_xpu_decode_is_off_unless_enabled(self):
        with envs.SGLANG_ENABLE_XPU_VIDEO_HW_DECODE.override(True):
            envs.SGLANG_ENABLE_XPU_VIDEO_HW_DECODE.clear()
            with (
                patch.object(
                    video_decoder_xpu.importlib.util, "find_spec", lambda name: object()
                ),
                patch.object(common, "is_xpu", lambda: True),
            ):
                self.assertIsNone(video_decoder_xpu.xpu_video_decode_device())

    def test_env_switch_disables_xpu_decode(self):
        self.assertIsNone(self._resolver(is_xpu=True, env_on=False, plugin=True))

    def test_load_video_keeps_cuda_off_xpu(self):
        with patch.object(common, "is_xpu", lambda: False):
            self.assertEqual(common._video_decode_device(use_gpu=True), "cuda")

    def test_load_video_on_xpu_uses_the_cpu_unless_the_media_engine_is_usable(self):
        with patch.object(common, "is_xpu", lambda: True):
            with patch.object(
                video_decoder_xpu, "xpu_video_decode_device", lambda: "xpu"
            ):
                self.assertEqual(common._video_decode_device(use_gpu=False), "cpu")
            with patch.object(
                video_decoder_xpu, "xpu_video_decode_device", lambda: None
            ):
                self.assertEqual(common._video_decode_device(use_gpu=True), "cpu")


if __name__ == "__main__":
    unittest.main()
