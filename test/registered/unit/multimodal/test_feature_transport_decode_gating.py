"""CPU feature transport must keep the tokenizer worker off the base GPU.

With mm_feature_transport="cpu" (the default when unset), _load_single_item
must force CPU video decode, PIL image decode and unpinned frames: nvJPEG and
pin_memory() each initialize a sticky CUDA context in the worker process.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    feature_transport_uses_gpu,
)
from sglang.srt.multimodal.processors.ernie45_vl import (
    preprocess_video as ernie_preprocess_video,
)
from sglang.srt.multimodal.processors.qwen_vl import (
    preprocess_video as qwen_preprocess_video,
)
from sglang.srt.runtime_context import get_context
from sglang.srt.utils import load_video
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.test_utils import CustomTestCase


class _StubProcessor(BaseMultimodalProcessor):
    gpu_image_decode = True

    async def process_mm_data_async(self, *args, **kwargs):
        raise NotImplementedError


class _StubQwenVideo(VideoDecoderWrapper):
    """Pre-built frames stand in for a decoded video; the pin gate is the subject."""

    def __init__(self, frames, fps):
        self._frames = frames
        self._fps = fps

    def __len__(self):
        return self._frames.shape[0]

    @property
    def avg_fps(self):
        return self._fps

    def get_frames_as_tensor(self, indices):
        return self._frames


class _StubErnieVideo:
    def __init__(self, frames, fps):
        self._frames = frames
        self._fps = fps

    def __len__(self):
        return self._frames.shape[0]

    def get_avg_fps(self):
        return self._fps

    def get_batch(self, indices):
        return SimpleNamespace(asnumpy=lambda: self._frames[indices])


class TestFeatureTransportDecodeGating(CustomTestCase):
    def _with_transport(self, transport):
        return get_context().override_server_args(mm_feature_transport=transport)

    def test_feature_transport_uses_gpu(self):
        cases = (
            (None, False),
            ("cpu", False),
            ("cuda_ipc", True),
            ("cuda_vmm", True),
        )
        for transport, expected in cases:
            with self.subTest(transport=transport):
                with self._with_transport(transport):
                    self.assertIs(feature_transport_uses_gpu(), expected)

    def test_video_decode_device_follows_transport(self):
        # (transport, use_gpu, pin)
        cases = (
            (None, False, False),
            ("cpu", False, False),
            ("cuda_ipc", True, True),
            ("cuda_vmm", True, True),
        )
        for transport, use_gpu, pin in cases:
            with self.subTest(transport=transport):
                with self._with_transport(transport), patch(
                    "sglang.srt.multimodal.processors.base_processor.load_video"
                ) as load_video:
                    _StubProcessor._load_single_item("video.mp4", Modality.VIDEO)
                load_video.assert_called_once_with(
                    "video.mp4", use_gpu=use_gpu, pin=pin
                )

    def test_image_decode_follows_transport(self):
        for transport, gpu_decode in (
            (None, False),
            ("cpu", False),
            ("cuda_ipc", True),
            ("cuda_vmm", True),
        ):
            with self.subTest(transport=transport):
                with self._with_transport(transport), patch(
                    "sglang.srt.multimodal.processors.base_processor.load_image"
                ) as load_image:
                    load_image.return_value = (Image.new("RGB", (4, 4)), None)
                    _StubProcessor._load_single_item(b"jpeg", Modality.IMAGE)
                load_image.assert_called_once_with(b"jpeg", gpu_decode)

    def test_load_video_pin_flag_reaches_wrapper(self):
        with patch("sglang.srt.utils.common.VideoDecoderWrapper") as wrapper:
            load_video(b"fake-video", use_gpu=False, pin=False)
            _, kwargs = wrapper.call_args
            self.assertEqual(kwargs, {"device": "cpu", "pin": False})

    def test_load_video_defaults_unchanged(self):
        with patch("sglang.srt.utils.common.VideoDecoderWrapper") as wrapper:
            load_video(b"fake-video")
            _, kwargs = wrapper.call_args
            self.assertEqual(kwargs, {"device": "cuda", "pin": True})

    def test_maybe_pin_disabled_returns_tensor_untouched(self):
        wrapper = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
        wrapper._pin = False
        tensor = torch.zeros(2, 2)
        self.assertIs(wrapper._maybe_pin(tensor), tensor)

    def test_maybe_pin_pinning_behavior(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        wrapper = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
        wrapper._pin = True
        # CUDA tensors are never re-pinned.
        cuda_tensor = torch.zeros(2, 2, device="cuda")
        self.assertIs(wrapper._maybe_pin(cuda_tensor), cuda_tensor)
        # CPU tensors still pin by default.
        self.assertTrue(wrapper._maybe_pin(torch.zeros(2, 2)).is_pinned())

    def _stub_frames(self):
        # float32 keeps torchvision's resize cheap; the pin gate is dtype-agnostic.
        return torch.zeros(4, 16, 16, 3, dtype=torch.float32)

    def test_qwen_preprocess_video_pin_follows_transport(self):
        for transport, expect_pinned in (("cpu", False), ("cuda_ipc", True)):
            with self.subTest(transport=transport):
                if expect_pinned and not torch.cuda.is_available():
                    self.skipTest("CUDA not available")
                with self._with_transport(transport):
                    video, _ = asyncio.run(
                        qwen_preprocess_video(
                            _StubQwenVideo(self._stub_frames(), fps=2.0)
                        )
                    )
                self.assertIs(video.is_pinned(), expect_pinned)

    def test_ernie_preprocess_video_pin_follows_transport(self):
        for transport, expect_pinned in (("cpu", False), ("cuda_ipc", True)):
            with self.subTest(transport=transport):
                if expect_pinned and not torch.cuda.is_available():
                    self.skipTest("CUDA not available")
                with self._with_transport(transport):
                    video, _ = asyncio.run(
                        ernie_preprocess_video(
                            _StubErnieVideo(
                                np.zeros((4, 16, 16, 3), dtype=np.float32), fps=2.0
                            )
                        )
                    )
                self.assertIs(video.is_pinned(), expect_pinned)


if __name__ == "__main__":
    unittest.main()
