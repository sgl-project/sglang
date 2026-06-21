import io
import unittest
from array import array

import torch
from PIL import Image

from sglang.srt.utils.common import _load_image, flatten_arrays_to_int64_tensor
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestFlattenArraysToInt64Tensor(CustomTestCase):
    """`flatten_arrays_to_int64_tensor` is invoked by `prepare_for_extend`
    to build the per-batch input_ids tensor (pinned, async H2D) from a
    list of array.array('q') per-req get_fill_ids() slices. Tests the
    full matrix of (device, pin) the production code paths through.
    """

    DEVICES = ("cpu", "cuda")
    PIN_OPTIONS = (False, True)

    def _check(self, parts: list, expected: list[int]) -> None:
        for device in self.DEVICES:
            for pin in self.PIN_OPTIONS:
                with self.subTest(device=device, pin=pin):
                    out = flatten_arrays_to_int64_tensor(parts, device, pin)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    self.assertEqual(out.dtype, torch.int64)
                    self.assertEqual(out.device.type, device)
                    self.assertEqual(out.shape, (len(expected),))
                    self.assertEqual(out.cpu().tolist(), expected)

    def test_single_part(self):
        parts = [array("q", [1, 2, 3, 4, 5])]
        self._check(parts, [1, 2, 3, 4, 5])

    def test_multiple_parts(self):
        parts = [
            array("q", [10, 20, 30]),
            array("q", [100, 200]),
            array("q", [1000]),
        ]
        self._check(parts, [10, 20, 30, 100, 200, 1000])


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA (nvJPEG GPU decode)")
class TestLoadImageGpuJpegRGB(CustomTestCase):
    """`_load_image` GPU fast-path must decode JPEGs to 3 channels regardless of
    the source channel count. With the torchvision default
    (ImageReadMode.UNCHANGED) a grayscale JPEG decodes to a [1, H, W] tensor,
    which later breaks torch.stack against 3-channel images during multimodal
    batching. Forcing ImageReadMode.RGB keeps every JPEG at [3, H, W], matching
    the PIL fallback's convert("RGB").
    """

    @staticmethod
    def _jpeg_bytes(mode: str, size=(64, 48)) -> bytes:
        color = 128 if mode == "L" else (130, 90, 60)
        buf = io.BytesIO()
        Image.new(mode, size, color=color).save(buf, format="JPEG")
        return buf.getvalue()

    def _decode_on_gpu(self, mode: str) -> torch.Tensor:
        img = _load_image(image_bytes=self._jpeg_bytes(mode), gpu_image_decode=True)
        # The GPU fast-path returns a device tensor; if nvJPEG is unavailable it
        # falls back to PIL, which this test cannot exercise.
        if not isinstance(img, torch.Tensor):
            self.skipTest("nvJPEG GPU decode unavailable; fell back to PIL")
        return img

    def test_grayscale_jpeg_decodes_to_3_channels(self):
        self.assertEqual(self._decode_on_gpu("L").shape[0], 3)

    def test_rgb_jpeg_stays_3_channels(self):
        self.assertEqual(self._decode_on_gpu("RGB").shape[0], 3)

    def test_grayscale_and_rgb_are_stackable(self):
        # Regression: [1, H, W] vs [3, H, W] previously made torch.stack raise.
        stacked = torch.stack([self._decode_on_gpu("L"), self._decode_on_gpu("RGB")])
        self.assertEqual(tuple(stacked.shape[:2]), (2, 3))


if __name__ == "__main__":
    unittest.main()
