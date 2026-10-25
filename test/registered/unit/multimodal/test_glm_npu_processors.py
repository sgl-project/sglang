"""GLM NPU preprocessing must patch every concrete image/video implementation.

The outer processor name alone must not advertise NPU preprocessing when an
inner processor lacks patchify. The fixtures model Transformers and remote
GLM5-Next image/video classes.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.modules.glm5_next_processor import (
    _npu_safe_flatten_patches,
    npu_apply_glm5_next_preprocess_patch,
)
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BASE = "sglang.srt.multimodal.processors.base_processor"


def _upstream_flatten(patches, patch_size=2, merge_size=2, temporal_patch_size=2):
    """The upstream 10-D layout is the independent CPU reference."""
    if patches.shape[1] % temporal_patch_size:
        pad = temporal_patch_size - patches.shape[1] % temporal_patch_size
        patches = torch.cat([patches, patches[:, -1:].repeat(1, pad, 1, 1, 1)], 1)
    batch, frames, channel, height, width = patches.shape
    grid_t = frames // temporal_patch_size
    grid_h, grid_w = height // patch_size, width // patch_size
    values = patches.reshape(
        batch,
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    ).permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return values.reshape(batch, grid_t * grid_h * grid_w, -1)


def _inner_processor(*, video=False, remote=False):
    def unsupported(self, *args, **kwargs):
        raise AssertionError("unpatched high-dimensional preprocessing")

    kind = "Video" if video else "Image"
    name = f"Glm5Next{kind}Processor"
    namespace = "transformers_modules.fixture" if remote else "transformers.models"
    attrs = {
        "__module__": f"{namespace}.glm5_next.processing",
        "patchify": unsupported,
    }
    return type(name, (), attrs)()


def _outer_processor(name="Glm5NextProcessor"):
    return type(name, (), {})()


class TestGlmNpuProcessors(CustomTestCase):
    def _device(self, image=None, video=None, *, outer_name="Glm5NextProcessor"):
        processor = _outer_processor(outer_name)
        processor.image_processor = image
        processor.video_processor = video
        owner = SimpleNamespace(server_args=SimpleNamespace(base_gpu_id=0))
        with patch.multiple(
            BASE,
            _is_cpu=False,
            _is_xpu=False,
            _is_npu=True,
            get_exec=lambda: SimpleNamespace(
                deterministic=SimpleNamespace(rl_on_policy_target=None)
            ),
        ):
            return BaseMultimodalProcessor._fast_image_processor_device(
                owner, processor
            )

    def _values(self, inner, *, video):
        pixels = torch.arange(3 * 3 * 8 * 8).reshape(3, 3, 8, 8).float()
        if not video:
            pixels = pixels[0]
        batched = pixels.unsqueeze(0) if video else pixels.unsqueeze(0).unsqueeze(1)
        expected = _upstream_flatten(batched)
        result = inner.patchify(pixels.unsqueeze(0), 2, 2, 2)
        self.assertTrue(torch.equal(result[0], expected))
        self.assertEqual(result[1:], (2, 4, 4) if video else (4, 4))

    def test_image_and_video_processors_run_the_npu_safe_layout(self):
        for remote in (False, True):
            with self.subTest(remote=remote):
                image = _inner_processor(remote=remote)
                video = _inner_processor(video=True, remote=remote)
                self.assertEqual(self._device(image, video), "npu")
                self._values(image, video=False)
                self._values(video, video=True)
                image_patch = type(image).patchify
                video_patch = type(video).patchify
                self.assertEqual(self._device(image, video), "npu")
                self.assertIs(type(image).patchify, image_patch)
                self.assertIs(type(video).patchify, video_patch)
                self._values(image, video=False)
                self._values(video, video=True)

    def test_remote_inner_classes_select_the_patch_without_an_outer_class_name(self):
        image = _inner_processor(remote=True)
        video = _inner_processor(video=True, remote=True)
        self.assertEqual(
            self._device(image, video, outer_name="RemoteProcessor"), "npu"
        )
        self._values(image, video=False)
        self._values(video, video=True)

    def test_unsupported_or_missing_inner_processors_do_not_advertise_npu(self):
        unsupported = SimpleNamespace(_preprocess=lambda *args: None)

        class UnrelatedProcessor:
            def _preprocess(self):
                return "unrelated preprocessing"

        for image, video in (
            (None, None),
            (unsupported, None),
            (UnrelatedProcessor(), None),
            (_inner_processor(), unsupported),
            (unsupported, _inner_processor(video=True)),
        ):
            with self.subTest(image=image, video=video):
                original_patches = [
                    (type(inner), type(inner).patchify)
                    for inner in (image, video)
                    if inner is not None
                    and callable(getattr(type(inner), "patchify", None))
                ]
                self.assertIsNone(self._device(image, video))
                for inner_type, original in original_patches:
                    self.assertIs(inner_type.patchify, original)

    def test_application_reports_success_for_supported_processors_only(self):
        for image_present, video_present in ((True, False), (False, True)):
            with self.subTest(image=image_present, video=video_present):
                supported = _outer_processor()
                supported.image_processor = (
                    _inner_processor() if image_present else None
                )
                supported.video_processor = (
                    _inner_processor(video=True) if video_present else None
                )
                self.assertTrue(npu_apply_glm5_next_preprocess_patch(supported))
                self.assertTrue(npu_apply_glm5_next_preprocess_patch(supported))
                inner = (
                    supported.image_processor
                    if image_present
                    else supported.video_processor
                )
                self._values(inner, video=video_present)
        self.assertFalse(npu_apply_glm5_next_preprocess_patch(_outer_processor()))

    def test_flatten_preserves_upstream_temporal_padding_and_merge_order(self):
        for frames in (1, 2, 3, 4):
            with self.subTest(frames=frames):
                pixels = torch.arange(2 * frames * 3 * 8 * 12).reshape(
                    2, frames, 3, 8, 12
                )
                values, grid_t, grid_h, grid_w = _npu_safe_flatten_patches(
                    pixels, 2, 2, 2
                )
                self.assertTrue(torch.equal(values, _upstream_flatten(pixels)))
                self.assertEqual((grid_t, grid_h, grid_w), ((frames + 1) // 2, 4, 6))


if __name__ == "__main__":
    unittest.main()
