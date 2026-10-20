"""GLM NPU preprocessing must patch every concrete image/video implementation.

Regression: an outer Glm5NextProcessor selected the patchify path even when its
inner GLMGA processors exposed only _preprocess, then advertised an unpatched
processor as NPU-safe. The fixtures model both Transformers and remote classes.
"""

import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch
from transformers.image_utils import SizeDict

from sglang.srt.hardware_backend.npu.modules.glm5_next_processor import (
    _npu_safe_flatten_patches,
    npu_apply_glm5_next_preprocess_patch,
)
from sglang.srt.hardware_backend.npu.modules.glmga_processor import (
    _resize_glmga_inputs,
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


def _inner_processor(*, video=False, legacy=False, remote=False):
    def unsupported(self, *args, **kwargs):
        raise AssertionError("unpatched high-dimensional preprocessing")

    family = "glmga" if legacy else "glm5_next"
    kind = "Video" if video else "Image"
    name = f"{'Glmga' if legacy else 'Glm5Next'}{kind}Processor"
    namespace = "transformers_modules.fixture" if remote else "transformers.models"
    attrs = {
        "__module__": f"{namespace}.{family}.processing",
        "_preprocess" if legacy else "patchify": unsupported,
        "rescale_and_normalize": lambda self, inputs, *args: inputs,
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

    def _values(self, inner, *, video, legacy):
        pixels = torch.arange(3 * 3 * 8 * 8).reshape(3, 3, 8, 8).float()
        if not video:
            pixels = pixels[0]
        batched = pixels.unsqueeze(0) if video else pixels.unsqueeze(0).unsqueeze(1)
        expected = _upstream_flatten(batched)
        if legacy:
            common = dict(
                do_resize=False,
                size=SizeDict(shortest_edge=16, longest_edge=64),
                resample=None,
                do_rescale=False,
                rescale_factor=1,
                do_normalize=False,
                image_mean=None,
                image_std=None,
                patch_size=2,
                temporal_patch_size=2,
                merge_size=2,
                patch_expand_factor=1,
                return_tensors="pt",
            )
            if video:
                result = inner._preprocess([pixels], do_convert_rgb=False, **common)
                self.assertEqual(set(result), {"pixel_values_videos", "video_grid_thw"})
                values, grid = result["pixel_values_videos"], result["video_grid_thw"]
            else:
                result = inner._preprocess([pixels], disable_grouping=False, **common)
                self.assertEqual(set(result), {"pixel_values", "image_grid_thw"})
                values, grid = result["pixel_values"], result["image_grid_thw"]
            self.assertTrue(torch.equal(values, expected[0]))
            self.assertEqual(grid.tolist(), [[2 if video else 1, 4, 4]])
        else:
            result = inner.patchify(pixels.unsqueeze(0), 2, 2, 2)
            self.assertTrue(torch.equal(result[0], expected))
            self.assertEqual(result[1:], (2, 4, 4) if video else (4, 4))

    def test_legacy_and_mixed_processors_run_the_npu_safe_layout(self):
        for remote in (False, True):
            for image_legacy, video_legacy in (
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            ):
                with self.subTest(
                    remote=remote, image_legacy=image_legacy, video_legacy=video_legacy
                ):
                    image = _inner_processor(legacy=image_legacy, remote=remote)
                    video = _inner_processor(
                        video=True, legacy=video_legacy, remote=remote
                    )
                    self.assertEqual(self._device(image, video), "npu")
                    self._values(image, video=False, legacy=image_legacy)
                    self._values(video, video=True, legacy=video_legacy)
                    # Repeated device selection must preserve the patched behavior.
                    self.assertEqual(self._device(image, video), "npu")
                    self._values(image, video=False, legacy=image_legacy)

    def test_remote_inner_classes_select_the_patch_without_an_outer_class_name(self):
        image = _inner_processor(legacy=True, remote=True)
        video = _inner_processor(video=True, remote=True)
        self.assertEqual(
            self._device(image, video, outer_name="RemoteProcessor"), "npu"
        )
        self._values(image, video=False, legacy=True)
        self._values(video, video=True, legacy=False)

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
                self.assertIsNone(self._device(image, video))

    def test_application_reports_success_for_supported_processors_only(self):
        supported = _outer_processor()
        supported.image_processor = _inner_processor()
        self.assertTrue(npu_apply_glm5_next_preprocess_patch(supported))
        self.assertTrue(npu_apply_glm5_next_preprocess_patch(supported))
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

    def test_old_and_new_resize_apis_preserve_video_batch_and_size(self):
        pixels = torch.arange(2 * 3 * 3 * 8 * 8).reshape(2, 3, 3, 8, 8)
        size = SizeDict(shortest_edge=16, longest_edge=64)

        class OldResize:
            def resize(self, inputs, size, resample):
                # The old backend accepts a batch of images, not a video batch.
                if inputs.ndim != 4:
                    raise ValueError("old resize expects image batches")
                return inputs[..., : size.height, : size.width]

        class NewResize:
            def resize(self, inputs, size, resample, factor, temporal_factor):
                return inputs[..., :factor, :temporal_factor]

        # A remote module can provide GLMGA without a built-in model namespace.
        module = ModuleType("transformers_modules.fixture.glmga")
        module.smart_resize = lambda **kwargs: (4, 2)
        OldResize.__module__ = module.__name__
        with patch.dict("sys.modules", {module.__name__: module}):
            old = _resize_glmga_inputs(
                OldResize(),
                pixels,
                size=size,
                resample=None,
                factor=4,
                temporal_factor=2,
                num_frames=3,
            )
        new = _resize_glmga_inputs(
            NewResize(),
            pixels,
            size=size,
            resample=None,
            factor=4,
            temporal_factor=2,
            num_frames=3,
        )
        self.assertTrue(torch.equal(old, pixels[..., :4, :2]))
        self.assertTrue(torch.equal(new, old))


if __name__ == "__main__":
    unittest.main()
