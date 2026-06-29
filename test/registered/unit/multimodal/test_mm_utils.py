"""Unit tests for srt/multimodal/mm_utils.py — no server, no model weights."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

import base64
import os
import sys
import unittest

import numpy as np
import torch
from PIL import Image

if os.name == "nt" and "resource" not in sys.modules:
    # SGLang imports `resource` from `sglang.srt.utils.common`, but the standard
    # library module does not exist on Windows. Provide a stub so unit tests can
    # run on Windows without changing runtime code paths.
    import types

    resource_stub = types.ModuleType("resource")
    resource_stub.RLIMIT_NOFILE = 7

    def _getrlimit(_):
        return (0, 0)

    def _setrlimit(_, __):
        return None

    resource_stub.getrlimit = _getrlimit
    resource_stub.setrlimit = _setrlimit
    sys.modules["resource"] = resource_stub

from sglang.srt.multimodal import mm_utils
from sglang.test.test_utils import CustomTestCase


class TestEnsureNumpy(CustomTestCase):
    def test_tensor_to_numpy(self):
        x = torch.tensor([1, 2, 3], dtype=torch.int64)
        y = mm_utils.ensure_numpy(x)
        self.assertIsInstance(y, np.ndarray)
        np.testing.assert_array_equal(y, np.array([1, 2, 3], dtype=np.int64))

    def test_numpy_passthrough(self):
        x = np.array([1, 2], dtype=np.float32)
        y = mm_utils.ensure_numpy(x)
        self.assertIs(y, x)

    def test_other_type_passthrough(self):
        x = [1, 2, 3]
        y = mm_utils.ensure_numpy(x)
        self.assertIs(y, x)


class TestHasValidData(CustomTestCase):
    def test_none_is_invalid(self):
        self.assertFalse(mm_utils.has_valid_data(None))

    def test_empty_list_is_invalid(self):
        self.assertFalse(mm_utils.has_valid_data([]))

    def test_nested_list_with_truthy_is_valid(self):
        self.assertTrue(mm_utils.has_valid_data([[None, []], [0, [False, [1]]]]))


class TestSelectBestResolution(CustomTestCase):
    def test_selects_best_fit_by_effective_resolution(self):
        original = (1000, 500)
        possible = [(256, 256), (1024, 1024), (512, 512)]
        self.assertEqual(
            mm_utils.select_best_resolution(original, possible), (1024, 1024)
        )

    def test_single_resolution_returns_it(self):
        original = (640, 480)
        self.assertEqual(
            mm_utils.select_best_resolution(original, [(224, 224)]), (224, 224)
        )

    def test_tie_breaker_min_wasted_resolution(self):
        # Construct a case where effective resolution ties, but wasted differs.
        original = (400, 400)  # area=160k
        # Both can fully cover original (effective=160k), but wasted differs.
        possible = [(450, 450), (500, 500)]
        self.assertEqual(
            mm_utils.select_best_resolution(original, possible), (450, 450)
        )


class TestGetAnyresImageGridShape(CustomTestCase):
    def test_grid_pinpoints_as_list(self):
        image_size = (800, 600)
        grid_pinpoints = [(224, 224), (448, 448)]
        w, h = mm_utils.get_anyres_image_grid_shape(
            image_size, grid_pinpoints, patch_size=224
        )
        self.assertEqual((w, h), (448 // 224, 448 // 224))

    def test_grid_pinpoints_as_range_string(self):
        # String of the form "(ax b)(...)" with 'x' triggers range expansion.
        image_size = (640, 480)
        grid_pinpoints = "(1x1)(2x2)"
        w, h = mm_utils.get_anyres_image_grid_shape(
            image_size, grid_pinpoints, patch_size=224
        )
        # Expanded possible resolutions are multiples of patch_size. Best will be 448x448.
        self.assertEqual((w, h), (2, 2))

    def test_invalid_patch_size_raises(self):
        with self.assertRaises(AssertionError):
            mm_utils.get_anyres_image_grid_shape(
                (640, 480), "(1x1)(2x2)", patch_size=128
            )

    def test_literal_eval_string_list_supported(self):
        shape = mm_utils.get_anyres_image_grid_shape(
            (640, 480),
            "[(224, 224), (448, 448)]",
            patch_size=224,
        )
        self.assertEqual(shape, (2, 2))


class TestDivideToPatches(CustomTestCase):
    def test_divides_even_grid(self):
        img = Image.new("RGB", (32, 16), color=(1, 2, 3))
        patches = mm_utils.divide_to_patches(img, patch_size=8)
        self.assertEqual(len(patches), (32 // 8) * (16 // 8))
        self.assertTrue(all(p.size == (8, 8) for p in patches))

    def test_exact_one_patch(self):
        img = Image.new("RGB", (8, 8), color=(1, 2, 3))
        patches = mm_utils.divide_to_patches(img, patch_size=8)
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].size, (8, 8))

    def test_uneven_size_includes_partial_patches(self):
        img = Image.new("RGB", (10, 10), color=(1, 2, 3))
        patches = mm_utils.divide_to_patches(img, patch_size=8)
        # ceil(10/8)=2 in each dimension → 4 crops total.
        self.assertEqual(len(patches), 4)


class TestResizeAndPadImage(CustomTestCase):
    def test_output_is_exact_target_resolution(self):
        img = Image.new("RGB", (100, 50), color=(10, 20, 30))
        out = mm_utils.resize_and_pad_image(img, (224, 224))
        self.assertEqual(out.size, (224, 224))

    def test_landscape_and_portrait_both_supported(self):
        landscape = Image.new("RGB", (200, 100), color=(10, 20, 30))
        portrait = Image.new("RGB", (100, 200), color=(10, 20, 30))
        self.assertEqual(
            mm_utils.resize_and_pad_image(landscape, (224, 224)).size, (224, 224)
        )
        self.assertEqual(
            mm_utils.resize_and_pad_image(portrait, (224, 224)).size, (224, 224)
        )

    def test_noop_when_already_target_size(self):
        img = Image.new("RGB", (224, 224), color=(10, 20, 30))
        out = mm_utils.resize_and_pad_image(img, (224, 224))
        self.assertEqual(out.size, (224, 224))

    def test_padding_background_is_black(self):
        img = Image.new("RGB", (100, 50), color=(255, 255, 255))
        out = mm_utils.resize_and_pad_image(img, (224, 224))
        # Corners are guaranteed to be padded area in this aspect ratio.
        self.assertEqual(out.getpixel((0, 0)), (0, 0, 0))


class TestExpand2Square(CustomTestCase):
    def test_returns_identity_for_square(self):
        img = Image.new("RGB", (64, 64), color=(1, 2, 3))
        out = mm_utils.expand2square(img, (0, 0, 0))
        self.assertEqual(out.size, (64, 64))

    def test_pads_width_greater_than_height(self):
        img = Image.new("RGB", (80, 40), color=(1, 2, 3))
        out = mm_utils.expand2square(img, (0, 0, 0))
        self.assertEqual(out.size, (80, 80))

    def test_converts_L_to_RGB(self):
        img = Image.new("L", (10, 5), color=7)
        out = mm_utils.expand2square(img, (0, 0, 0))
        self.assertEqual(out.mode, "RGB")


class TestUnpadImage(CustomTestCase):
    def test_unpads_width_dominant_original(self):
        # current is square-ish, original is wider → crop height.
        x = torch.zeros((3, 10, 10))
        out = mm_utils.unpad_image(x, original_size=(20, 10))  # wider than tall
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(out.shape[2], 10)
        self.assertLessEqual(out.shape[1], 10)

    def test_unpads_height_dominant_original(self):
        x = torch.zeros((3, 10, 10))
        out = mm_utils.unpad_image(x, original_size=(10, 20))  # taller than wide
        self.assertEqual(out.shape[0], 3)
        self.assertEqual(out.shape[1], 10)
        self.assertLessEqual(out.shape[2], 10)

    def test_same_aspect_ratio_returns_full_tensor(self):
        x = torch.zeros((3, 12, 6))
        out = mm_utils.unpad_image(x, original_size=(10, 20))  # aspect 0.5 matches 6/12
        self.assertEqual(out.shape, x.shape)


class TestUnpadImageShape(CustomTestCase):
    def test_shape_width_dominant_original(self):
        new_shape = mm_utils.unpad_image_shape(10, 10, original_size=(20, 10))
        self.assertIsInstance(new_shape, tuple)
        self.assertEqual(len(new_shape), 2)

    def test_shape_height_dominant_original(self):
        new_shape = mm_utils.unpad_image_shape(10, 10, original_size=(10, 20))
        self.assertIsInstance(new_shape, tuple)
        self.assertEqual(len(new_shape), 2)

    def test_shape_boundary_1x1(self):
        new_shape = mm_utils.unpad_image_shape(1, 1, original_size=(1, 1))
        self.assertEqual(new_shape, (1, 1))

    def test_shape_matches_unpad_image_result(self):
        x = torch.zeros((3, 20, 10))
        result = mm_utils.unpad_image(x, original_size=(10, 20))
        expected_shape = mm_utils.unpad_image_shape(20, 10, original_size=(10, 20))
        self.assertEqual(result.shape[1:], expected_shape)


class TestLoadImageFromBase64(CustomTestCase):
    def test_loads_valid_png(self):
        import io

        img = Image.new("RGB", (2, 3), color=(123, 45, 67))
        b = io.BytesIO()
        img.save(b, format="PNG")
        encoded = base64.b64encode(b.getvalue()).decode("ascii")

        out = mm_utils.load_image_from_base64(encoded)
        self.assertEqual(out.size, (2, 3))

    def test_invalid_base64_raises(self):
        with self.assertRaises(Exception):
            mm_utils.load_image_from_base64("not-base64!!!")

    def test_empty_string_raises(self):
        with self.assertRaises(Exception):
            mm_utils.load_image_from_base64("")


class TestGetDpEncoderLbAssignment(CustomTestCase):
    def test_balances_by_total_size_greedy(self):
        sizes = [1000, 100, 200, 50]
        shuffle, counts, loads = mm_utils.get_dp_encoder_lb_assignment(
            sizes, num_gpus=2
        )
        self.assertEqual(sorted(shuffle), [0, 1, 2, 3])
        self.assertEqual(sum(counts), len(sizes))
        self.assertEqual(len(counts), 2)
        self.assertEqual(len(loads), 2)
        # Greedy should put 1000 alone on one GPU, others on the other.
        self.assertIn(1000, loads)
        self.assertIn(350, loads)

    def test_empty_sizes(self):
        shuffle, counts, loads = mm_utils.get_dp_encoder_lb_assignment([], num_gpus=3)
        self.assertEqual(shuffle, [])
        self.assertEqual(counts, [0, 0, 0])
        self.assertEqual(loads, [0, 0, 0])

    def test_single_gpu_assigns_all(self):
        sizes = [5, 6, 7]
        shuffle, counts, loads = mm_utils.get_dp_encoder_lb_assignment(
            sizes, num_gpus=1
        )
        self.assertEqual(shuffle, [2, 1, 0])
        self.assertEqual(counts, [3])
        self.assertEqual(loads, [18])

    def test_more_gpus_than_samples(self):
        sizes = [9, 1]
        shuffle, counts, loads = mm_utils.get_dp_encoder_lb_assignment(
            sizes, num_gpus=4
        )
        self.assertEqual(sorted(shuffle), [0, 1])
        self.assertEqual(sum(counts), 2)
        self.assertEqual(len(counts), 4)
        self.assertEqual(len(loads), 4)


class _DummyProcessor:
    def __init__(self, *, crop_size, size, image_mean=(0.5, 0.5, 0.5)):
        self.crop_size = crop_size
        self.size = size
        self.image_mean = image_mean

    def preprocess(self, image):
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        arr = arr.transpose(2, 0, 1)
        return {"pixel_values": [arr]}

    def __call__(self, images):
        vals = [self.preprocess(img)["pixel_values"][0] for img in images]
        return {"pixel_values": np.stack(vals, axis=0)}


class _DummyCfg:
    def __init__(self, image_aspect_ratio, image_grid_pinpoints=None):
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints


class TestProcessAnyresImage(CustomTestCase):
    def test_uses_crop_size_when_present(self):
        img = Image.new("RGB", (100, 50), color=(5, 6, 7))
        processor = _DummyProcessor(
            crop_size={"height": 224}, size={"height": 224, "shortest_edge": 224}
        )
        out = mm_utils.process_anyres_image(img, processor, "[(224, 224), (448, 448)]")
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape[1:], (3, 224, 224))
        self.assertGreaterEqual(out.shape[0], 2)

    def test_falls_back_to_shortest_edge_when_crop_size_none(self):
        img = Image.new("RGB", (100, 50), color=(5, 6, 7))
        processor = _DummyProcessor(
            crop_size=None, size={"height": 224, "shortest_edge": 224}
        )
        out = mm_utils.process_anyres_image(img, processor, "(1x1)(2x2)")
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape[1:], (3, 224, 224))


class TestProcessImages(CustomTestCase):
    def test_pad_path_stacks_when_shapes_match(self):
        images = [
            Image.new("RGB", (32, 16), color=(1, 2, 3)),
            Image.new("RGB", (16, 32), color=(1, 2, 3)),
        ]
        processor = _DummyProcessor(crop_size={"height": 32}, size={"height": 32})
        cfg = _DummyCfg("pad")
        out = mm_utils.process_images(images, processor, cfg)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2, 3, 32, 32))

    def test_anyres_path_stacks_when_shapes_match(self):
        images = [
            Image.new("RGB", (32, 16), color=(1, 2, 3)),
            Image.new("RGB", (128, 32), color=(1, 2, 3)),
        ]
        processor = _DummyProcessor(
            crop_size={"height": 224}, size={"height": 224, "shortest_edge": 224}
        )
        cfg = _DummyCfg("anyres", "[(224, 224), (448, 448)]")
        out = mm_utils.process_images(images, processor, cfg)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape[0], 2)

    def test_anyres_path_keeps_list_for_mismatched_shapes(self):
        original = mm_utils.process_anyres_image

        def _fake_process_anyres_image(image, _processor, _pinpoints):
            # Force one image to produce a different first-dimension length.
            n = 2 if image.size[0] < 100 else 3
            return np.zeros((n, 3, 8, 8), dtype=np.float32)

        mm_utils.process_anyres_image = _fake_process_anyres_image
        try:
            images = [
                Image.new("RGB", (32, 16), color=(1, 2, 3)),
                Image.new("RGB", (128, 32), color=(1, 2, 3)),
            ]
            processor = _DummyProcessor(
                crop_size={"height": 224}, size={"height": 224, "shortest_edge": 224}
            )
            cfg = _DummyCfg("anyres", "[(224, 224), (448, 448)]")
            out = mm_utils.process_images(images, processor, cfg)
            self.assertIsInstance(out, list)
            self.assertEqual(len(out), 2)
        finally:
            mm_utils.process_anyres_image = original

    def test_default_path_uses_processor_call(self):
        images = [Image.new("RGB", (24, 24), color=(1, 2, 3))]
        processor = _DummyProcessor(crop_size={"height": 24}, size={"height": 24})
        cfg = _DummyCfg("")
        out = mm_utils.process_images(images, processor, cfg)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1, 3, 24, 24))


if __name__ == "__main__":
    unittest.main()
