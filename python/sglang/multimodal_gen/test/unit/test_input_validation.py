"""Unit tests for InputValidationStage.preprocess_condition_image resolution logic."""

import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    WanI2V480PConfig,
    WanI2V720PConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)

# Patch path for get_global_server_args used by Stage.__init__
_GLOBAL_ARGS_PATCH = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages.base.get_global_server_args"
)


def _make_batch(condition_image: Image.Image, width=None, height=None) -> Req:
    """Create a minimal Req with a condition image and optional user dimensions."""
    sp = SamplingParams(
        seed=42,
        num_outputs_per_prompt=1,
        width=width,
        height=height,
    )
    batch = Req(sampling_params=sp, condition_image=condition_image)
    return batch


def _make_server_args(pipeline_config):
    """Create a mock ServerArgs with the given pipeline config."""
    sa = MagicMock()
    sa.pipeline_config = pipeline_config
    return sa


class TestCalculateDimensionsFromArea(unittest.TestCase):
    """Tests for InputValidationStage._calculate_dimensions_from_area."""

    def test_square_aspect_ratio(self):
        # area=921600, aspect=1.0, mod=16 → sqrt(921600)=~960
        w, h = InputValidationStage._calculate_dimensions_from_area(921600, 1.0, 16)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertEqual((w, h), (960, 960))

    def test_16_9_aspect_ratio(self):
        # aspect = 720/1280 = 0.5625
        w, h = InputValidationStage._calculate_dimensions_from_area(921600, 9 / 16, 16)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertEqual((w, h), (1280, 720))

    def test_9_16_aspect_ratio(self):
        w, h = InputValidationStage._calculate_dimensions_from_area(921600, 16 / 9, 16)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertEqual((w, h), (720, 1280))

    def test_mod_alignment(self):
        # Ensure dimensions are always multiples of mod_value
        w, h = InputValidationStage._calculate_dimensions_from_area(500000, 1.3, 16)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)


class TestPreprocessConditionImageResolution(unittest.TestCase):
    """Tests for the WanI2V480PConfig branch of preprocess_condition_image.

    Verifies that:
    - Aspect ratio always comes from the condition image
    - User-specified width/height controls target area (scale)
    - Output is clamped to max_area when user dimensions exceed it
    - Dimensions are always mod-aligned
    """

    def setUp(self):
        with patch(_GLOBAL_ARGS_PATCH, return_value=MagicMock()):
            self.stage = InputValidationStage()

    def _run(self, config, img_w, img_h, user_w=None, user_h=None):
        """Run preprocess_condition_image and return (batch.width, batch.height)."""
        img = Image.new("RGB", (img_w, img_h), color="red")
        batch = _make_batch(img, width=user_w, height=user_h)
        server_args = _make_server_args(config)
        self.stage.preprocess_condition_image(batch, server_args, img_w, img_h)
        return batch.width, batch.height

    def test_720p_no_user_dims_16_9_image(self):
        """16:9 image, no user dims → 1280×720."""
        w, h = self._run(WanI2V720PConfig(), 1920, 1080)
        self.assertEqual((w, h), (1280, 720))

    def test_720p_no_user_dims_9_16_image(self):
        """9:16 image, no user dims → 720×1280."""
        w, h = self._run(WanI2V720PConfig(), 1080, 1920)
        self.assertEqual((w, h), (720, 1280))

    def test_720p_no_user_dims_square_image(self):
        """Square image, no user dims → ~960×960 (max_area=921600, sqrt≈960)."""
        w, h = self._run(WanI2V720PConfig(), 1024, 1024)
        self.assertEqual((w, h), (960, 960))
        self.assertEqual(w % 16, 0)

    def test_720p_user_dims_equal_max_area_16_9_image(self):
        """16:9 image + user 1280×720 (=max_area) → 1280×720."""
        w, h = self._run(WanI2V720PConfig(), 1920, 1080, 1280, 720)
        self.assertEqual((w, h), (1280, 720))

    def test_720p_user_dims_equal_max_area_square_image(self):
        """Square image + user 1280×720 → still square (~960×960) because
        aspect ratio comes from image, not from user dimensions."""
        w, h = self._run(WanI2V720PConfig(), 1024, 1024, 1280, 720)
        self.assertEqual((w, h), (960, 960))

    def test_720p_user_dims_smaller_area(self):
        """Square image + user 832×480 → smaller square (target_area=399360)."""
        w, h = self._run(WanI2V720PConfig(), 1024, 1024, 832, 480)
        self.assertEqual((w, h), (624, 624))
        self.assertEqual(w % 16, 0)

    def test_720p_user_dims_exceed_max_area(self):
        """4K request clamped to max_area."""
        w, h = self._run(WanI2V720PConfig(), 1920, 1080, 3840, 2160)
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertEqual((w, h), (1280, 720))

    def test_480p_no_user_dims_16_9_image(self):
        """480p config, 16:9 image → area-based calc from max_area=399360."""
        w, h = self._run(WanI2V480PConfig(), 1920, 1080)
        # max_area=480*832=399360, aspect=9/16 → (832, 464) due to rounding
        self.assertEqual(w % 16, 0)
        self.assertEqual(h % 16, 0)
        self.assertEqual((w, h), (832, 464))

    def test_condition_image_resized_to_output_dims(self):
        """Condition image is resized to match output dimensions."""
        img = Image.new("RGB", (1920, 1080), color="blue")
        batch = _make_batch(img)
        server_args = _make_server_args(WanI2V720PConfig())
        self.stage.preprocess_condition_image(batch, server_args, 1920, 1080)
        self.assertEqual(batch.condition_image.size, (batch.width, batch.height))

    def test_list_condition_image_takes_first(self):
        """List of condition images → uses first one."""
        img1 = Image.new("RGB", (1920, 1080), color="red")
        img2 = Image.new("RGB", (800, 600), color="green")
        batch = _make_batch(img1)
        batch.condition_image = [img1, img2]
        server_args = _make_server_args(WanI2V720PConfig())
        self.stage.preprocess_condition_image(batch, server_args, 1920, 1080)
        self.assertIsInstance(batch.condition_image, Image.Image)
        self.assertEqual((batch.width, batch.height), (1280, 720))


if __name__ == "__main__":
    unittest.main()
