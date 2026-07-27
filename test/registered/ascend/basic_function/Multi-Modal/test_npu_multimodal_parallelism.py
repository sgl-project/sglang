"""
NPU parallelism + multimodal tests (TP / CP / PP / DP).

Verify each parallel strategy works correctly with image input:
  - TP parallelism + image -> multi-card inference correct
  - Context Parallelism + multi-image -> long context correct
  - Pipeline Parallelism + image -> pipeline correct
  - Full model DP + image -> multi-replica inference correct
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_9B_WEIGHTS_PATH,
    QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.test_npu_multimodal_utils import (
    Color,
    Shape,
    assert_color_and_shape,
    chat,
    create_test_image,
    image_content,
    launch_server,
    text_content,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

# ---------------------------------------------------------------------------
# CI registration
# ---------------------------------------------------------------------------
register_npu_ci(est_time=300, suite="full-2-npu-a3", nightly=True)


# ============================================
# TP parallelism + image -> multi-card inference correct
# ============================================
class TestMultimodalTPParallelism(CustomTestCase):
    """Verify tensor parallelism + image inference is correct.

    [Test Category] multimodal
    [Test Target] multimodal + TP parallelism (2-NPU)
    """

    _model = QWEN3_5_9B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._process, cls._base_url = launch_server(
            cls._model,
            extra_args=[
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.6",
                "--dtype",
                "bfloat16",
                "--mamba-ssm-dtype",
                "bfloat16",
                # "--disable-cuda-graph",
                # "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    def test_tp_parallelism_image_inference(self):
        """Send image to TP=2 server and verify output references image content."""
        text = chat(
            self._base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            max_tokens=128,
            temperature=0,
            seed=42,
        )
        self.assertIsNotNone(text, "TP=2 server returned empty output")
        self.assertGreater(len(text), 5, f"TP=2 output too short: '{text}'")
        assert_color_and_shape(
            self,
            text,
            "blue",
            "rectangle",
            prefix="test_tp_parallelism_image_inference: ",
        )


# ============================================
# Context Parallelism + multi-image -> long context correct
# ============================================
class TestMultimodalContextParallelism(CustomTestCase):
    """Verify context parallelism works with multi-image input.

    [Test Category] multimodal
    [Test Target] multimodal + CP parallelism (2-NPU)

    Deploy Qwen3-VL-4B with ``--tp-size 2 --attn-cp-size 2``, send
    two images with a comparison prompt, and verify the output
    independently references **both** images.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image1_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.RED,
            shape=Shape.ELLIPSE,
        )
        _, cls._image2_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.BLUE,
            shape=Shape.RECTANGLE,
        )
        cls._process, cls._base_url = launch_server(
            cls._model,
            extra_args=[
                "--tp-size",
                "2",
                "--attn-cp-size",
                "2",
                "--mem-fraction-static",
                "0.3",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    def test_context_parallelism_multi_image(self):
        """Send 2 images with CP=2, verify both are independently described."""
        text = chat(
            self._base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image1_b64),
                        image_content(self._image2_b64),
                        text_content(
                            "Describe both images. "
                            "What colors and shapes do you see?"
                        ),
                    ],
                }
            ],
            max_tokens=256,
        )

        self.assertIsNotNone(text, "Response is None")
        self.assertGreater(len(text), 0, "Response is empty")

        # Both images must be described — each color mentioned
        text_lower = text.lower()
        self.assertIn(
            "red",
            text_lower,
            "First image (red ellipse) not described — " "CP may have lost its tokens",
        )
        self.assertIn(
            "blue",
            text_lower,
            "Second image (blue rectangle) not described — "
            "CP may have lost its tokens",
        )
        self.assertIn(
            "ellipse",
            text_lower,
            "Ellipse shape from first image not mentioned",
        )
        self.assertIn(
            "rectangle",
            text_lower,
            "Rectangle shape from second image not mentioned",
        )


# ============================================
# Pipeline Parallelism + image -> pipeline correct
# ============================================
class TestMultimodalPipelineParallelism(CustomTestCase):
    """Verify pipeline parallelism with image input.

    [Test Category] multimodal
    [Test Target] multimodal + PP parallelism (2-NPU)

    Deploy Qwen3-VL-4B with ``--pp-size 2``, send an image request,
    and verify the output correctly identifies the expected color and
    shape.  ViT runs on stage 0 — image embeddings cross the P2P
    boundary to stage 1.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.PURPLE,
            shape=Shape.ELLIPSE,
        )
        cls._process, cls._base_url = launch_server(
            cls._model,
            extra_args=[
                "--pp-size",
                "2",
                "--mem-fraction-static",
                "0.3",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    def test_pipeline_parallelism_image(self):
        """Send image with PP=2, verify output and feature gating."""
        text = chat(
            self._base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            max_tokens=128,
        )

        self.assertIsNotNone(text, "Response is None")
        self.assertGreater(len(text), 0, "Response is empty")

        assert_color_and_shape(
            self,
            text,
            "purple",
            "ellipse",
            prefix="test_pipeline_parallelism_image: ",
        )


# ============================================
# Full model DP + image -> multi-replica inference correct
# ============================================
class TestMultimodalFullModelDP(CustomTestCase):
    """Verify full-model data parallelism (2 replicas) with image input.

    [Test Category] multimodal
    [Test Target] multimodal + DP parallelism (2-NPU)

    Deploy Qwen3-VL-4B with ``--dp-size 2``, send an image request, and
    verify the output references visual content.
    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            color=Color.PURPLE,
            shape=Shape.RECTANGLE,
        )
        cls._process, cls._base_url = launch_server(
            cls._model,
            extra_args=[
                "--dp-size",
                "2",
                "--mem-fraction-static",
                "0.4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    def test_full_model_dp_image(self):
        """Send image request with DP=2, verify output and feature gating."""
        text = chat(
            self._base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            max_tokens=128,
        )

        self.assertIsNotNone(text, "Response is None")
        self.assertGreater(len(text), 0, "Response is empty")

        assert_color_and_shape(
            self,
            text,
            "purple",
            "rectangle",
            prefix="test_full_model_dp_image: ",
        )


if __name__ == "__main__":
    unittest.main()
