"""
NPU multimodal basic test cases.

  TC-001: Single image + text -> describe image content
  TC-002: Same image twice -> cache hit (Radix Cache prefix caching)
  TC-003: Multi-image -> compare two images
  TC-004: Variable size images -> different resolutions
  TC-005: Long text + image triggering chunked prefill
  TC-006: Graph compilation enabled multimodal inference
"""

import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.test_npu_multimodal_utils import (
    Color,
    Shape,
    assert_color_and_shape,
    assert_text_contains,
    chat,
    chat_single_image,
    create_test_image,
    image_content,
    launch_server,
    text_content,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
class TestMultimodalBasicFunction(CustomTestCase):
    """Multimodal basic test cases for NPU.

    [Test Category] multimodal
    [Test Target] multimodal basic function
    """

    @classmethod
    def setUpClass(cls):
        """Start the SGLang server for VLM inference."""
        cls.model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH
        cls.api_key = "sk-123456"

        cls.process, cls.base_url = launch_server(
            cls.model,
            extra_args=[
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.857",
                "--tp-size",
                "1",
                "--enable-cache-report",
            ],
        )
        cls.client = openai.Client(
            api_key=cls.api_key,
            base_url=f"{cls.base_url}/v1",
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up the server process."""
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    # -- helpers -----------------------------------------------------------

    def _build_msg(self, image_b64, prompt):
        return [
            {
                "role": "user",
                "content": [image_content(image_b64), text_content(prompt)],
            }
        ]

    # -- test cases --------------------------------------------------------

    def test_001_single_image(self):
        """Single image + text -> describe image content.

        Sends one test image with a description prompt and checks:
        - Non-empty, semantically relevant output
        """
        _, image_b64 = create_test_image(640, 480, color=Color.RED, shape=Shape.ELLIPSE)

        output = chat(
            self.base_url,
            self._build_msg(image_b64, "Please describe this image"),
            max_tokens=128,
        )

        self.assertTrue(
            output,
            "TC-001: Model returned empty response for single-image description",
        )
        self.assertGreater(
            len(output),
            5,
            f"TC-001: Output suspiciously short: '{output}'",
        )

        assert_color_and_shape(self, output, "red", "ellipse", prefix="TC-001: ")

        print(f"  [TC-001] output_len={len(output)}")

    def test_002_same_image_cache_hit(self):
        """Same image twice -> cache hit.

        Sends two requests with an identical image but different text prompts.
        The second request should benefit from Radix Cache prefix matching
        on the image-token prefix, reflected in cached_tokens > 0.
        """
        _, image_b64 = create_test_image(
            640, 480, color=Color.BLUE, shape=Shape.ELLIPSE
        )

        # --- Request 1 (cache miss, populates image prefix) ---
        output1 = chat(
            self.base_url,
            self._build_msg(image_b64, "Describe the image"),
            max_tokens=128,
        )
        self.assertTrue(output1, "TC-002: First request returned empty response")
        assert_color_and_shape(self, output1, "blue", "ellipse", prefix="TC-002/req1: ")

        # --- Request 2 (same image, different text -> prefix cache hit) ---
        response2 = self.client.chat.completions.create(
            model="default",
            messages=self._build_msg(
                image_b64,
                "Describe the shape and color of the object in the image",
            ),
            temperature=0,
            max_tokens=128,
        )
        output2 = response2.choices[0].message.content
        self.assertTrue(output2, "TC-002: Second request returned empty response")
        assert_color_and_shape(self, output2, "blue", "ellipse", prefix="TC-002/req2: ")

        # --- Verify cache hit ---
        cached_tokens = getattr(
            getattr(response2.usage, "prompt_tokens_details", None),
            "cached_tokens",
            0,
        )
        self.assertGreater(
            cached_tokens,
            0,
            f"TC-002: Expected cache hit (cached_tokens > 0) but got "
            f"cached_tokens={cached_tokens}. The image prefix from request 1 "
            f"should be cached.",
        )

        print(
            f"  [TC-002] output1_len={len(output1)}  output2_len={len(output2)}  "
            f"cached_tokens={cached_tokens}/{response2.usage.prompt_tokens}"
        )

    def test_003_multi_image_compare(self):
        """Multi-image -> compare two images.

        Sends two different images with a comparison prompt. Verifies that
        the model produces a coherent output referencing both images, and
        that no OOM / crash occurs.
        """
        # Two visually distinct images
        _, img1_b64 = create_test_image(320, 240, color=Color.RED, shape=Shape.ELLIPSE)
        _, img2_b64 = create_test_image(320, 240, color=Color.BLUE, shape=Shape.ELLIPSE)

        # Multi-image request: two image_url blocks + comparison prompt
        response = self.client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        image_content(img1_b64),
                        image_content(img2_b64),
                        text_content("Please compare these two images"),
                    ],
                }
            ],
            temperature=0,
            max_tokens=256,
        )

        output = response.choices[0].message.content

        self.assertTrue(
            output,
            "TC-003: Model returned empty response for multi-image comparison",
        )
        self.assertGreaterEqual(
            len(output),
            10,
            f"TC-003: Output too short for a meaningful comparison: '{output}'",
        )

        # The response should reference both images
        assert_color_and_shape(self, output, "red", "ellipse", prefix="TC-003: ")
        assert_color_and_shape(self, output, "blue", "ellipse", prefix="TC-003: ")

        # Basic usage fields should be present
        self.assertGreater(
            response.usage.prompt_tokens,
            0,
            "TC-003: prompt_tokens should be > 0 for multi-image request",
        )
        self.assertGreater(
            response.usage.completion_tokens,
            0,
            "TC-003: completion_tokens should be > 0",
        )

    def test_004_variable_size_images(self):
        """Variable size images -> different resolutions.

        Verifies two distinct image processing paths from SGLang's own
        smart_resize() in srt/multimodal/processors/qwen_vl.py:

          MIN_PIXELS = 4*28*28 = 3136        → below triggers upscale
          MAX_PIXELS = 16384*28*28 ≈ 12.8M   → above triggers downscale
          (MAX_PIXELS is configurable via SGLANG_IMAGE_MAX_PIXELS env var)

        Test images:
          32x32 (1024 px)   → below MIN_PIXELS → upscale path
          1920x1080 (2.0M px) → within normal range (below 12.8M MAX_PIXELS)
                                  → does NOT trigger downscale, but stresses
                                  ViT memory allocation (large patch count)

        """
        sizes = [
            (32, 32, Color.PURPLE, "below_min"),  # 1024 px → upscale
            (1920, 1080, Color.TEAL, "large"),  # 2M px → memory stress, not downscale
        ]

        for width, height, color, label in sizes:
            with self.subTest(size=f"{width}x{height}"):
                _, img_b64 = create_test_image(
                    width, height, color=color, shape=Shape.ELLIPSE
                )

                output = chat(
                    self.base_url,
                    self._build_msg(
                        img_b64,
                        "Describe the shape and color of the object in the image",
                    ),
                    max_tokens=64,
                )

                # All sizes: must produce non-empty output
                self.assertTrue(
                    output,
                    f"TC-004: Empty output for {label} image ({width}x{height})",
                )

                # All sizes: must reference image content (not gibberish)
                assert_color_and_shape(
                    self,
                    output,
                    color.name.lower(),
                    "ellipse",
                    prefix=f"TC-004/{label}: ",
                )


_LONG_PREFIX = (
    "To verify that chunked prefill does not truncate image features in the "
    "multimodal inference pipeline, a sufficient amount of text must precede "
    "the actual image content so that the vision tokens inevitably cross a "
    "chunk boundary. When the prefill stage encounters an input longer than "
    "the configured threshold, it partitions the sequence into fixed-size "
    "segments and processes them one at a time through the attention layers. "
    "Any modality-specific embeddings that sit on a later segment must be "
    "correctly materialized and merged by the scheduler. This paragraph is "
    "deliberately crafted with varied vocabulary and diverse sentence "
    "structures to exercise the tokenizer thoroughly and avoid degenerate "
    "repetition patterns that could trigger unusual model behavior or "
    "attention collapse in the decoder. Ensuring correctness under chunked "
    "scheduling is essential because production deployments routinely handle "
    "long-context requests containing embedded images, diagrams, scanned "
    "documents, and other visual artifacts alongside substantial textual "
    "narrative. The interaction between the ViT encoder output and the LLM "
    "decoder's autoregressive generation must remain consistent regardless "
    "of how the prefill phase splits the input, as any inconsistency could "
    "manifest as hallucinations, missing visual details, or garbled responses "
    "that degrade the user experience in subtle but consequential ways. This "
    "text alone does not reference any specific color or shape; it serves "
    "solely as a neutral preceding context whose only purpose is to push the "
    "subsequent image data past the first and second chunk boundaries, so "
    "that the test can validate whether cross-chunk visual feature forwarding "
    "works correctly on the target hardware platform without interference "
    "from the semantic content of the prefix itself."
)


# ============================================
# Long text + image -> chunked prefill
# ============================================


class TestMultimodalChunkedPrefill(CustomTestCase):
    """Verify that chunked prefill does not truncate image features.

    [Test Category] multimodal
    [Test Target] multimodal + chunked prefill

    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._long_prefix = _LONG_PREFIX * 10  # ~3K tokens with Qwen tokenizer
        cls.process, cls.base_url = launch_server(
            cls._model,
            extra_args=[
                "--chunked-prefill-size",
                "512",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.25",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            try:
                kill_process_tree(cls.process.pid)
            except Exception:
                pass

    def test_long_prefix_with_image(self):
        """Send long prefix + image + prompt, verify both are understood."""
        text = chat(
            self.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        text_content(self._long_prefix),
                        image_content(self._image_b64),
                        text_content(
                            "Describe the image and briefly summarize the text above it."
                        ),
                    ],
                },
            ],
            max_tokens=128,
        )
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        # Image features must NOT be ignored
        assert_color_and_shape(self, text, "blue", "rectangle")

        # The long prefix content should also be referenced
        assert_text_contains(
            self,
            text,
            hints=["chunked", "prefill", "boundary", "scheduler", "embedding"],
        )


# ============================================
# Graph compilation + multimodal
# ============================================


class TestMultimodalGraphCompilation(CustomTestCase):
    """Verify multimodal inference works with graph compilation enabled.

    Launches a server without --disable-cuda-graph so the NPU ViT graph
    runner (vit_npu_graph_runner) handles the ViT forward pass.  Validates
    that graph capture + replay does not crash, corrupt features, or produce
    gibberish.


    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.RED, shape=Shape.RECTANGLE
        )
        cls._prompt = "Describe the image"

    def test_graph_multimodal_inference(self):
        """Verify multimodal inference with graph compilation enabled."""
        process_graph, url_graph = launch_server(
            self._model,
            extra_args=["--cuda-graph-max-bs", "1"],
        )
        try:
            result = chat_single_image(
                url_graph,
                self._image_b64,
                self._prompt,
                max_tokens=256,
            )
            self.assertIsNotNone(result)
            self.assertGreater(len(result), 0)
            assert_color_and_shape(self, result, "red", "rectangle")
        finally:
            kill_process_tree(process_graph.pid)


if __name__ == "__main__":
    unittest.main()
