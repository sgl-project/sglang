"""
NPU quantization (w8a8) multimodal tests.

  - Quantization (w8a8) + image
  - Quantization + chunked prefill + image
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_35B_A3B_W8A8_MTP_WEIGHTS_PATH,
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

register_npu_ci(est_time=250, suite="full-2-npu-a3", nightly=True)


_COMMON_EXTRA_ARGS = [
    "--tp-size",
    "2",
    "--mem-fraction-static",
    "0.7",
]

_LONG_PREFIX = (
    "This paragraph is used to generate a long text prefix that triggers "
    "chunked prefill in the SGLang multimodal inference pipeline. When a "
    "large language model processes a sequence longer than the configured "
    "chunk size, the prefill phase splits the sequence into multiple chunks "
    "and computes attention on each chunk sequentially. This text is designed "
    "to produce approximately five thousand tokens with the Qwen tokenizer, "
    "ensuring that any image tokens appended after it will land on a later "
    "chunk rather than the first one. The vocabulary is intentionally varied "
    "to avoid degenerate repetition patterns that could affect model output. "
    "Chunk boundaries are a known source of subtle bugs in multimodal "
    "inference because visual features computed by the ViT encoder must be "
    "preserved and correctly reloaded when the next chunk is processed. "
)


# ============================================
# Quantization (w8a8) + image -> accuracy acceptable
# ============================================
class TestMultimodalQuantization(CustomTestCase):
    """Verify w8a8 quantization does not degrade image understanding.

    Deploy Qwen3.5-35B-A3B-w8a8-mtp (weight-only INT8 quantization),
    send an image request, and verify the output references visual content.

    [Test Category] multimodal
    [Test Target] multimodal + quantization + w8a8
    """

    _model = QWEN3_5_35B_A3B_W8A8_MTP_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.GREEN,
            shape=Shape.ELLIPSE,
        )
        cls._process, cls._url = launch_server(
            cls._model,
            extra_args=_COMMON_EXTRA_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    def test_multimodal_quantization(self):
        """Send image to w8a8 quantized model, verify output."""
        text = chat(
            self._url,
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

        self.assertIsNotNone(text, "test_multimodal_quantization: Response is None")
        self.assertGreater(
            len(text), 0, "test_multimodal_quantization: Response is empty"
        )

        assert_color_and_shape(
            self,
            text,
            "green",
            "ellipse",
            prefix="test_multimodal_quantization: ",
        )

        print(f"  [test_multimodal_quantization] output_len={len(text)}")


# ============================================
# Quantization + chunked prefill + image -> accuracy persists
# ============================================
class TestMultimodalQuantizationChunked(CustomTestCase):
    """Verify quantized accuracy survives chunked prefill boundaries.

    Deploy Qwen3.5-35B-A3B-w8a8-mtp with chunked prefill, send a long
    text prefix + image request, and verify the image content is
    recognized despite quantization + chunk boundary crossing.

    [Test Category] multimodal
    [Test Target] multimodal + quantization + w8a8 + chunked prefill
    """

    _model = QWEN3_5_35B_A3B_W8A8_MTP_WEIGHTS_PATH
    _long_text = _LONG_PREFIX * 25  # ~5.7K tokens

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            color=Color.TEAL,
            shape=Shape.ELLIPSE,
        )
        cls._process, cls._url = launch_server(
            cls._model,
            extra_args=_COMMON_EXTRA_ARGS
            + [
                "--chunked-prefill-size",
                "512",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls._process.pid)

    def test_multimodal_quantization_chunked_image(self):
        """Send long prefix + image with quantized chunked prefill, verify output."""
        text = chat(
            self._url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        text_content(self._long_text),
                        image_content(self._image_b64),
                        text_content("Describe the image"),
                    ],
                }
            ],
            max_tokens=128,
        )

        self.assertIsNotNone(
            text, "test_multimodal_quantization_chunked_image: Response is None"
        )
        self.assertGreater(
            len(text),
            0,
            "test_multimodal_quantization_chunked_image: Response is empty",
        )

        assert_color_and_shape(
            self,
            text,
            "teal",
            "ellipse",
            prefix="test_multimodal_quantization_chunked_image: ",
        )


if __name__ == "__main__":
    unittest.main()
