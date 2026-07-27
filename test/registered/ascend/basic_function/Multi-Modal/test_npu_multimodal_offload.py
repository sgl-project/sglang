"""
NPU multimodal + CPU offloading -> inference correctness.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
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
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(est_time=90, suite="full-1-npu-a3", nightly=True)


def _generate_long_prefix_5k():
    """Generate a long text prefix (~5K tokens) for chunked prefill + offloading.

    English text is approximately 4 characters per token. This produces
    roughly 5000 tokens to ensure robust chunk boundary crossing.
    """
    paragraph = (
        "This is a paragraph used for testing chunked prefill and CPU offloading "
        "functionality in the SGLang multimodal inference pipeline. It contains "
        "multiple sentences with varied vocabulary to ensure the token count reaches "
        "approximately five thousand tokens when processed by the Qwen tokenizer. "
        "The purpose is to verify that image features are not truncated or lost "
        "when they span across chunk boundaries during the prefill phase, and "
        "that CPU offloading correctly preserves and restores image embeddings "
        "when model weights are swapped between GPU and CPU memory. "
        "Chunked prefill splits long sequences into multiple chunks and computes "
        "attention on each chunk sequentially. If image features overlap with "
        "a chunk boundary, the offloaded embeddings must be correctly reloaded "
        "when the next chunk is processed. "
        "CPU offloading moves infrequently accessed model weights from GPU memory "
        "to CPU memory, reducing GPU memory usage at the cost of increased "
        "latency for transferring weights back when needed. "
    )
    # ~340 characters per paragraph * 60 = ~20400 chars ≈ 5100 tokens
    return paragraph * 60


# ============================================
# Offloading + image -> inference correctness
# ============================================
class TestMultimodalOffload(CustomTestCase):
    """Verify image inference is correct when weights are offloaded to CPU.

    [Test Category] multimodal
    [Test Target] multimodal + CPU offloading

    """

    _model = QWEN3_VL_4B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._long_prefix = _generate_long_prefix_5k()
        cls.process, cls.base_url = launch_server(
            cls._model,
            extra_args=[
                "--mem-fraction-static",
                "0.45",
                "--chunked-prefill-size",
                "512",
                "--cpu-offload-gb",
                "4",
                # NPU: CPU offloading triggers rtMemcpyAsync (CPU↔NPU weight transfer)
                # during graph capture, but Ascend driver does not allow
                # async memory copies inside a captured stream.
                # Decode CUDA graph must be disabled.
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_offload_image_inference(self):
        """Send image + long prefix request with offloaded weights, verify output."""
        text = chat(
            self.base_url,
            messages=[
                {
                    "role": "user",
                    "content": [
                        text_content(self._long_prefix),
                        image_content(self._image_b64),
                        text_content("Describe the image content in detail"),
                    ],
                },
            ],
            max_tokens=256,
        )

        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        # Expected color and shape must be mentioned
        assert_color_and_shape(
            self,
            text,
            "blue",
            "rectangle",
            prefix="test_offload_image_inference: ",
        )


if __name__ == "__main__":
    unittest.main()
