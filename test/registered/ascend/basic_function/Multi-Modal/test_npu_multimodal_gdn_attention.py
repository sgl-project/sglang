"""
NPU multimodal GDN linear attention test.

Verifies that GDN linear attention + visual encoder produce correct
multimodal output on Qwen3.5-9B.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_5_9B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_multimodal_utils import (
    Color,
    Shape,
    assert_color_and_shape,
    chat_single_image,
    create_test_image,
    launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


class TestMultimodalGDNLinearAttention(CustomTestCase):
    """Verify GDN linear attention + visual encoder produce correct multimodal output.

    [Test Category] multimodal
    [Test Target] multimodal + GDN linear attention
    """

    _model = QWEN3_5_9B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            extra_args=[
                "--disable-radix-cache",
                "--mem-fraction-static",
                "0.7",
                "--mamba-scheduler-strategy",
                "no_buffer",
                "--mamba-ssm-dtype",
                "bfloat16",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_gdn_multimodal_inference(self):
        """Send image to GDN model, verify correct multimodal inference."""
        text = chat_single_image(
            self.base_url,
            self._image_b64,
            "Describe this image",
            max_tokens=256,
        )
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        assert_color_and_shape(self, text, "blue", "rectangle")


if __name__ == "__main__":
    unittest.main()
