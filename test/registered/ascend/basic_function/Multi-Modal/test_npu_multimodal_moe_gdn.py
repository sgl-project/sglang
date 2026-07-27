"""
NPU multimodal GDN + MoE test.

Verifies that GDN linear attention + MoE + visual encoder work together
correctly on Qwen3.5-35B-A3B.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_5_35B_A3B_WEIGHTS_PATH
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

register_npu_ci(est_time=120, suite="full-2-npu-a3", nightly=True)


class TestMultimodalGDNMoE(CustomTestCase):
    """Verify GDN + MoE + visual encoder work together correctly.

    Uses Qwen3.5-35B-A3B (GDN + MoE + DeepStack ViT) to verify the full
    GDN+MoE+Vision interaction.

    [Test Category] multimodal
    [Test Target] multimodal + GDN + MoE + visual encoder
    """

    _model = QWEN3_5_35B_A3B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.GREEN, shape=Shape.RECTANGLE
        )
        # 35B MoE model: ~70GB weights, needs TP>=2 on 64GB NPU
        cls.process, cls.base_url = launch_server(
            cls._model,
            extra_args=[
                # "--disable-radix-cache",
                "--tp-size",
                "2",
                "--mem-fraction-static",
                "0.7",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_gdn_moe_multimodal_inference(self):
        """Send image to GDN+MoE model, verify correct output."""
        text = chat_single_image(
            self.base_url,
            self._image_b64,
            "Describe this image",
            max_tokens=256,
        )
        self.assertIsNotNone(text)
        self.assertGreater(len(text), 0)

        assert_color_and_shape(self, text, "green", "rectangle")


if __name__ == "__main__":
    unittest.main()
