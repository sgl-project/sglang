"""
NPU multimodal + EPLB tests.

Verify EPLB (Expert Parallel Load Balancing) does not disrupt image inference.
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_5_35B_A3B_WEIGHTS_PATH,
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

register_npu_ci(est_time=150, suite="full-2-npu-a3", nightly=True)


# ============================================
# EPLB + image -> elastic load balancing correct
# ============================================


class TestMultimodalEPLB(CustomTestCase):
    """Verify EPLB does not disrupt image inference.

    Deploy Qwen3.5-35B-A3B with ``--enable-eplb``, send an image
    request, and verify the output correctly identifies the expected
    color and shape.

    [Test Category] multimodal
    [Test Target] multimodal + EPLB
    """

    _model = QWEN3_5_35B_A3B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256,
            height=256,
            color=Color.PURPLE,
            shape=Shape.RECTANGLE,
        )
        cls.process, cls.base_url = launch_server(
            cls._model,
            extra_args=[
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "normal",
                # "--disable-cuda-graph",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                "4",
                "--eplb-rebalance-num-iterations",
                "50",
                "--expert-distribution-recorder-buffer-size",
                "50",
                "--expert-distribution-recorder-mode",
                "stat",
                "--ep-dispatch-algorithm",
                "static",
            ],
            env={
                "HCCL_BUFFSIZE": "1024",
                # EPLB rebalance triggers expert weight redistribution,
                # which conflicts with ACL format weight conversion.
                # This env var must be set when using EPLB on NPU.
                "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multimodal_eplb(self):
        """Send image with EPLB enabled, verify output."""
        text = chat(
            self.base_url,
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

        self.assertIsNotNone(text, "test_multimodal_eplb: Response is None")
        self.assertGreater(len(text), 0, "test_multimodal_eplb: Response is empty")

        assert_color_and_shape(
            self,
            text,
            "purple",
            "rectangle",
            prefix="test_multimodal_eplb: ",
        )


if __name__ == "__main__":
    unittest.main()
