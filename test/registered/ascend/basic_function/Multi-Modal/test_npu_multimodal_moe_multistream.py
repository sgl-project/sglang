"""
NPU multimodal + multistream MoE tests.

Verify dual-stream MoE execution does not break image routing on NPU.
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
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(est_time=90, suite="full-2-npu-a3", nightly=True)


# The 35B-A3B MoE model needs TP=2 on 64 GB NPU cards.
_SERVER_ARGS = [
    "--tp-size",
    "2",
    "--mem-fraction-static",
    "0.7",
    # "--disable-radix-cache",
]


class TestMultimodalMultistreamMoE(CustomTestCase):
    """Verify dual-stream MoE execution does not break image routing.

    Deploy Qwen3.5-35B-A3B with ``SGLANG_NPU_USE_MULTI_STREAM=1``, send
    an image request, and verify expert routing remains correct under
    dual-stream execution.
    """

    _model = QWEN3_5_35B_A3B_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(color=Color.RED, shape=Shape.ELLIPSE)

        cls.process, cls.base_url = launch_server(
            cls._model,
            extra_args=_SERVER_ARGS,
            extra_env={"SGLANG_NPU_USE_MULTI_STREAM": "1"},
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_multimodal_multistream_moe(self):
        """Send image with dual-stream MoE enabled, verify output."""
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

        self.assertIsNotNone(text, "Response is None")
        self.assertGreater(len(text), 0, "Response is empty")

        assert_color_and_shape(
            self,
            text,
            "red",
            "ellipse",
            prefix="test_multimodal_multistream_moe: ",
        )


if __name__ == "__main__":
    unittest.main()
