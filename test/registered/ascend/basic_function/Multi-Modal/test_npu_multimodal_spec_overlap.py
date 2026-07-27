"""
NPU multimodal + overlap schedule + speculative decoding.

Deploy Qwen3-VL-8B-Instruct with EAGLE3 + overlap schedule env vars,
send an image request, and verify output correctness.

"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_8B_EAGLE3_WEIGHTS_PATH,
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
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

register_npu_ci(est_time=90, suite="full-2-npu-a3", nightly=True)


class TestMultimodalOverlapSchedule(CustomTestCase):
    """Verify overlap schedule does not break multimodal draft generation.

    [Test Category] multimodal
    [Test Target] Overlap Schedule + speculative decoding (EAGLE3)
    """

    _model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
    _draft_model = QWEN3_VL_8B_EAGLE3_WEIGHTS_PATH

    _COMMON_ARGS = [
        "--mem-fraction-static",
        "0.6",
        "--cuda-graph-bs",
        "1",
        "--disable-radix-cache",
        "--tp-size",
        "2",
    ]

    _SPEC_ARGS = [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        _draft_model,
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-draft-model-quantization",
        "unquant",
    ]

    _OVERLAP_ENV = {
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
    }

    @classmethod
    def setUpClass(cls):
        _, cls._image_b64 = create_test_image(
            width=256, height=256, color=Color.BLUE, shape=Shape.RECTANGLE
        )
        cls._messages = [
            {
                "role": "user",
                "content": [
                    image_content(cls._image_b64),
                    text_content("Describe the image"),
                ],
            }
        ]

        cls.process, cls.base_url = launch_server(
            cls._model,
            extra_args=cls._COMMON_ARGS + cls._SPEC_ARGS,
            extra_env=cls._OVERLAP_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process is not None:
            kill_process_tree(cls.process.pid)

    def test_overlap_schedule_with_image(self):
        """Send image request with overlap schedule + EAGLE3 spec decoding."""
        output = chat(self.base_url, self._messages, max_tokens=64)

        self.assertIsNotNone(
            output, "test_overlap_schedule_with_image: Overlap schedule output is None"
        )
        self.assertGreater(
            len(output),
            0,
            "test_overlap_schedule_with_image: Overlap schedule output is empty",
        )
        assert_color_and_shape(
            self,
            output,
            "blue",
            "rectangle",
            prefix="test_overlap_schedule_with_image: ",
        )


if __name__ == "__main__":
    unittest.main()
