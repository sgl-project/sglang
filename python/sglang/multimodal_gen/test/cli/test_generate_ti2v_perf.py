import unittest

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.test_utils import TestGenerateBase

logger = init_logger(__name__)


class TestGenerateTI2VBase(TestGenerateBase):
    data_type: DataType = DataType.VIDEO

    @classmethod
    def setUpClass(cls):
        cls.base_command = [
            "sglang",
            "generate",
            "--prompt",
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
            "--image-path",
            "https://github.com/Wan-Video/Wan2.2/blob/990af50de458c19590c245151197326e208d7191/examples/i2v_input.JPG?raw=true",
            "--save-output",
            "--log-level=debug",
            f"--output-path={cls.output_path}",
        ] + cls.extra_args

    def test_single_gpu(self):
        pass

    def test_cfg_parallel(self):
        pass

    def test_mixed(self):
        pass


class TestWan2_1_I2V_14B_480P(TestGenerateTI2VBase):
    model_path = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    thresholds = {
        "test_usp": 557.9 * 1.05,
    }


class TestWan2_1_I2V_14B_720P(TestGenerateTI2VBase):
    model_path = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    thresholds = {
        "test_usp": 558.4 * 1.05,
    }


class TestWan2_2_TI2V_5B(TestGenerateTI2VBase):
    model_path = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    # FIXME: doesn't work with vsa at the moment
    # extra_args = ["--attention-backend=video_sparse_attn"]
    thresholds = {
        "test_usp": 82.3 * 1.05,
    }


# OOM
# class TestWan2_2_I2V_A14B(TestGenerateTI2VBase):
#     model_path = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
#     # FIXME: doesn't work with vsa at the moment
#     thresholds = {
#         "test_usp": 66.3 * 1.05,
#     }


if __name__ == "__main__":
    del TestGenerateTI2VBase, TestGenerateBase
    unittest.main()
