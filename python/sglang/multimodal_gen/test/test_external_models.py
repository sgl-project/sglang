import os
import unittest

from sglang.multimodal_gen import DiffGenerator
from sglang.multimodal_gen.test.test_utils import (
    DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
)


class TestExternalModels(unittest.TestCase):
    def test_external_model(self):
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = (
            "sglang.multimodal_gen.test.external_models"
        )

        generator = DiffGenerator.from_pretrained(
            model_path=DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST,
            num_gpus=1,
            text_encoder_cpu_offload=True,
            pin_cpu_memory=True,
        )

        result = generator.generate(
            sampling_params_kwargs=dict(
                prompt="A cat sitting on a table",
                num_frames=17,
                height=480,
                width=720,
                num_inference_steps=3,
                seed=42,
            )
        )

        generator.shutdown()
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
