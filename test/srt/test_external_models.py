import os
import unittest

import sglang as sgl
from sglang.test.test_utils import CustomTestCase


class TestExternalModels(CustomTestCase):
    def test_external_model(self):
        os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "external_models"
        os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] = "external_models"
        prompt = "Today is a sunny day and I like"
        model_path = "Qwen/Qwen2-VL-2B-Instruct"

        engine = sgl.Engine(
            model_path=model_path,
            cuda_graph_max_bs=1,
            max_total_tokens=64,
            enable_multimodal=True,
        )
        out = engine.generate(prompt)["text"]
        engine.shutdown()

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
