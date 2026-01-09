import unittest

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


class TestExternalModels(CustomTestCase):
    def test_external_model(self):
        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("external_models")
            model_path=model_path,
            cuda_graph_max_bs=1,
            enable_multimodal=True,
        )
        out = engine.generate(prompt)["text"]
        engine.shutdown()

        self.assertGreater(len(out), 0)


if __name__ == "__main__":
    unittest.main()
