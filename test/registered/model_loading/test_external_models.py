import unittest

import sglang as sgl
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=45, suite="stage-b-test-small-1-gpu-amd")


class TestExternalModels(CustomTestCase):
    def test_external_model(self):
        envs.SGLANG_EXTERNAL_MODEL_PACKAGE.set("sglang.test.external_models")
        envs.SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE.set("sglang.test.external_models")
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
