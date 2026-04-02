import unittest

import sglang as sgl
from sglang.srt.environ import temp_set_env
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")

TEST_GCS_MODEL = "gs://vertex-model-garden-public-us/codegemma/codegemma-2b/"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestRunaiModelLoader(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        with temp_set_env(
            GOOGLE_CLOUD_PROJECT="fake-project",
            RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS="true",
            CLOUD_STORAGE_EMULATOR_ENDPOINT="https://storage.googleapis.com",
        ):
            cls.engine = sgl.Engine(
                model_path=TEST_GCS_MODEL,
                load_format="runai_streamer",
                cuda_graph_max_bs=1,
                max_total_tokens=64,
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine:
            cls.engine.shutdown()

    def test_generate_produces_output(self):
        outputs = self.engine.generate(PROMPTS)
        self.assertEqual(len(outputs), len(PROMPTS))
        for i, output in enumerate(outputs):
            text = output["text"]
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0, f"Prompt {i} produced empty output")


if __name__ == "__main__":
    unittest.main()
