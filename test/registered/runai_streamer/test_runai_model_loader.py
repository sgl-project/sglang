import unittest

import sglang as sgl
from sglang.srt.environ import temp_set_env
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, suite="stage-b-test-small-1-gpu")

runai_load_format = "runai_streamer"
test_gcs_model = "gs://vertex-model-garden-public-us/codegemma/codegemma-2b/"

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


class TestRunaiModelLoader(unittest.TestCase):
    def test_runai_model_loader(self):

        with temp_set_env(
            GOOGLE_CLOUD_PROJECT="fake-project",
            RUNAI_STREAMER_GCS_USE_ANONYMOUS_CREDENTIALS="true",
            CLOUD_STORAGE_EMULATOR_ENDPOINT="https://storage.googleapis.com",
        ):

            print("building engine with load_format", runai_load_format)

            engine = sgl.Engine(
                model_path=test_gcs_model,
                load_format=runai_load_format,
                cuda_graph_max_bs=1,
                max_total_tokens=64,
                enable_multimodal=True,
            )
            outputs = engine.generate(prompts)

            engine.shutdown()

            self.assertGreater(len(outputs), 0)


if __name__ == "__main__":
    unittest.main()
