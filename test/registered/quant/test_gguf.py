import unittest

from huggingface_hub import hf_hub_download

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=96, suite="stage-b-test-small-1-gpu")


class TestGGUF(CustomTestCase):
    def test_models(self):
        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        model_path = hf_hub_download(
            "Qwen/Qwen2-1.5B-Instruct-GGUF",
            filename="qwen2-1_5b-instruct-q4_k_m.gguf",
        )

        engine = sgl.Engine(model_path=model_path, random_seed=42, cuda_graph_max_bs=2)
        outputs = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        self.assertEqual(outputs, " it. I have a lot of work")


if __name__ == "__main__":
    unittest.main()
