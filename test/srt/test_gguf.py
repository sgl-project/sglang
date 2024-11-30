import unittest

from huggingface_hub import hf_hub_download

import sglang as sgl


class TestGGUF(unittest.TestCase):
    def test_models(self):
        prompt = "Today is a sunny day and I like"
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        model_path = hf_hub_download(
            "Qwen/Qwen2-1.5B-Instruct-GGUF",
            filename="qwen2-1_5b-instruct-q4_k_m.gguf",
        )

        engine = sgl.Engine(model_path=model_path, random_seed=42)
        outputs = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        self.assertEqual(outputs, " it. I have a lot of work")


if __name__ == "__main__":
    unittest.main()
