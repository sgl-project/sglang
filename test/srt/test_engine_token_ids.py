import unittest

import sglang as sgl


class TestEngineTokenIds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm = sgl.Engine(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
        cls.prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        cls.sampling_params = {"temperature": 0.8, "top_p": 0.95}

    @classmethod
    def tearDownClass(cls):
        cls.llm.shutdown()

    def test_token_ids_in_generate(self):
        outputs = self.llm.generate(self.prompts, self.sampling_params)

        for prompt, output in zip(self.prompts, outputs):
            # Test output contains token IDs
            self.assertIn("input_ids", output)
            self.assertIn("output_ids", output)

            # Test meta info contains token counts
            self.assertIn("prompt_tokens", output["meta_info"])
            self.assertIn("completion_tokens", output["meta_info"])

            # Verify token counts match array lengths
            self.assertEqual(
                len(output["input_ids"]), output["meta_info"]["prompt_tokens"]
            )
            self.assertEqual(
                len(output["output_ids"]), output["meta_info"]["completion_tokens"]
            )

            # Verify completion length
            self.assertEqual(output["meta_info"]["completion_tokens"], 128)


if __name__ == "__main__":
    unittest.main()
