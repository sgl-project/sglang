import os
import unittest

from sglang import LLM, SamplingParams
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestLLMGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_name = DEFAULT_MODEL_NAME_FOR_TEST
        cls.prompts_list = [
            "Hello, my name is",
            "The capital of China is",
            "What is the meaning of life?",
            "The future of AI is",
        ]
        cls.single_prompt = "What is the meaning of life?"
        # Turn off tokernizers parallelism to enable running multiple tests
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def test_generate_with_sampling_params(self):
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        llm = LLM(model=self.model_name)
        outputs = llm.generate(self.prompts_list, sampling_params)

        self.assertEqual(len(outputs), len(self.prompts_list))
        for output in outputs:
            self.assertIn(output["index"], range(len(self.prompts_list)))
            self.assertTrue(output["text"].strip())

    def test_generate_without_sampling_params(self):
        llm = LLM(model=self.model_name)
        outputs = llm.generate(self.prompts_list)

        self.assertEqual(len(outputs), len(self.prompts_list))
        for output in outputs:
            self.assertIn(output["index"], range(len(self.prompts_list)))
            self.assertTrue(output["text"].strip())

    def test_generate_with_single_prompt_and_sampling_params(self):
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
        llm = LLM(model=self.model_name)
        outputs = llm.generate(self.single_prompt, sampling_params)

        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0]["text"].strip())

    def test_generate_with_single_prompt_without_sampling_params(self):
        llm = LLM(model=self.model_name)
        outputs = llm.generate(self.single_prompt)

        self.assertEqual(len(outputs), 1)
        self.assertTrue(outputs[0]["text"].strip())


if __name__ == "__main__":
    unittest.main()
