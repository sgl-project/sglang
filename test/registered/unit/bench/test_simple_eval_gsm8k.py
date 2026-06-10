import json
import os
import tempfile
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.simple_eval_common import ChatCompletionSampler
from sglang.test.simple_eval_gsm8k import (
    CHAT_MODE_INSTRUCTION,
    GSM8KEval,
    extract_answer,
    get_chat_one_example,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


def _write_synthetic_dataset(path: str, n: int) -> None:
    with open(path, "w") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "question": f"Synthetic question {i}: what is {i} + {i}?",
                        "answer": f"The answer is {2 * i}. #### {2 * i}",
                    }
                )
                + "\n"
            )


class TestSimpleEvalGSM8K(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("OPENAI_API_KEY", "EMPTY")
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._data_path = os.path.join(cls._tmpdir.name, "synthetic.jsonl")
        _write_synthetic_dataset(cls._data_path, 16)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_extract_answer_prefers_explicit_answer_line(self):
        response = "Answer: 42\n\nQuestion: extra example?\nAnswer: 100"
        self.assertEqual(extract_answer(response), 42)

    def test_chat_sampler_accepts_stop_sequences(self):
        sampler = ChatCompletionSampler(
            base_url="http://127.0.0.1:1/v1",
            model="test-model",
            stop=["\nQuestion:", "\n\nQuestion:"],
        )
        self.assertEqual(sampler.stop, ["\nQuestion:", "\n\nQuestion:"])

    def test_chat_few_shot_formats_explicit_final_answer(self):
        lines = [
            {
                "question": "Synthetic question: what is 2 + 2?",
                "answer": "Add the two numbers. #### 4",
            }
        ]
        example = get_chat_one_example(lines, 0)
        self.assertIn("Reasoning: Add the two numbers.", example)
        self.assertIn("Answer: 4", example)

    def test_mistral_chat_prompt_uses_instruction_wrapper(self):
        evaluator = GSM8KEval(
            num_examples=1,
            num_threads=1,
            num_shots=2,
            data_path=self._data_path,
        )
        sampler = ChatCompletionSampler(
            base_url="http://127.0.0.1:1/v1",
            model="mistralai/Mistral-7B-Instruct-v0.3",
        )
        prompt = evaluator._build_prompt(
            0,
            "Question: Synthetic question 2: what is 2 + 2?\nAnswer:",
            sampler,
        )
        self.assertTrue(prompt.startswith(CHAT_MODE_INSTRUCTION))
        self.assertIn("Now solve this problem:", prompt)
        self.assertIn("Reasoning:", prompt)

    def test_non_mistral_chat_prompt_keeps_completion_style(self):
        evaluator = GSM8KEval(
            num_examples=1,
            num_threads=1,
            num_shots=2,
            data_path=self._data_path,
        )
        sampler = ChatCompletionSampler(
            base_url="http://127.0.0.1:1/v1",
            model="google/gemma-2-27b-it",
        )
        question = "Question: Synthetic question 2: what is 2 + 2?\nAnswer:"
        prompt = evaluator._build_prompt(0, question, sampler)
        self.assertFalse(prompt.startswith(CHAT_MODE_INSTRUCTION))
        self.assertTrue(prompt.endswith(question))


if __name__ == "__main__":
    unittest.main()
