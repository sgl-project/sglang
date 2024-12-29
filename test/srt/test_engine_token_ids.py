import unittest

from transformers import AutoTokenizer

import sglang as sgl


def get_test_engine():
    return sgl.Engine(
        model_path="meta-llama/Meta-Llama-3.1-8B-Instruct", return_token_ids=True
    )


class TestEngineTokenIds(unittest.TestCase):
    def test_token_ids_in_generate(self):
        llm = get_test_engine()
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3.1-8B-Instruct"
        )

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]
        sampling_params = {"temperature": 0.8, "top_p": 0.95}
        outputs = llm.generate(prompts, sampling_params)

        # Hugging Face tokenizer has a start token in its output,
        # while SGLang only adds next_token_id in output_ids.
        # We remove start token in output_ids for comparison.
        for prompt, output in zip(prompts, outputs):
            hf_input_ids = tokenizer.encode(prompt)
            self.assertEqual(
                output["input_ids"],
                hf_input_ids,
                f"Input token IDs mismatch for: {prompt}",
            )

            hf_output_ids = tokenizer.encode(output["text"])[1:]  # remove start token
            self.assertEqual(
                output["output_ids"],
                hf_output_ids,
                f"Output token IDs mismatch for: {output['text']}",
            )

            self.assertEqual(
                len(output["input_ids"]),
                output["meta_info"]["prompt_tokens"],
                "Prompt token count mismatch",
            )
            self.assertEqual(
                len(output["output_ids"]),
                output["meta_info"]["completion_tokens"],
                "Completion token count mismatch",
            )

        llm.shutdown()


if __name__ == "__main__":
    unittest.main()
