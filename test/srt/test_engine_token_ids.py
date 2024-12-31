import unittest

from transformers import AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestEngineTokenIds(unittest.TestCase):
    def test_token_ids_in_generate(self):
        llm = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST, return_token_ids=True
        )
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_SMALL_MODEL_NAME_FOR_TEST)

        prompts = [
            "Hello, my name is",
            "The president of the United States is",
            "The capital of France is",
            "The future of AI is",
        ]

        sampling_params = {"temperature": 0, "top_p": 0.95}
        outputs = llm.generate(prompts, sampling_params)

        for prompt, output in zip(prompts, outputs):
            deocode_input = tokenizer.decode(
                output["input_ids"], skip_special_tokens=True
            )
            assert (deocode_input in prompt) or (
                prompt in deocode_input
            ), f"Decode input: {deocode_input} mismatch for: {prompt}"

            deocode_output = tokenizer.decode(
                output["output_ids"], skip_special_tokens=True
            )
            assert (deocode_output in output["text"]) or (
                output["text"] in deocode_output
            ), f"Decode output: {deocode_output} mismatch for: {output['text']}"

        llm.shutdown()


if __name__ == "__main__":
    unittest.main()
