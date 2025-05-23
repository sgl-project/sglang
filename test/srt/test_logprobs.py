import unittest

from transformers import AutoTokenizer

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


class TestLogprobs(CustomTestCase):
    def test_logprobs(self):

        # send requests
        prompts = [
            "Hello, my name is",
            "The future of AI is",
            "The president of the United States is",
            "The capital of France is ",
            "The capital of China is",
            "The capital of Japan is",
            "The capital of Korea is",
        ]
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_ids = tokenizer(prompts).input_ids

        sampling_params = {
            "temperature": 0,
            "top_p": 0.9,
            "max_new_tokens": 32,
            "n": 1,
        }
        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            skip_tokenizer_init=True,
            # tp_size=2,
        )
        outputs = engine.generate(
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=3,
            token_ids_logprob=[1, 2, 3, 4, 5],
        )
        engine.shutdown()

        for output in outputs:
            len_input_ids = len(output["meta_info"]["input_token_logprobs"])
            self.assertEqual(
                len(output["meta_info"]["input_top_logprobs"]), len_input_ids
            )
            self.assertEqual(
                len(output["meta_info"]["input_token_ids_logprobs"]), len_input_ids
            )
            len_output_ids = len(output["output_ids"])
            self.assertEqual(
                len(output["meta_info"]["output_token_logprobs"]), len_output_ids
            )
            self.assertEqual(
                len(output["meta_info"]["output_top_logprobs"]), len_output_ids
            )
            self.assertEqual(
                len(output["meta_info"]["output_token_ids_logprobs"]), len_output_ids
            )

            for i in range(len_output_ids):
                output_token_logprobs = output["meta_info"]["output_token_logprobs"][i]
                assert isinstance(output_token_logprobs[0], float)
                assert output_token_logprobs[0] < 0

                output_top_logprobs = output["meta_info"]["output_top_logprobs"][i]
                assert isinstance(output_top_logprobs[0][0], float)
                assert output_top_logprobs[0][0] < 0

                output_token_ids_logprobs = output["meta_info"][
                    "output_token_ids_logprobs"
                ][i]
                assert isinstance(output_token_ids_logprobs[0][0], float)
                assert output_token_ids_logprobs[0][0] < 0


if __name__ == "__main__":
    unittest.main()
