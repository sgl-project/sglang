import unittest

import sglang as sgl
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase


class TestSRTEngineWithQuantArgs(CustomTestCase):
    def test_1_quantization_args(self):

        # we only test fp8 because other methods are currently dependent on vllm. We can add other methods back to test after vllm dependency is resolved.
        quantization_args_list = [
            # "awq",
            "fp8",
            # "gptq",
            # "marlin",
            # "gptq_marlin",
            # "awq_marlin",
            # "bitsandbytes",
            # "gguf",
        ]

        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        for quantization_args in quantization_args_list:
            engine = sgl.Engine(
                model_path=model_path, random_seed=42, quantization=quantization_args
            )
            engine.generate(prompt, sampling_params)
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
