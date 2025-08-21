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

    def test_2_torchao_args(self):

        # we don't test int8dq because currently there is conflict between int8dq and capture cuda graph
        torchao_args_list = [
            # "int8dq",
            "int8wo",
            "fp8wo",
            "fp8dq-per_tensor",
            "fp8dq-per_row",
        ] + [f"int4wo-{group_size}" for group_size in [32, 64, 128, 256]]

        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        for torchao_config in torchao_args_list:
            engine = sgl.Engine(
                model_path=model_path, random_seed=42, torchao_config=torchao_config
            )
            engine.generate(prompt, sampling_params)
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
