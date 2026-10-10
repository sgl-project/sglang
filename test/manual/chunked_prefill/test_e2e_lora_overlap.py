import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase


class TestChunkedFeatureLoRAOverlap(ChunkedTestBase):
    __test__ = True
    model = "meta-llama/Llama-3.2-1B-Instruct"
    gsm8k_threshold = 0.20
    feature_args = [
        "--enable-lora",
        "--enable-lora-overlap-loading",
        "--lora-paths",
        "nicoboss/Llama-3.2-1B-Instruct-Uncensored-Lora",
        "codelion/Llama-3.2-1B-Instruct-tool-calling-lora",
        "--max-loras-per-batch",
        "2",
        "--max-loaded-loras",
        "4",
        "--mem-fraction-static",
        "0.75",
    ]


if __name__ == "__main__":
    unittest.main()
