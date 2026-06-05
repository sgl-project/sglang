import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase


class TestChunkedFeatureLoRA(ChunkedTestBase):
    __test__ = True  # re-enable: the shared base sets __test__ = False
    model = "meta-llama/Llama-2-7b-hf"
    gsm8k_threshold = 0.50
    feature_args = [
        "--enable-lora",
        "--lora-paths",
        "winddude/wizardLM-LlaMA-LoRA-7B",
        "RuterNorway/Llama-2-7b-chat-norwegian-LoRa",
        "--max-loras-per-batch",
        "2",
        "--max-loaded-loras",
        "4",
        "--mem-fraction-static",
        "0.75",
    ]


if __name__ == "__main__":
    unittest.main()
