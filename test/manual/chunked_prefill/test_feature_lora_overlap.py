import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase


class TestChunkedFeatureLoRAOverlap(ChunkedTestBase):
    model = "meta-llama/Llama-2-7b-hf"
    feature_args = [
        "--enable-lora",
        "--enable-lora-overlap-loading",
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
