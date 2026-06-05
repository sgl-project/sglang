import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedTestBase


class TestChunkedFeatureSpec(ChunkedTestBase):
    __test__ = True  # re-enable: the shared base sets __test__ = False
    # kv-canary cannot instrument the eagle draft decode's topk expansion
    # (positions are bs*topk vs the canary's bs-sized plan); same opt-out as
    # the scripted TestSpecBasic.
    use_kv_canary = False
    # The same proven EAGLE3 stack as the scripted TestSpecBasic: the legacy
    # EAGLE(v1) Llama-2 pairing hits an illegal memory access under chunked
    # prefill on this image and Llama-2-7b-chat cannot clear a gsm8k bar anyway.
    model = "Qwen/Qwen3-8B"
    gsm8k_threshold = 0.50
    feature_args = [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        "Tengyunw/qwen3_8b_eagle3",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "8",
        "--speculative-num-draft-tokens",
        "64",
        "--mem-fraction-static",
        "0.7",
    ]


if __name__ == "__main__":
    unittest.main()
