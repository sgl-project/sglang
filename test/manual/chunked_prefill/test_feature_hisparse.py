import unittest

from sglang.test.chunked_prefill_test_utils import (
    LONG_PROMPT_NUM_SHOTS,
    ChunkedRefactorTestBase,
)


class TestChunkedFeatureHiSparse(ChunkedRefactorTestBase):
    model = "deepseek-ai/DeepSeek-V4-Flash"
    num_shots = LONG_PROMPT_NUM_SHOTS
    # DSV4-Flash generations can be longer; allow more headroom.
    max_tokens = 4000
    feature_args = [
        "--trust-remote-code",
        "--tp",
        "8",
        "--dp",
        "8",
        "--enable-dp-attention",
        "--page-size",
        "64",
        "--max-running-requests",
        "200",
        "--mem-fraction-static",
        "0.85",
        "--disable-radix-cache",
        "--kv-cache-dtype",
        "bfloat16",
        "--dsa-decode-backend",
        "flashmla_sparse",
        "--enable-hisparse",
        "--hisparse-config",
        '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 5}',
        "--model-loader-extra-config",
        '{"enable_multithread_load": true, "num_threads": 64}',
    ]


if __name__ == "__main__":
    unittest.main()
