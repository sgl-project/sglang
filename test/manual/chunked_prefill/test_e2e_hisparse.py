import unittest

from sglang.test.chunked_prefill_test_utils import (
    LONG_PROMPT_NUM_SHOTS,
    ChunkedTestBase,
)


class TestChunkedFeatureHiSparse(ChunkedTestBase):
    __test__ = True  # re-enable: the shared base sets __test__ = False
    model = "deepseek-ai/DeepSeek-V4-Flash"
    num_shots = LONG_PROMPT_NUM_SHOTS
    max_tokens = 4000
    gsm8k_threshold = 0.50
    feature_args = [
        "--trust-remote-code",
        "--tp",
        "4",
        "--moe-runner-backend",
        "flashinfer_mxfp4",
        "--disable-radix-cache",
        "--enable-hisparse",
        "--hisparse-config",
        '{"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 5}',
    ]


if __name__ == "__main__":
    unittest.main()
