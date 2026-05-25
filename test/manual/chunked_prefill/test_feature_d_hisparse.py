"""Feature (d): HiSparse + chunked prefill.

HiSparse top-k selection only meaningfully kicks in when the prompt is
long enough that the sparse attention selects a strict subset of KV.
With ``top_k=2048`` (matches the existing CI test), we need prompts well
above 2048 tokens — ``LONG_PROMPT_NUM_SHOTS=24`` gives ~3000-4000 tokens
so chunked prefill + hisparse staging interactions are actually
exercised.

Server arg template borrowed from
``test/registered/8-gpu-models/test_dsa_models_hisparse.py::TestGLM5DPHiSparse``.

GPU requirement: 8 GPUs (H200 or equivalent; GLM-5-FP8 with TP=8 DP=8).

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest
from test.manual.chunked_prefill.common import (
    LONG_PROMPT_NUM_SHOTS,
    ChunkedRefactorTestBase,
)


class TestChunkedFeatureD_HiSparse(ChunkedRefactorTestBase):
    model = "zai-org/GLM-5-FP8"
    num_shots = LONG_PROMPT_NUM_SHOTS
    # GLM-5-FP8 generations can be longer; allow more headroom.
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
