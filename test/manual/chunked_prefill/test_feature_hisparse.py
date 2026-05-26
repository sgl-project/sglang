"""HiSparse + chunked prefill.

HiSparse top-k selection only meaningfully kicks in when the prompt is
long enough that the sparse attention selects a strict subset of KV.
With ``top_k=2048`` (the existing CI default) we need prompts well above
2048 tokens — ``LONG_PROMPT_NUM_SHOTS=24`` gives ~3000-4000 tokens so
chunked prefill + hisparse staging interactions are actually exercised.

Reference config sources:
  - HiSparse flag set / ``--hisparse-config`` JSON:
    ``test/registered/8-gpu-models/test_dsa_models_hisparse.py::TestGLM5DPHiSparse``
  - DeepSeek-V4-Flash model id + launch style:
    ``test/manual/dsv4/test_b200_flash.py`` (`MODEL` constant)

We use DeepSeek-V4-Flash here (instead of GLM-5-FP8) because it is the
codebase's primary first-party hisparse target — the ``is_dsv4_model``
branch in ``model_runner_kv_cache_mixin.py:752`` and the v4 fast path in
``hisparse_hook.py:60`` are only exercised on DSV4 models.

GPU requirement: 8 GPUs (H200 / B200 class; DSV4-Flash with TP=8 DP=8).

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

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
