"""The DeepSeek-V4.1-Flash servers that the monokernel serving test and
``test/manual/kernels/mono/validate.py`` launch: MI35x, TP4, with DSpark (verify
steps) or without it (decode steps)."""

import os

MODEL = os.environ.get(
    "DEEPSEEK_V41_FLASH_MODEL_PATH", "deepseek-ai/DeepSeek-V4.1-Flash"
)
ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_ROCM700A": "0",
    "AITER_BF16_FP8_MOE_BOUND": "0",
}
COMMON = [
    "--trust-remote-code",
    "--tp",
    "4",
    "--enforce-shared-experts-fusion",
    # K1 and K2 read the attention weights row-major; other backends reorder them
    "--fp8-gemm-backend",
    "aiter",
    "--mem-fraction-static",
    "0.7",
]
# Without --enable-monokernel, so a test can launch the normal path too.
# DSpark: verify steps of 6 rows a request, 8 requests at the largest width (48 rows)
ARGS = COMMON + [
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-dspark-block-size",
    "5",
    "--cuda-graph-max-bs-decode",
    "16",
    "--max-running-requests",
    "16",
]
# decode steps, a row a request, 48 at the largest width: 56 requests step past it
DECODE_ARGS = COMMON + [
    "--cuda-graph-max-bs-decode",
    "64",
    "--max-running-requests",
    "56",
]
