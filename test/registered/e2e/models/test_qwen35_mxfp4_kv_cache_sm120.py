"""End-to-end MXFP4 KV cache serving test on SM120.

Boots Llama-3.1-8B-Instruct with --kv-cache-dtype mxfp4 (FlashInfer BF16 PLAIN
prefill + native triton MXFP4 decode, CUDA graph enabled) and runs the shared
basic-decode-correctness probes, which target exactly the KV/attention
corruption and cuda-graph edge cases this serving path could introduce. Codec
and kernel bit-exactness are covered by their own tests; this exercises the
full wiring (CLI -> hook -> configurator -> pool -> backend -> kernel).
"""

import unittest

from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=200, stage="extra-a", runner_config="1-gpu-large")


@unittest.skipUnless(
    is_sm120_supported(), "MXFP4 native decode requires an SM120 GPU (CUDA 12.8+)"
)
class TestLlama8BMxfp4KVCacheSM120(BasicDecodeCorrectnessMixin, DefaultServerBase):
    """Llama-3.1-8B-Instruct with MXFP4 KV cache: FlashInfer PLAIN prefill +
    native triton MXFP4 decode on SM120."""

    model = DEFAULT_MODEL_NAME_FOR_TEST
    other_args = [
        "--kv-cache-dtype",
        "mxfp4",
        "--prefill-attention-backend",
        "flashinfer",
        "--decode-attention-backend",
        "triton",
    ]


if __name__ == "__main__":
    unittest.main()
