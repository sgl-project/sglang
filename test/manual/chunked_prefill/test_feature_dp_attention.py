"""DP attention + chunked prefill.

DP attention runs per-rank scheduler instances synchronized via DP
broadcast. The interaction with chunked prefill lives in the per-rank
state machine for chunked requests; if scheduler state diverges between
DP ranks the resulting collective ops hang.

Server arg template borrowed from
``test/registered/dp_attn/test_dp_attention.py::TestDPAttentionMixedChunk``.

GPU requirement: 2 GPUs (tp=2 + dp=2 over the same 2 ranks).

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest
from test.manual.chunked_prefill.common import ChunkedRefactorTestBase

from sglang.test.test_utils import DEFAULT_MLA_MODEL_NAME_FOR_TEST


class TestChunkedFeatureDPAttention(ChunkedRefactorTestBase):
    model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
    feature_args = [
        "--trust-remote-code",
        "--tp",
        "2",
        "--enable-dp-attention",
        "--dp",
        "2",
        "--enable-mixed-chunk",
    ]


if __name__ == "__main__":
    unittest.main()
