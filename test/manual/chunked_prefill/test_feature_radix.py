"""radix cache prefix match + chunked prefill.

Mixed-prefix gsm8k is the perfect workload for this feature: mode 0 (25%
of requests) all share a single prefix, mode 1 (25%) shares within
clusters of 5, mode 2 (25%) is fully unique, mode 3 (25%) has no prefix.
The radix cache hit / miss / branching paths all run within a single
fixture.

Server arg template borrowed from
``test/registered/radix_cache/test_radix_attention.py::TestRadixCacheFCFS``.

GPU requirement: 1 small GPU.

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest
from test.manual.chunked_prefill.common import ChunkedRefactorTestBase

from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeatureRadix(ChunkedRefactorTestBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    feature_args = [
        "--max-total-tokens",
        "20000",
        "--schedule-policy",
        "fcfs",
    ]


if __name__ == "__main__":
    unittest.main()
