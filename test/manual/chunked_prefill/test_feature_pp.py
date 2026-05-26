"""pipeline parallelism + dynamic chunking + chunked prefill.

Manual fixture for the chunked-prefill refactor accuracy net. Combines
``--pp-size 2 --tp-size 2`` (template borrowed from
``test/registered/pp/test_pp_single_node.py::TestPPAccuracy``) with
``--enable-dynamic-chunking`` and a forced small ``--chunked-prefill-size``.

GPU requirement: 4 GPUs (TP=2 × PP=2). The PP-last-chunk-in-flight scenario
(the original motivator for chunked_last_in_flight in the refactor) only
manifests when PP>=2.

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedRefactorTestBase
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePP(ChunkedRefactorTestBase):
    model = DEFAULT_MODEL_NAME_FOR_TEST
    feature_args = [
        "--tp-size",
        "2",
        "--pp-size",
        "2",
        "--enable-dynamic-chunking",
    ]


if __name__ == "__main__":
    unittest.main()
