import unittest

from sglang.test.chunked_prefill_test_utils import ChunkedRefactorTestBase
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestChunkedFeaturePriority(ChunkedRefactorTestBase):
    model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
    feature_args = [
        "--enable-priority-scheduling",
        # Keep concurrency moderate so the priority queue actually fills and
        # the early-exit-with-chunked-resume branches trigger.
        "--max-running-requests",
        "8",
        "--max-queued-requests",
        "128",
    ]


if __name__ == "__main__":
    unittest.main()
