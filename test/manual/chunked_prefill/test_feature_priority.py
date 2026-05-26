"""priority scheduling + chunked prefill.

The early-exit guards in the chunked admission loop have explicit
``not self.enable_priority_preemption`` branches (see scheduler.py
2530-2536); this fixture exercises the priority-scheduling code path
while chunking is forced. The mixed-prefix workload's 100 concurrent
requests is enough to create queue pressure under a tight
max-running-requests.

Server arg template borrowed from
``test/registered/scheduler/test_priority_scheduling.py``.

GPU requirement: 1 small GPU.

Not registered with CI. Run by hand from
``test/manual/chunked_prefill/``.
"""

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
