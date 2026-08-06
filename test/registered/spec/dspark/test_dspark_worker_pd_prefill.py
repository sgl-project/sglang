import unittest
from unittest.mock import Mock

from sglang.srt.speculative.dspark_components.dspark_worker_v2 import (
    DSparkWorkerV2,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDsparkWorkerPdPrefill(CustomTestCase):
    def test_shared_decode_modules_are_not_attached(self):
        """PD prefill must not configure the unused Markov/LM-head TP shard."""
        worker = object.__new__(DSparkWorkerV2)
        worker._is_pd_prefill = True
        worker.draft_model = Mock()

        worker._attach_shared_modules()

        worker.draft_model.attach_shared_modules.assert_not_called()


if __name__ == "__main__":
    unittest.main()
